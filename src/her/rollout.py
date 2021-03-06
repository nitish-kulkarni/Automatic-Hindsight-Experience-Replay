from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException
import pdb

from her.utils.misc import convert_episode_to_batch_major, store_args
import her.constants as C


class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, gg_k=1, reward_fun=None, d0=0.05,
                 replay_strategy=C.REPLAY_STRATEGY_FUTURE,
                 **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]
        self.replay_strategy = replay_strategy

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            self.tstmp_indices , self.episode_indices = self.prepare_ep_tstmp_indices()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())

            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(
            o=obs,
            u=acts,
            g=goals,
            ag=achieved_goals
        )

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        ## Generated Goals for HER #
        batch_major_episode = convert_episode_to_batch_major(episode)

        if self.replay_strategy == C.REPLAY_STRATEGY_BEST_K:
            gg, gg_idx = self.heuristic_top_k_goals(batch_major_episode)
            batch_major_episode['gg'] = gg
            batch_major_episode['gg_idx'] = gg_idx

        elif self.replay_strategy == C.REPLAY_STRATEGY_GEN_K_GMM:
            gg, gg_idx = self.heuristic_top_k_goals(batch_major_episode, noise=True)
            batch_major_episode['gg'] = gg
            batch_major_episode['gg_idx'] = gg_idx

        elif self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            # print("True achieved goal: ")
            # print(batch_major_episode['ag'])
            e, mask = get_ggn_input(batch_major_episode['ag'][self.episode_indices, :self.T, :], self.tstmp_indices)
            u_goal = batch_major_episode['u'][self.episode_indices, self.tstmp_indices, :]
            goal_output = self.policy.get_goals(u_goal, e, mask, use_target_net=self.use_target_net)
            batch_major_episode['gg'] = self.prepare_gg_gen_k(np.array(episode['ag']).shape[-1], goal_output, mask)
            batch_major_episode['e'] = np.reshape(e, (self.rollout_batch_size, self. T, -1))
            batch_major_episode['mask'] = np.reshape(mask, (self.rollout_batch_size, self. T, -1))

        return batch_major_episode

    def prepare_gg_gen_k(self, d, goal_output, mask):
        goal_output = np.reshape(goal_output, (self.rollout_batch_size, self.T, d))
        goal_output_flat = np.reshape(goal_output, (self.rollout_batch_size, self.T * d))
        mask = np.reshape(mask, (self.rollout_batch_size, self.T, self.T))
        all_goals = np.zeros((self.rollout_batch_size, self.T, self.T * d))
        for i in range(self.rollout_batch_size):
            for t in range(self.T):
                all_goals[i, t, :(self.T - t) * d] = np.tile(goal_output[i, t, :][np.newaxis, :], (1, np.sum(mask[i, t, :])))
                all_goals[i, t, (self.T - t) * d:] = np.reshape(np.reshape(goal_output_flat[i, :t * d], (t, d))[::-1], (t * d))
        all_goals = np.reshape(all_goals, (self.rollout_batch_size, self.T, self.T, d))
        return all_goals[:, :, :self.gg_k, :]

    def prepare_ep_tstmp_indices(self):
        tidx = []
        eidx = []
        for i in range(self.rollout_batch_size):
            tidx += list(range(self.T))
            eidx += [i] * self.T
        tstmp_indices = np.array(tidx)
        episode_indices = np.array(eidx)
        return tstmp_indices, episode_indices

    def heuristic_top_k_goals(self, episode, noise=False):
        shapes = dict([(key, value.shape) for key, value in episode.items()])

        # Initialize generated goals as the last achieved goal
        gg_shape = (self.rollout_batch_size, self.T, self.gg_k, shapes['ag'][-1])
        gg = np.tile(episode['ag'][:, -1, :], (1, self.gg_k * self.T)).reshape(gg_shape)
        ag = episode['ag'].copy()

        gg_idx = np.ones((self.rollout_batch_size, self.T, self.gg_k)) * (self.T - 1)
        for t in range(self.T):
            if noise:
                sd = self.d0 / (np.sqrt(shapes['ag'][-1]))
                episode['ag'] = ag + np.random.normal(0.0, sd, ag.shape)
            # Create a single "forward pass batch" from all future transitions
            future_transitions_t = self.future_transitions(episode, shapes, t)
            batch_transitions = self.policy.batch_from_transitions(future_transitions_t)

            # Compute TD Errors for each transition
            self.policy.stage_batch(batch=batch_transitions)
            td_errors = self.policy.sess.run([self.policy.Q_loss_tf_vec])[0].reshape(self.rollout_batch_size, self.T - t)

            if np.isnan(td_errors).any():
                self.logger.warning('NaN caught during td_error computation for goal selection. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            # Assign the goals from top k TD errors as generated goals
            top_k_indices = td_errors.argsort(axis=1)[:, - self.gg_k:]
            gg_size = top_k_indices.shape[-1]
            future_goals = future_transitions_t['g'].reshape((self.rollout_batch_size, self.T - t, shapes['ag'][-1]))
            for b_idx in range(self.rollout_batch_size):
                gg[b_idx, t, :gg_size, :] = future_goals[b_idx, top_k_indices[b_idx]]
                gg_idx[b_idx, t, :gg_size] = top_k_indices[b_idx] + t

        if noise:
            episode['ag'] = ag

        return gg, gg_idx

    def future_transitions(self, episode, shapes, t):
        n_t = self.T - t
        b = self.rollout_batch_size
        transitions = {}

        for key in ['o', 'ag', 'u']:
            transitions[key] = np.tile(episode[key][:,t,:], (1, n_t)).reshape((b * n_t, shapes[key][-1]))
        for key in ['o', 'ag']:
            transitions[key + '_2'] = np.tile(episode[key][:,t + 1,:], (1, n_t)).reshape((b * n_t, shapes[key][-1]))

        transitions['g'] = episode['ag'][:,(t + 1):,:].reshape(b * n_t, shapes['ag'][-1])

        # Compute rewards for the new goals
        info = {}
        for key, value in episode.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = self.reward_fun(**reward_params)
        return transitions

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)


def get_ggn_input(ag, tstmp_indices):
    """
    :param o: batch_size * T * dim_{ag} dimensional array
    (batch_size = rollout_batch_size*T OR (mini) batch_size for training)
    :param tstmp_indices: batch_size dimensional array containing timestamps of transitions
    :return: e, mask: where e is batch_size * (T*dim_{ag}) matrix and mask is batch_size * T matrix
    """
    batch_size, T, d = ag.shape
    e = np.reshape(ag, (batch_size, T*d))
    mask = np.array([[0] * i + [1] * (T - i) for i in tstmp_indices.tolist()])
    assert e.shape == (batch_size, T*d)
    assert mask.shape == (batch_size, T)
    return e, mask
