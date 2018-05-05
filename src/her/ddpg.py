from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from her import logger
from her.utils.misc import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from her.normalizer import Normalizer
from her.replay_buffer import ReplayBuffer
from her.utils.mpi_adam import MpiAdam
import her.constants as C


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, gg_k, replay_strategy, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        self.replay_strategy = replay_strategy

        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            self.max_g = kwargs['max_g']
            self.d0 = kwargs['d0']
            self.slope = kwargs['slope']
            self.goal_lr = kwargs['goal_lr']
            # reward shaping parameters
            self.rshape_lambda = kwargs['rshape_lambda']
            self.reshape_p = kwargs['rshape_p']
            self.rshaping = kwargs['rshaping']

            self.input_dims['e'] = self.dimg * self.T
            self.input_dims['mask'] = self.T
            self.dime = self.input_dims['e']
            self.dim_mask = self.input_dims['mask']

        input_shapes = dims_to_shapes(self.input_dims)

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)

        if self.replay_strategy in [C.REPLAY_STRATEGY_BEST_K, C.REPLAY_STRATEGY_GEN_K, C.REPLAY_STRATEGY_GEN_K_GMM]:
            buffer_shapes['gg'] = (self.T, self.gg_k, self.dimg)

        if self.replay_strategy in [C.REPLAY_STRATEGY_BEST_K, C.REPLAY_STRATEGY_GEN_K_GMM]:
            buffer_shapes['gg_idx'] = (self.T, self.gg_k)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preprocess_e(self, e):
        e = np.clip(e, -self.clip_obs, self.clip_obs)
        return e

    # def td_error(self, o, g):
    #     vals = [self.Q_loss_tf]
    def get_target_q_val(self, o, ag, g):
        vals = [self.target.Q_pi_tf]
        feed = {
            self.target.o_tf: o.reshape(-1, self.dimo),
            self.target.g_tf: g.reshape(-1, self.dimg)
        }
        ret = self.sess.run(vals, feed_dict=feed)
        return ret[0]

    def get_goals(self, u_goal, e, mask, use_target_net=False):
        """
        :param u_goal: batch_size * dim_u dimensional array
        :param e: batch_size * (T*dim_g) dimensional array
        :param mask: batch_size * T dimensional array
        :param use_target_net: True/False
        :return:
        """
        e = self._preprocess_e(e)
        policy = self.target if use_target_net else self.main
        vals = [policy.goal_tf, policy.distance, policy.e_reshaped, policy.goal_tf_repeated, policy.reward_sum]
        # feed
        feed = {
            policy.e_tf: e.reshape(-1, self.dime),
            policy.mask_tf: mask.reshape(-1, self.dim_mask),
            policy.u_tf: u_goal.reshape(-1, self.dimu)
        }
        ret = self.sess.run(vals, feed_dict=feed)
        # print("Generated goal: ")
        # print("Goal: ", ret[0])
        # print("Distance: ", ret[1])
        # print("Episode: ", ret[2])
        # print("Goal repeated: ", ret[3])
        # print("Reward: ", np.average(ret[4]))
        # print('---------------------------------------------------------------')
        # for var in self._vars('main/goal'):
        #     print("Name: " + var.name)
        #     print("Shape: " + str(var.shape))
        #     print(var.eval())
        return ret[0]

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

            if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
                e = transitions['e']
                transitions['e'] = self._preprocess_e(e)
                self.e_stats.update(transitions['e'])
                self.e_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            self.goal_adam.sync()
            self.Q_goal_adam.sync()
            self.pi_goal_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        tf_list = [self.Q_loss_tf, self.main.Q_pi_tf, self.Q_grad_tf, self.pi_grad_tf]
        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            tf_list.extend([self.goal_loss_tf, self.goal_grad_tf])
            tf_list.extend([self.Q_goal_loss_tf, self.Q_goal_grad_tf])
            tf_list.extend([self.pi_goal_loss_tf, self.pi_goal_grad_tf])
            tf_list.extend([self.main.mask_tf, self.main.d])
        return self.sess.run(tf_list)

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def _update_goal(self, goal_grad, Q_goal_grad, pi_goal_grad):
        # self.Q_goal_adam.update(Q_goal_grad, self.Q_lr)
        # self.pi_goal_adam.update(pi_goal_grad, self.pi_lr)
        self.goal_adam.update(goal_grad, self.goal_lr)

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)
        return self.batch_from_transitions(transitions)

    def batch_from_transitions(self, transitions):
        """
            transitions is a dictionary with keys: ['o', 'ag', 'u', 'o_2', 'ag_2', 'r', 'g']
            batch is a processed batch (normalizing, clipping, relative goal) for staging,
                and has the keys ['o', 'ag', 'u', 'o_2', 'ag_2', 'r', g', 'g_2']
        """
        # preprocess observations and goals
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            e = transitions['e']
            transitions['e'] = self._preprocess_e(e)

        # Set the correct order of keys in the batch
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        # print("*********************************Training*******************************")
        if stage:
            self.stage_batch()

        if self.replay_strategy != C.REPLAY_STRATEGY_GEN_K:
            critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        else:
            critic_loss, actor_loss, Q_grad, pi_grad,\
            goal_loss, goal_grad, Q_goal_loss, Q_goal_grad, \
            pi_goal_loss, pi_goal_grad, x, y = self._grads()
            self._update_goal(goal_grad, Q_goal_grad, pi_goal_grad)

        self._update(Q_grad, pi_grad)
        # print("Loss: ", goal_loss)
        # print("mask: ", np.sum(x, axis=1))
        # print("distance: ", y)
        # print("Reward: ", r)

        # if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
        #     goal_loss = self.sess.run(self.target_Q_goal_tf)
        #     # self.goal_adam.update(goal_grad, self.goal_lr)
        #     print("Goal loss: ", goal_loss)

        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            self.sess.run(self.copy_normal_to_goal_op)

        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            # running averages
            with tf.variable_scope('e_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.e_stats = Normalizer(self.dime, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf_vec = tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf)
        self.Q_loss_tf = tf.reduce_mean(self.Q_loss_tf_vec)
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')

        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            # loss functions for goal generation network
            target_Q_pi_goal_tf = self.target.Q_pi_goal_tf
            target_goal_tf = tf.clip_by_value(self.main.reward + self.gamma * target_Q_pi_goal_tf, *clip_range)
            self.goal_loss_tf = -self.LAMBDA * tf.reduce_mean(tf.square(tf.stop_gradient(target_goal_tf) - self.main.Q_goal_tf))
            # self.goal_loss_tf += 0.0 * tf.reduce_mean(tf.square(self.main.goal_tf / self.max_g))
            # self.goal_loss_tf = 0
            # self.reward_sum = tf.reduce_mean(self.main.reward_sum)
            self.goal_loss_tf += -tf.reduce_mean(self.main.reward_sum)

            # loss functions for Q_goal and pi_goal
            self.Q_goal_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_goal_tf) - self.main.Q_goal_tf))
            self.pi_goal_loss_tf = -tf.reduce_mean(self.main.Q_pi_goal_tf)
            self.pi_goal_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_goal_tf / self.max_u))

            # gradients
            goal_grads_tf = tf.gradients(self.goal_loss_tf, self._vars('main/goal'))
            self.goal_grad_tf = flatten_grads(grads=goal_grads_tf, var_list=self._vars('main/goal'))

            Q_goal_grads_tf = tf.gradients(self.Q_goal_loss_tf, self._vars('main/gQ'))
            self.Q_goal_grad_tf = flatten_grads(grads=Q_goal_grads_tf, var_list=self._vars('main/gQ'))

            pi_goal_grads_tf = tf.gradients(self.pi_goal_loss_tf, self._vars('main/gpi'))
            self.pi_goal_grad_tf = flatten_grads(grads=pi_goal_grads_tf, var_list=self._vars('main/gpi'))

            assert len(self._vars('main/goal')) == len(goal_grads_tf)
            assert len(self._vars('main/gQ')) == len(Q_goal_grads_tf)
            assert len(self._vars('main/gpi')) == len(pi_goal_grads_tf)

            # optimizers
            self.goal_adam = MpiAdam(self._vars('main/goal'), scale_grad_by_procs=False)
            self.Q_goal_adam = MpiAdam(self._vars('main/gQ'), scale_grad_by_procs=False)
            self.pi_goal_adam = MpiAdam(self._vars('main/gpi'), scale_grad_by_procs=False)

            self.main_vars += self._vars('main/goal') + self._vars('main/gQ') + self._vars('main/gpi')
            self.target_vars += self._vars('target/goal') + self._vars('target/gQ') + self._vars('target/gpi')
            self.stats_vars += self._global_vars('e_stats')

            self.normal_vars = self._vars('main/Q') + self._vars('main/pi') + self._vars('target/Q') + self._vars('target/pi')
            self.goal_vars = self._vars('main/gQ') + self._vars('main/gpi') + self._vars('target/gQ') + self._vars('target/gpi')

            self.copy_normal_to_goal_op = list(
                map(lambda v: v[0].assign(0 * v[0] + 1 * v[1]), zip(self.goal_vars, self.normal_vars)))

        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
