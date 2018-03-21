#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import gym
import sys
import argparse
import os
from gym import wrappers
import random
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


class PolicyNetwork:
    """Actor Network class. Input: state. Output: action (continuous)"""

    def __init__(self, param, scope='PolicyNetwork'):
        self.dim_s = param['dim_s']
        self.dim_a = param['dim_a']
        self.gamma = param['gamma']
        self.summary_dir = '%s_%s' % (param['tf_summary_dir'], scope)
        self.sess = param['sess']
        self.model_dir = '%s_%s' % (param['model_dir'], scope)
        self.scope = scope
        self.h = param['h']

        # All placeholders
        self.s = None
        self.q_grad = None
        self.lr = None

        # Derived tensors
        self.mu = None
        self.train_op = None
        self.summaries = None

        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        with tf.variable_scope(self.scope):
            self.build_model()
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        self.saver = tf.train.Saver(max_to_keep=101)

    def build_model(self):
        lambda_reg = 1e-3
        regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_reg)

        self.s = tf.placeholder(name="s", shape=[None, self.dim_s], dtype=tf.float32)
        self.q_grad = tf.placeholder(name="q_grad", shape=[None, self.dim_a], dtype=tf.float32)
        self.lr = tf.placeholder(name="lr", shape=[], dtype=tf.float32)

        x = self.s
        for _ in range(3):
            x = tf.layers.dense(x, self.h, activation=tf.nn.relu, kernel_regularizer=regularizer)
        self.mu = tf.layers.dense(x, self.dim_a, activation=tf.nn.tanh, kernel_regularizer=regularizer, use_bias=False)

        trainable_vars = tf.trainable_variables(scope=self.scope)
        mu_grad = tf.gradients(ys=self.mu, xs=trainable_vars, grad_ys=-self.q_grad)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(mu_grad, trainable_vars))

    def save_model(self):
        global_step = tf.train.global_step(self.sess, self.global_step_tensor)
        self.saver.save(self.sess, self.model_dir, global_step=global_step)

    @staticmethod
    def load_model(saver, sess, model_file):
        saver.restore(sess, model_file)

    def predict(self, s):
        mu = self.sess.run(self.mu, feed_dict={self.s: s})
        if mu.shape[0] == 1:
            mu = tf.reshape(mu, [-1])
            return mu.eval(session=self.sess)
        else:
            return mu

    def update(self, samples, lr, qnet, save_graph=False):
        state, action, reward, next_state, done = zip(*samples)
        mu = self.predict(state)
        q_grad = qnet.get_q_grad(state, mu)

        feed_dict = {self.s: state, self.lr: lr, self.q_grad: q_grad}
        self.sess.run([self.train_op], feed_dict=feed_dict)

        if save_graph:
            self.summary_writer.add_graph(self.sess.graph)


class QNetwork:
    """Critic Network class. Input: state, action. Output: Q-value"""

    def __init__(self, param, scope='QNetwork'):
        self.dim_s = param['dim_s']
        self.dim_a = param['dim_a']
        self.gamma = param['gamma']
        self.summary_dir = '%s_%s' % (param['tf_summary_dir'], scope)
        self.sess = param['sess']
        self.model_dir = '%s_%s' % (param['model_dir'], scope)
        self.scope = scope
        self.h = param['h']

        # All placeholders
        self.s = None
        self.q_target = None
        self.a = None
        self.lr = None
        self.q = None

        # Derived tensors
        self.loss = None
        self.train_op = None
        self.q_grad = None
        self.summaries = None

        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        with tf.variable_scope(self.scope):
            self.build_model()
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        self.saver = tf.train.Saver(max_to_keep=101)

    def build_model(self):
        lambda_reg = 1e-3
        regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_reg)

        self.s = tf.placeholder(name="s", shape=[None, self.dim_s], dtype=tf.float32)
        self.q_target = tf.placeholder(name="q_target", shape=[None, 1], dtype=tf.float32)
        self.a = tf.placeholder(name="a", shape=[None, self.dim_a], dtype=tf.float32)
        self.lr = tf.placeholder(name="lr", shape=[], dtype=tf.float32)

        x = tf.concat([self.s, self.a], axis=1)
        for _ in range(3):
            x = tf.layers.dense(x, self.h, activation=tf.nn.relu, kernel_regularizer=regularizer)

        self.q = tf.layers.dense(x, 1, kernel_regularizer=regularizer)
        self.q_grad = tf.gradients(self.q, self.a)

        self.loss = tf.losses.mean_squared_error(self.q_target, self.q)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        self.summaries = tf.summary.merge([tf.summary.scalar("loss", self.loss)])

    def save_model(self):
        global_step = tf.train.global_step(self.sess, self.global_step_tensor)
        self.saver.save(self.sess, self.model_dir, global_step=global_step)

    @staticmethod
    def load_model(saver, sess, model_file):
        saver.restore(sess, model_file)

    def predict(self, s, a):
        q = self.sess.run(self.q, feed_dict={self.s: s, self.a: a})
        return q

    def update(self, samples, lr, target_qnet, target_policynet, save_graph=False):
        global_step = tf.train.global_step(self.sess, self.global_step_tensor)
        state, action, reward, next_state, done = zip(*samples)
        mu = target_policynet.predict(next_state)
        q_next = np.reshape(target_qnet.predict(next_state, mu), [-1])
        q_target = reward + self.gamma * np.multiply((1 - np.array(done).astype(np.float32)), q_next)
        q_target = q_target[:, np.newaxis]

        feed_dict = {self.s: state, self.a: action, self.q_target: q_target, self.lr: lr}
        summaries, loss, _ = self.sess.run([self.summaries, self.loss, self.train_op], feed_dict=feed_dict)

        self.summary_writer.add_summary(summaries, global_step=global_step)
        if save_graph:
            self.summary_writer.add_graph(self.sess.graph)

    def get_q_grad(self, s, a):
        q_grad = self.sess.run(self.q_grad, feed_dict={self.s: s, self.a: a})
        sh = np.shape(q_grad)
        q_grad = np.reshape(q_grad, (sh[1], sh[2]))
        return q_grad


class ReplayMemory:
    def __init__(self, env, env_name, max_memory_size):
        burn_in = int(max_memory_size / 5)
        self.max_memory_size = max_memory_size
        self.memory = burn_in_memory(env, env_name, max_memory_size, burn_in)
        self.N = len(self.memory)

    def sample_batch(self, batch_size=32):
        return self.memory if self.N <= batch_size else self.memory[np.random.randint(0, self.N, batch_size)]

    def append(self, transition):
        new_sample = [transition]
        if self.N < self.max_memory_size:
            self.memory = np.concatenate([self.memory, new_sample])
            self.N += 1
        else:
            self.memory = np.concatenate([self.memory[1:], new_sample])


class Episode:
    def __init__(self, env, env_name):
        self.env = env
        self.env_name = env_name
        self.env.seed(seed)
        self.state_rep = None
        self.state_key = 'observation'
        self.goal_key = 'desired_goal'

    def reset(self):
        obs = self.env.reset()
        state = obs[self.state_key]
        goal = obs[self.goal_key]
        self.state_rep = _concat(state, goal)
        return state, goal, []

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_state = next_obs[self.state_key]
        goal = next_obs[self.goal_key]
        self.state_rep = _concat(next_state, goal)
        return next_state, reward, done, info, goal

    def close(self):
        self.env.close()


def burn_in_memory(env, env_name, max_memory_size, burn_in):
    episode = Episode(env, env_name)
    memory = []

    itr = 0

    while itr < burn_in:
        if itr % 10000 == 0:
            print("Burn in iteration: ", itr)
        state, goal, _ = episode.reset()
        done = False
        while not done:
            action = _random_action(env)
            next_state, reward, done, _, goal = episode.step(action)
            memory.append(_transition(state, action, reward, next_state, done, goal))
            itr += 1
            if itr == max_memory_size:
                break
            state = next_state
    return np.array(memory)


class Plotter:
    def __init__(self, env_name, plot_dir):
        self.env_name = env_name
        self.plot_dir = plot_dir

        self.test_rewards_intermediate = []
        self.test_rewards_final = []

    def plot_rewards(self):
        xlabel, ylabel = 'Number of updates (%s)' % r'$\times 10^4$', 'Average Reward (Test)'
        self._plotY(self.test_rewards_intermediate, 'test_rewards', xlabel, ylabel)
        print('Final Eval (100 iterations):\nAvg: %.2f, Std: %.2f' % self.test_rewards_final[0])

    def _plotY(self, series, filename, xlabel, ylabel):
        fig = plt.figure()
        plt.plot(series)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig.savefig('%s/%s.pdf' % (self.plot_dir, filename), bbox_inches='tight')


class DDPGAgent:

    def __init__(self, env_name, sess, param):
        self.env_name = env_name
        self.sess = sess

        self.render = param['render']
        self.num_epochs = param['num_epochs']
        self.num_episodes = param['num_episodes']
        self.num_cycles = param['num_cycles']
        self.max_updates = param['max_updates']

        self.num_eval = param['num_eval']
        self.num_test = param['num_test']
        self.epsilon0_train = param['epsilon0_train']
        self.epsilon0_test = param['epsilon0_test']

        self.monitor_base_dir = param['monitor_base_dir']
        self.summary_base_dir = param['summary_base_dir']
        self.plot_dir = param['plot_dir']

        self.model_dir = param['model_dir']
        self.minibatch_size = param['minibatch_size']
        self.replay_memory_size = param['replay_memory_size']
        self.stdev_noise = param['stdev_noise']
        self.lr_actor = param['lr_actor']
        self.lr_critic = param['lr_critic']

        self.env = gym.make(self.env_name)
        self.dim_s = self.env.reset()['observation'].shape[0]
        self.dim_g = self.env.reset()['desired_goal'].shape[0]
        self.dim_a = self.env.action_space.shape[0]

        self.plotter = Plotter(self.env_name, self.plot_dir)

        param_net = self.get_param_net(param)
        self.qnet = QNetwork(param_net, scope='QNetwork')
        self.target_qnet = QNetwork(param_net, scope='TargetQNetwork')
        self.policynet = PolicyNetwork(param_net, scope='PolicyNetwork')
        self.target_policynet = PolicyNetwork(param_net, scope='TargetPolicyNetwork')

        self.update_targetqnet = _update_targetnet(self.qnet, self.target_qnet)
        self.update_targetpolicynet = _update_targetnet(self.policynet, self.target_policynet)
        sess.run(tf.global_variables_initializer())

    def get_param_net(self, param_agent):
        param = {}
        pass
        param['dim_s'] = self.dim_s + self.dim_g
        param['dim_a'] = self.dim_a
        param['sess'] = self.sess
        param['model_dir'] = self.model_dir
        param['tf_summary_dir'] = param_agent['tf_summary_dir']
        param['h'] = param_agent['h']
        param['gamma'] = param_agent['gamma']
        return param

    def epsilon_greedy_policy(self, state, epsilon):
        return _random_action(self.env) if random.random() < epsilon else self.greedy_policy(state)

    def greedy_policy(self, state):
        return self.policynet.predict(state[np.newaxis, :]) + _random_noise(self.env, self.stdev_noise)

    def save_plotter(self):
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_path = '%s/%s.pickle' % (self.summary_base_dir, time)
        with open(save_path, 'wb') as fp:
            pickle.dump(self.plotter, fp, pickle.HIGHEST_PROTOCOL)

    def train(self):
        """Train the Q-Network for a given number of episodes."""
        save_graph = True
        save_vid = False
        replay_memory = ReplayMemory(self.env, self.env_name, self.replay_memory_size)

        vid_ckpts = [0, 1/3, 2/3, 1]
        vid_ckpts_iter = [int(i * self.num_epochs) for i in vid_ckpts]
        print("Video checkpoints: ", vid_ckpts_iter)
        episode = Episode(self.env, self.env_name)

        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)

            # Evaluate
            monitor_dir = os.path.join(self.monitor_base_dir, str(epoch))
            if epoch in vid_ckpts_iter:
                save_vid = True
            avg_reward = self.test(monitor_dir, save_vid=False)
            if save_vid:
                save_vid = False
            print("Average reward: ", avg_reward)

            # Save models
            self.qnet.save_model()
            self.policynet.save_model()

            for _ in range(self.num_cycles):
                # Update Target Networks (Q-network and Policy-network)
                self.sess.run(self.update_targetqnet)
                self.sess.run(self.update_targetpolicynet)

                for _ in range(self.num_episodes):
                    state, goal, transition_store = episode.reset()
                    done = False
                    while not done:
                        action = self.epsilon_greedy_policy(episode.state_rep, self.epsilon0_train)
                        next_state, reward, done, _, goal = episode.step(action)
                        transition_store.append((state, action, reward, next_state, done))
                        transition = _transition(state, action, reward, next_state, done, goal)
                        replay_memory.append(transition)
                        state = next_state

                for updates in range(self.max_updates):
                    samples = replay_memory.sample_batch(batch_size=self.minibatch_size)
                    self.policynet.update(samples, self.lr_actor, self.qnet, save_graph=save_graph)
                    self.qnet.update(samples, self.lr_critic, self.target_qnet, self.target_policynet, save_graph=save_graph)
                    if save_graph:
                        save_graph = False

        monitor_dir = os.path.join(self.monitor_base_dir, 'test')
        self.test(monitor_dir, evaluate=False)
        self.save_plotter()
        episode.close()

    def test(self, monitor_dir, evaluate=True, save_vid=False):
        n_episodes = self.num_eval if evaluate else self.num_test
        env = gym.make(self.env_name)
        if save_vid:
            env = wrappers.Monitor(env, monitor_dir, force=True)
        else:
            env = wrappers.Monitor(env, monitor_dir, force=True, video_callable=False)
        episode = Episode(env, self.env_name)
        all_rewards = []

        for ep in range(n_episodes):
            episode.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.epsilon_greedy_policy(episode.state_rep, self.epsilon0_test)
                if save_vid:
                    env.render()
                _, reward, _, done, _ = episode.step(action)
                total_reward += reward
            all_rewards.append(total_reward)
        episode.close()
        avg_reward, std_reward = np.average(all_rewards), np.std(all_rewards)

        if evaluate:
            self.plotter.test_rewards_intermediate.append(avg_reward)
        else:
            self.plotter.test_rewards_final.append((avg_reward, std_reward))
            print("Average reward at test time: ", avg_reward)
            print("Standard deviation at test time: ", std_reward)

        return avg_reward


def _update_targetnet(net, target_net):
    source_variables = tf.trainable_variables(scope=net.scope)
    target_variables = tf.trainable_variables(scope=target_net.scope)
    sort = lambda x: sorted(x, key=lambda v: v.name)
    return [t.assign(s) for s, t in zip(sort(source_variables), sort(target_variables))]


def _transition(state, action, reward, next_state, done, goal):
    state = _concat(state, goal)
    next_state = _concat(next_state, goal)
    return (state, action, reward, next_state, done)


def _concat(state, goal):
    return np.concatenate([state, goal])


def _random_action(env):
    a_lo = env.action_space.low
    a_hi = env.action_space.high
    return np.random.uniform(a_lo, a_hi)


def _random_noise(env, stdev):
    action_space = env.action_space
    a_hi = action_space.high
    a_lo = action_space.low
    return np.random.normal(np.zeros(action_space.shape[0]), stdev * (a_hi - a_lo))


def _validate_args(args):
    assert args.render in [0, 1]
    assert args.minibatch_size > 0


def _base_path(args):
    return '%s' % args.env


def parse_arguments():
    parser = argparse.ArgumentParser(description='DDPG with HER Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='FetchReach-v0')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model_file', dest='model_file', type=str)

    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=50)
    parser.add_argument('--num_cycles', dest='num_cycles', type=int, default=25)
    parser.add_argument('--num_episodes', dest='num_episodes', type=int, default=4)
    parser.add_argument('--max_updates', dest='max_updates', type=int, default=40)

    parser.add_argument('--num_eval', dest='num_eval', type=int, default=20)
    parser.add_argument('--num_test', dest='num_test', type=int, default=100)
    parser.add_argument('--epsilon0_train', dest='epsilon0_train', type=float, default=0.2)
    parser.add_argument('--epsilon0_test', dest='epsilon0_test', type=float, default=0.05)

    parser.add_argument('--stdev_noise', dest='stdev_noise', type=float, default=0.05)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99)
    parser.add_argument('--lr_actor', dest='lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', dest='lr_critic', type=float, default=1e-3)
    parser.add_argument('--minibatch_size', dest='minibatch_size', type=int, default=128)

    parser.add_argument('--plot_only', dest='plot_only', type=int, default=0)
    parser.add_argument('--plot_file_name', dest='plot_file_name')
    parser.add_argument('--hidden', dest='hidden', type=int, default=256)
    parser.add_argument('--replay_memory_size', dest='replay_memory_size', type=int, default=1000000)

    return parser.parse_args()


def main(args):
    args = parse_arguments()

    _validate_args(args)
    env_name = args.env

    base_path = _base_path(args)
    monitor_base_dir = '%s/stats' % base_path
    perf_summary_base_dir = '%s/perf-summary' % base_path
    save_dir = '%s/models' % base_path
    model_dir = '%s/model-ckpt' % save_dir
    tf_summary_dir = '%s/tf-summary' % base_path
    plot_dir = '%s/plots' % base_path

    if args.plot_only == 1:
        with open(args.plot_file_name, 'rb') as fp:
            plotter = pickle.load(fp)
        if not os.path.exists(plotter.plot_dir):
            os.makedirs(plotter.plot_dir)
        plotter.plot_rewards()
        return

    if not os.path.exists(perf_summary_base_dir):
        os.makedirs(perf_summary_base_dir)

    for path in [save_dir, monitor_base_dir, tf_summary_dir, plot_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    param_agent = {}
    pass
    param_agent['render'] = args.render

    param_agent['num_epochs'] = args.num_epochs
    param_agent['num_episodes'] = args.num_episodes
    param_agent['num_cycles'] = args.num_cycles
    param_agent['max_updates'] = args.max_updates

    param_agent['num_eval'] = args.num_eval
    param_agent['num_test'] = args.num_test
    param_agent['epsilon0_train'] = args.epsilon0_train
    param_agent['epsilon0_test'] = args.epsilon0_test

    param_agent['stdev_noise'] = args.stdev_noise
    param_agent['gamma'] = args. gamma
    param_agent['lr_actor'] = args.lr_actor
    param_agent['lr_critic'] = args.lr_critic
    param_agent['minibatch_size'] = args.minibatch_size

    param_agent['h'] = args.hidden
    param_agent['replay_memory_size'] = args.replay_memory_size

    param_agent['summary_base_dir'] = perf_summary_base_dir
    param_agent['monitor_base_dir'] = monitor_base_dir
    param_agent['model_dir'] = model_dir
    param_agent['plot_dir'] = plot_dir
    param_agent['tf_summary_dir'] = tf_summary_dir

    # Setting the session to allow growth, so it doesn't allocate all GPU
    # memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    ddpg_agent = DDPGAgent(env_name, sess, param_agent)
    ddpg_agent.train()


if __name__ == '__main__':
    main(sys.argv)
