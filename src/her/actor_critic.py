import tensorflow as tf
from her.utils.misc import store_args, nn
import her.constants as C


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, replay_strategy,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

        if self.replay_strategy == C.REPLAY_STRATEGY_GEN_K:
            self.e_tf = inputs_tf['e']
            self.mask_tf = inputs_tf['mask']

            # e = self.e_stats.normalize(self.e_tf)
            e = self.e_tf

            # Prepare inputs for goal.
            input_goal = tf.concat(axis=1, values=[self.u_tf/self.max_u, e, self.mask_tf])

            with tf.variable_scope('goal'):
                self.goal_tf = 5 * self.max_g * tf.sigmoid(nn(input_goal, [self.hidden] * self.layers + [self.dimg]))

            g_inp = self.g_stats.normalize(self.goal_tf)

            # self.max_g *= 5
            # g_inp = self.goal_tf.copy()/self.max_g

            input_pi_goal = tf.concat(axis=1, values=[o, g_inp])  # for actor
            with tf.variable_scope('gpi'):
                self.pi_goal_tf = self.max_u * tf.tanh(nn(input_pi_goal, [self.hidden] * self.layers + [self.dimu]))

            with tf.variable_scope('gQ'):
                # for policy training
                input_Q = tf.concat(axis=1, values=[o, g_inp, self.pi_goal_tf / self.max_u])
                self.Q_pi_goal_tf = nn(input_Q, [self.hidden] * self.layers + [1])
                # for Q training
                input_Q = tf.concat(axis=1, values=[o, g_inp, self.u_tf / self.max_u])
                self.Q_goal_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

            self.e_reshaped = tf.reshape(e, (-1, self.T, self.dimg))
            self.goal_tf_repeated = tf.transpose(tf.tile(self.goal_tf[tf.newaxis, :, :], (self.T, 1, 1)), perm=[1, 0, 2])
            self.distance = self.goal_tf if self.relative_goals else (self.goal_tf_repeated - self.e_reshaped)
            self.d = tf.norm(self.distance, axis=2)

            if self.rshaping:
                self.d_2 = self.d[:, 1:]
                # self.reward = self.rshape_lambda * tf.square(d[:, self.o_ind]) - tf.square(d[:, self.o2_ind])
            # else:
            #     self.reward = -1 / (1 + tf.exp(-self.slope * (self.d - self.d0)))
                self.reward = 1 /(1 + tf.square(self.d))
            #     self.reward = -tf.square(self.d)
            masked_reward = tf.multiply(self.reward, self.mask_tf)
            self.reward_sum = tf.reduce_sum(masked_reward, axis=1)