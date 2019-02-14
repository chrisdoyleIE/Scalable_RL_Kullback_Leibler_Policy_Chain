import numpy as np 
import tensorflow as tf 
import math

# -------------------------------------------------------------------------------------------------------------------------
# REINFORCE in tf (Géron,2018)
#-------------------------------------------------------------------------------------------------------------------------

#  1. Monitor the agent as it explores the environment. At each step, compute the gradients to update actions accordingly (but do not apply).
#  2. Compute each action's score after several episodes by applying the calculated gradients.
#  3. Compute the mean of the resulting gradient vectors (of each episode), and use it to perform a Gradient Descent update.

#  -------------------------------------------------------------------------------------------------------------------------

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            gridworld_dims,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.gridworld_dims = gridworld_dims
        self.policy = [[[]for col in range( self.gridworld_dims[1] )] for row in range( self.gridworld_dims[0] ) ]

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation, flag):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})

        # EXPERIMENTAL OUTPUT
        #if flag: print('prob_weights',prob_weights)

        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob

        #Try to Experiment with Uniform
        #action = math.trunc(np.random.uniform(0,4))
        return action

    def store_transition(self, s, a, r, flag):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

        # EXPERIMENTAL OUTPUT
        #if flag: print('REWARDS: ',self.ep_rs)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        
        # EXPERIMENTAL OUTPUT
        #print('discounted_ep_rs_norm',discounted_ep_rs_norm)

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        discounted_ep_rs = discounted_ep_rs.astype(float)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)

        # EXPERIMENTAL OUTPUT
        #print('np.std(discounted_ep_rs)', np.std(discounted_ep_rs))

        if (np.std(discounted_ep_rs) != 0):
            discounted_ep_rs /= np.std(discounted_ep_rs)


        return discounted_ep_rs

    def return_policy(self):
        for row in range(self.gridworld_dims[0]):
            for col in range(self.gridworld_dims[1]):    
                state = np.array([row,col])
                self.policy[row][col] = self.sess.run(self.all_act_prob, feed_dict = {self.tf_obs: state[np.newaxis, :]})
                #print("P(a|s = ", state, ") = ",self.policy[row][col])

        return self.policy


