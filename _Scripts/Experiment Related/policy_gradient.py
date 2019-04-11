import numpy as np 
import tensorflow as tf 
import math
# For Plotting Purposes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
            dir_for_run,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=True,
    ):
        # -----------------------------------------------------------------
        # Attributes for each RL agent
        #------------------------------------------------------------------
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.gridworld_dims = gridworld_dims
        self.policy = [[[]for col in range( self.gridworld_dims[1] )] for row in range( self.gridworld_dims[0] ) ]
        self.loss_values = []

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.observations, self.actions, self.rewards = [], [], []
        self.running_average = []
        self.dir_for_run = dir_for_run
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        #------------------------------------------------------------------
        # Attributes LLRL
        #------------------------------------------------------------------

        self.policy_files = []
        self.saver = tf.train.Saver()
        self.ckpt_path = None

    def _build_net(self):

        # NN PLACEHOLDERS
        self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value") #based on SUM of discounted rewards

        # NN ARCHITECTURE
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

        # LOSS FUNCTION
        #with tf.name_scope('loss'):
        # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
        # logits = output from NN hidden layers before Softmax, labels = all actions taken in the episode
        # Equivalent to : 
        # nlp = tf.nn.softmax(logits)
        # nlp = cross_entropy(nlp,self.tf_acts)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt, name = 'loss') # reward guided loss, see project copybook

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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

    def learn(self, training = True):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        
        # EXPERIMENTAL OUTPUT
        #print('discounted_ep_rs_norm',discounted_ep_rs_norm)

        if training:
            self.sess.run([self.train_op,self.loss], feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
                self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
                self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            })
        #self.loss_values.append(loss_val)
        #
        # Save episode rewards
        self.rewards.append( sum(self.ep_rs) )
        self.running_average.append( sum(self.rewards) / len(self.rewards) )    # Running Average
  

        # empty episode data
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        discounted_ep_rs = discounted_ep_rs.astype(float)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # # Prioritise later actions
        # for t in range(0, len(self.ep_rs)):
        #     running_add = running_add * self.gamma + self.ep_rs[t]
        #     discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)

        # EXPERIMENTAL OUTPUT
        #print('np.std(discounted_ep_rs)', np.std(discounted_ep_rs))

        if (np.std(discounted_ep_rs) != 0):
            discounted_ep_rs /= np.std(discounted_ep_rs)


        return discounted_ep_rs

    def save_policy(self):

        policy_csv = self.dir_for_run + '/policy.csv'
        policy_csv = open(policy_csv,"a")
        policy_csv.write("ROW,COL,UP,DOWN,LEFT,RIGHT\n")

        # Write Policy to policy.csv
        for row in range(0,8):
            for col in range(0,8):
                    state = np.array([row,col])
                    self.policy[row][col] = self.sess.run(self.all_act_prob, feed_dict = {self.tf_obs: state[np.newaxis, :]})
                    policy_csv.write("[{},{}],{},{},{},{}\n".format(
                                    row,col,self.policy[row][col][0][0],self.policy[row][col][0][1],self.policy[row][col][0][2],self.policy[row][col][0][3]
                                    ))

        # Save tensorflow model policy checkpoint
        self.ckpt_path = self.dir_for_run + '/policy.ckpt'
        self.saver.save(self.sess, self.ckpt_path)
        print('POLICY STORED')

        # # Plot loss
        # episodes = range(len(self.loss_values))
        # plt.figure(2)
        # plt.plot(episodes, self.loss_values)
        # plt.xlabel('Episode')
        # plt.ylabel('Loss')
        # plt.savefig( (self.dir_for_run+'/'+'Learning Curve (Loss)') )

    def load_policy(self, policy_path):
        graph_path = policy_path + '.meta'
        self.saver = tf.train.import_meta_graph(graph_path)
        self.saver.restore(self.sess, policy_path)
        print('POLICY LOADED')

    def print_stats(self, episode):
        # print("epsiode: {}, reward: {}, avg_reward: {}".format(
        #     episode+1, sum(self.ep_rs), int(self.running_average[episode-1]) ))

        print("epsiode: {}, reward: {}".format(
            episode+1, sum(self.ep_rs) ))