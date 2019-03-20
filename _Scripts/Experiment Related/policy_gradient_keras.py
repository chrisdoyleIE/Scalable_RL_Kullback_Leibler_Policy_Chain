import numpy as np 
import tensorflow as tf 
import math
# For Plotting Purposes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import keras
from keras import losses, backend
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import model_from_json

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
        self.action_rewards = np.zeros([2,1])
        self._build_net()

        #------------------------------------------------------------------
        # Attributes LLRL
        #------------------------------------------------------------------

        self.policy_files = []
        self.saver = tf.train.Saver()
        self.ckpt_path = None

    def model_loss(self):
        '''This function is needed as a wrapper for a custom loss 
        function for model.compile(loss = model_loss())'''

        # External Required input to the loss function
        action_rewards = self.action_rewards
        print('LOSS CALLED')

        # Function to be returned to model.compile()
        def loss_function(y_true,y_pred):
                # See diagram included in 'custom_loss.png'
                neg_log_prob = keras.losses.categorical_crossentropy(y_true, y_pred) # returns scalar
                loss = keras.backend.mean(neg_log_prob * action_rewards)
                return loss
        return loss_function

    def _build_net(self):

        # Define Network Architecture
        self.model = Sequential()
        self.model.add(Dense(10, activation='tanh',input_shape = (2,1)))
        self.model.add(Dense(4, activation=None))
        self.model.add(Activation('softmax'))

        # Compile model with custom loss
        self.loss_func = self.model_loss()
        self.model.compile(optimizer=Adam(self.lr),
              loss=self.loss_func,
              metrics=['accuracy'])

        # Print Summary
        #self.model.summary()
  
    def learn(self, training = True):
        # discount and normalize episode reward
        self.action_rewards = self._discount_and_norm_rewards()
        
        # X = observations, Y = actions (for each episode)
        if training:
            self.model.fit(self.ep_obs,self.ep_as)
        
        # Save episode rewards
        self.rewards.append( sum(self.ep_rs) )
        self.running_average.append( sum(self.rewards) / len(self.rewards) )    # Running Average


        # empty episode data
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    
        return self.action_rewards

    def choose_action(self, observation, flag):
        
        # Return disctribution of actions given state
        prob_weights = self.model.predict(observation)

        # Select action w.r.t the actions propability
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  

        # EXPERIMENTAL OUTPUT
        # if flag: print('prob_weights',prob_weights)

        return action

    def store_transition(self, s, a, r, flag):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

        # EXPERIMENTAL OUTPUT
        #if flag: print('REWARDS: ',self.ep_rs)
  
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        discounted_ep_rs = discounted_ep_rs.astype(float)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # # Prioritise later actions
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
        for row in range(self.gridworld_dims[0]):
            for col in range(self.gridworld_dims[1]):
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
