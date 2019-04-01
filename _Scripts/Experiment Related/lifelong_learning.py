#-------------------------------------------------------------------------------------------
#               Usage: To compare policies & KL result
#               Author: Chris Doyle
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
#   REQUIREMENTS
#-------------------------------------------------------------------------------------------

# Header Files
import policy_gradient as pg
import grid_world as gw

# Tensorflow Dependancies
import numpy as np
from numpy import random 
import tensorflow as tf 
import math

# For Plotting Purposes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------
#   CLASS DEFINITIONS
#-------------------------------------------------------------------------------------------

class LifelongLearning:
    
    def __init__(self):
        
        # INITIALISE ATTRIBUTES
        self.lr = 0.001                 # Learning rate
        self.n_actions = 4              # Number of possible actions
        self.n_features = 2             # State observation
        self.n_epochs = 1000               # Number of epochs for training
        self.disp_X_times = 1            # Number of times training displayed
        self.dir_for_run = 'Results/CRL_training/'
        self.policy = [[[]for col in range( 0,8 )] for row in range(0,8 ) ]
        self.delta = 0.000001            # to prevent divide by zero

    
        # Step 1: Initialise Graphs for each model
        self.next_policy = tf.Graph()
        self.current_chain = tf.Graph()
        self.create_new_chain()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.load_policies(
            'Results/Stored_Policies/R22_N8/policy.ckpt',
            'Results/Stored_Policies/R66_N8/policy.ckpt'
            )

        # Step 2: Create a uniform policy to begin with for the resulting policy

    def load_policies(self, policy1_path, policy2_path):
        
        # Step 1: Load the graph for new policy 
        with self.next_policy.as_default():
            graph_path = policy1_path + '.meta'
            self.saver = tf.train.import_meta_graph(graph_path)
            self.saver.restore(self.sess, policy1_path)

        # Step 2: Load the graph for policy chain 
        with self.current_chain.as_default():
            graph_path = policy2_path + '.meta'
            self.saver = tf.train.import_meta_graph(graph_path)
            self.saver.restore(self.sess, policy2_path)

    def create_new_chain(self):

        # NN PLACEHOLDERS
        self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")

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
        # Want actions to be result of KL Divergence function (actions) of input actions
        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   

        with tf.name_scope('train'):
            self.train_op_crl = tf.train.AdamOptimizer(self.lr).minimize(self.neg_log_prob)

    def variational_crl_loss(self, policy1, policy2, inclusive = True):

        # # temporary implementation of loss function:
        # policy = 0.5 * ( np.asarray(policy1) + np.asarray(policy1) )

        #-------------------------------------------------------
        # SEE 3.2.1 IN THESIS FOR INFORMATION ON CODE BELOW
        #-------------------------------------------------------

        # Kullback Leibler Divergence
        def kullback_leibler(p,q, inclusive):

            # Intialise kld value
            kld = 0       

            # See equation 3.4 in thesis
            if inclusive: 
                for i in range(0,4): 
                    kld = kld + (p[i]*np.log(p[i]/(q[i]+self.delta)))
            else:
                for i in range(0,4): kld = kld + (q[i]*np.log(q[i]/(p[i]+self.delta)))

            return kld            

        # Loss function and gradient as defined in chapter 3 [Design] of VPC thesis
        def loss_gradient(q,p1,p2,alpha, inclusive):

            #  Section 3.2.1 first half
            loss = (alpha * kullback_leibler(q, p1, inclusive)) + ((1 - alpha) * kullback_leibler(q, p2, inclusive)) 
            gradient = np.zeros([4,])   # initialise gradient
            

            # Section 3.2.1 second half
            if inclusive:
                for i in range(0,4):
                    gradient[i] = (1-alpha)*( p2[i]/(q[i]+self.delta) )- alpha*( p1[i]/(q[i]+self.delta) )
                    if gradient[i] < -50: gradient[i] = -50
            else:
                for i in range(0,4):
                    gradient[i] = (alpha)*(np.log(q[i]/p1[i]+self.delta)+1) + (1-alpha)*(np.log(q[i]/p2[i]+self.delta))
                    if gradient[i] < -50: gradient[i] = -50
            return loss, gradient
        

        q = np.full([4,], 0.25)                     # Initialise new policy as uniform distribution
        p1 = np.reshape(np.asarray(policy1), [4,])  # Convert policies to np array format
        p2 = np.reshape(np.asarray(policy2), [4,])
        alpha = 0.4                    # alpha = 1/t, we have 2 polices to take information from
        lr_new = 0.0001

        # TEST OUTPUT
        #print('P1, P2 = {}'.format(p1))

        # Gradient Descent
        for episode in range(1,100):
            loss,gradient = loss_gradient(q,p1,p2,alpha, inclusive = False)    # Compute loss and gradient
            q = q - (lr_new * gradient)                    # Perform gradient descent step
            q_mask = q > 0
            q = q * q_mask

            # Monitor Progress
            #if (episode % 100 == 0 and episode != 0): print('Episode: {}, Loss: {}'.format(episode,loss))

        # Return New Policy
        q = np.asarray(q)
        q = q.reshape(-1,4) # Now shape = (1,4) as required
        q = q / np.sum(q)   # Normalise q 
        #print('q = {}'.format(q))
        return q
     
    def train_new_chain(self):

        # We want train the policy on every state
        # Obtain action estimates for each state
        # Get the distribution that minimises the CRL variational CRL loss function
        # Train new policy on this result

        for e in range(0,self.n_epochs):

            # Explore a new state
            state = np.asarray([np.random.randint(0,8),np.random.randint(0,8)])
    

            # Get action from newest policy
            with self.next_policy.as_default():
                policy1 = self.sess.run([self.all_act_prob],
                                feed_dict={
                                    self.tf_obs: state[np.newaxis, :]  # shape=[None, n_obs]
                                })                 
            # Get action from policy chain
            with self.current_chain.as_default():
                policy2 = self.sess.run([self.all_act_prob],
                                feed_dict={
                                    self.tf_obs: state[np.newaxis, :]  # shape=[None, n_obs]
                                })
            # Minimise the variational CRL loss function to find distribution over actions
            policy3 = self.variational_crl_loss(policy1,policy2)

            # Select action based on resulting distribution
            action = np.random.choice(range(policy3.shape[1]), p=policy3.ravel())  # select action w.r.t the actions prob
            action = np.asarray(action).reshape(-1,)

            # Train the new policy to choose this action
            self.sess.run([self.train_op_crl], 
            feed_dict={
                self.tf_obs: state[np.newaxis, :] ,  # shape=[None, n_obs]
                self.tf_acts: np.asarray(action),  # shape=[None, ]
            })

            # Print the loss 10 times throughout training:
            if(e % (self.n_epochs / self.disp_X_times) == 0 and e > 0):

                nn_policy3 = self.sess.run([self.all_act_prob],
                                            feed_dict={
                                                self.tf_obs: state[np.newaxis, :]  # shape=[None, n_obs]
                                            })  
                loss = np.linalg.norm(policy3 - nn_policy3)  
                print('Episode: ',e,'  Loss: ',loss)

    def save_policy(self):

        # Convert graph so it is suitable for rl training.
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value") #based on SUM of discounted rewards
        self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt, name = 'loss') # reward guided loss, see project copybook
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Save tensorflow model policy checkpoint
        self.ckpt_path = self.dir_for_run + '/policy.ckpt'
        self.saver.save(self.sess, self.ckpt_path)

        # Save Policy to .csv file for visualisation
        policy_csv = self.dir_for_run + '/policy.csv'
        policy_csv = open(policy_csv,"a")
        policy_csv.write("ROW,COL,UP,DOWN,LEFT,RIGHT\n")

        # Write Policy to policy.csv
        for row in range(0,8):
            for col in range(0,8):
                    state = np.asarray([row,col])
                    self.policy[row][col] = self.sess.run(self.all_act_prob, feed_dict = {self.tf_obs: state[np.newaxis, :]})
                    policy_csv.write("[{},{}],{},{},{},{}\n".format(
                                    row,col,self.policy[row][col][0][0],self.policy[row][col][0][1],self.policy[row][col][0][2],self.policy[row][col][0][3]
                                    ))



ll = LifelongLearning()
ll.train_new_chain()
ll.save_policy()

