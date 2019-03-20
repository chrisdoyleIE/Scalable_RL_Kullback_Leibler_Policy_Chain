
#---------------------------------------------------
#       Gridwolrd Implementation for OpenAI GYM
#       Author: chrisdoyleIE
#---------------------------------------------------


#---------------------------
# Requiremnts
#---------------------------

import numpy as np
import math

# For Plotting Purposes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# For Creating Action and Observation Spaces etc
import gym
from gym import spaces, logger
from gym.utils import seeding

# For Saving Episode GIF
import os
import imageio
import time
import datetime
import uuid
import natsort



#print('IMPORT: grid_world.py')


class GridWorldEnv:

    #---------------------------
    # Information
    #---------------------------

    # - colormap = 'spectral'   # Theme for plt.matplot()
    # - boundary_color = 0.95   # Red in spectral
    # - agent_color = 0.25      # XX in spectral
    # - reward_color = 0.75     # XX in spectral

    #---------------------------
    # Function Attributes
    #---------------------------


    def __init__(self, num_episodes, default_reward_pos, training):

        # Initialise Attributes
        self.num_episodes = num_episodes                # number of episodes we will train our RL agent for
        self.dims = np.array([8,8])                   # dimensions of gridworld
        self.GW = np.ones(self.dims)                    # default value of grid is 1
        self.colormap = 'nipy_spectral'                 # color scheme for matshow() gridworld
        self.boundary_value = 0.95                      # Vaule of gridworld where boundaries lie
        self.agent_value = 0.25                         # Value of gridworld where agent is located
        self.reward_value = 0.75                        # Value of gridworld where reward is located
        self.goal_flag = 'N'                            # Used in naming convention for GIFs
        self.episode = 1                               # Used in env.step(action) for terminus counter
        self.episodes = []                              # for debugging
        self.default_reward_pos = default_reward_pos    # Default Reward Position
        self.starting_pos = []                          # To record where the agent starts his adventure
        self.states_visited = np.zeros(self.dims)       # Required to generate appropriate step reward
        self.starting_pos.append([0,0])                 # Ensures starting_pos[episode] is correct
        self.steps_initial = 30                       # Alllowed number of steps

        # Initialise Action Space {Up, Down, Left, Right}
        self.action_space = gym.spaces.Discrete(4)

        # Initialise Observation Space {agent_x, agent_y}
        low = np.array([0,0])
        high = np.array(self.dims)
        self.observation_space = spaces.Box(low,high,dtype=np.float32)

         # Initialise Specific Reward Values
        self.goal_reward = 100
        self.boundary_reward = -100
        self.step_reward = -1
        self.new_state_reward = 0

        # Create Directory for the run
        if training: self.dir_for_run = 'Results/Logged Training Runs/'+ str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H:%M:%S'))
        else: self.dir_for_run = 'Results/Logged Test Runs/'+ str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H:%M:%S'))

        os.mkdir(self.dir_for_run)
        self.log_dir = self.dir_for_run + '/stats.csv'
        self.log_file = open(self.log_dir, "a")
        self.log_file.write('Grid Size: %d\r\nEpisdodes: %d\n\r\nReward Location: [%d %d]\nGoal Reward: %d,Boundary Reward: %d,New State Reward: %d,Step Reward: %d\r\n' % (
                            self.dims[0],self.num_episodes,self.default_reward_pos[0],self.default_reward_pos[1],self.goal_reward,self.boundary_reward,self.new_state_reward,self.step_reward)
                            )
        self.log_file.close()


        # Require to save cause of failure
        self.terminus_states = []                               # Label of Terminus state for each episode
        self.terminus_count = np.zeros(3)                       # (#Goal, #Boundary, #Steps)
        self.terminus_counts = np.zeros((self.num_episodes,3))    # For plotting representation of above

        # See lines 81 - 85, cartpole.py from OpenAI
        self.seed()
        self.viewer = None
        self.state = None
        self.frame_number = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        agent_position = self.state
        reward = 0
        self.visited(self.state)
        self.steps_remaining -= 1

        # Account for initialising on top of reward
        if( self.state[0] == self.default_reward_pos[0] and 
            self.state[1] == self.default_reward_pos[1] ): 
            self.goal_flag = 'Y'
            self.terminus_states.append('GOAL')
            self.terminus_count[0] += 1
            return np.array(self.state), self.goal_reward, True, {}
        
        
        # Set Tile Agent has Left to Default Value
        self.GW[ self.state[0] ][ self.state[1] ] = 1

        # Take Action to update Agent Position, +/- 1 based on matshow() grid co-ordinates
        if(action==0): self.state = [agent_position[0] -1, agent_position[1]] # UP
        if(action==1): self.state = [agent_position[0] +1, agent_position[1]] # DOWN
        if(action==2): self.state = [agent_position[0], agent_position[1] -1] # LEFT
        if(action==3): self.state = [agent_position[0], agent_position[1] +1] # RIGHT
        #print('AGENT: ', self.state, ', VALUE = ', self.GW[ self.state[0] ][ self.state[1] ],', ACTION = ',action )

        # Check if Done
        done = bool((self.GW[self.state[0]][self.state[1]] == self.boundary_value) or 
                    (self.GW[ self.state[0] ][ self.state[1] ] == self.reward_value) or
                    (self.steps_remaining <= 0)
                    )
        

        
        if done:
            # Hit the boundary
            if (self.GW[ self.state[0] ][ self.state[1] ] == self.boundary_value):
                reward = self.boundary_reward
                self.terminus_states.append('BOUNDARY')
                self.terminus_count[1] += 1

            # Reached the Goal
            elif (self.GW[ self.state[0] ][ self.state[1] ] == self.reward_value):
                reward = self.goal_reward
                self.goal_flag = 'Y'
                self.terminus_states.append('GOAL')
                self.terminus_count[0] += 1

            # Ran out of steps
            else: 
                reward = self.step_reward
                self.terminus_states.append('STEPS')
                self.terminus_count[2] += 1

            # Update GridWorld to Show Agent Position (must be done after if checks)
            self.GW[self.state[0]][self.state[1]] = self.agent_value

            return np.array(self.state), reward, done, {}

        # Otherwise continue as normal
        else:
            if self.is_visited(self.state): 
                reward = self.step_reward             
            else:
                reward = self.new_state_reward        # Reward for entering a new state

        # Update GridWorld to Show Agent Position
        self.GW[self.state[0]][self.state[1]] = self.agent_value
        return np.array(self.state), reward, done, {}

    def debug_step(self,action):

        agent_position = self.state
        # Set Tile Agent has Left to Default Value
        self.GW[ self.state[0] ][ self.state[1] ] = 1


        # Take Action to update Agent Position, +/- 1 based on matshow() grid co-ordinates
        if(action==0): self.state = [agent_position[0] -1, agent_position[1]] # UP
        if(action==1): self.state = [agent_position[0] +1, agent_position[1]] # DOWN
        if(action==2): self.state = [agent_position[0], agent_position[1] -1] # LEFT
        if(action==3): self.state = [agent_position[0], agent_position[1] +1] # RIGHT
        #print('AGENT: ', self.state, ', VALUE = ', self.GW[ self.state[0] ][ self.state[1] ],', ACTION = ',action )

        # Update GridWorld to Show Agent Position
        self.GW[self.state[0]][self.state[1]] = self.agent_value

    def reset(self):

        # Incremment the episode number
        #self.episode += 1 # incremented in top module file now.          

        # Reset GridWorld
        self.GW = np.ones(self.dims) 
        self.states_visited = np.zeros(self.dims)
        self.steps_remaining = self.steps_initial
        
        # Initialise Boundary of GridWorld
        for i in range( self.dims[0] ): 
            for j in range( self.dims[1] ): 
                if i == 0: 
                    self.GW[i][j] = self.boundary_value
                elif i == self.dims[0]-1: 
                    self.GW[i][j] = self.boundary_value
                elif j == 0: 
                    self.GW[i][j] = self.boundary_value
                elif j == self.dims[1]-1: 
                    self.GW[i][j] = self.boundary_value

        # Determine Reward Position
        self.set_reward(fixed = True)

        #Reset Agent to Random Location
        self.state = np.array( [math.trunc( np.random.uniform(low = 1, high = self.dims[0] -1) ),
                                math.trunc( np.random.uniform(low = 1, high = self.dims[1] -1) ) ]
                                )
        self.GW[ self.state[0] ][ self.state[1] ] = self.agent_value
        self.starting_pos.append(self.state)
        #print('AGENT: ',self.state)

        # Reset Step Counter 
        self.frame_number = 0
        self.goal_flag = 'N'

        return np.array(self.state)
 
    def set_reward(self, fixed):

        self.reward_position = self.default_reward_pos

        if not fixed:
            # Reset Reward Position such that the agent is on another tile
            self.reward_position = self.state

            # Ensure that reward position is different to starting position
            while np.array_equal( self.reward_position, self.state ):
                    self.reward_position = np.array( [math.trunc( np.random.uniform(low = 1, high = self.dims[0] -1) ),
                                                      math.trunc( np.random.uniform(low = 1, high = self.dims[1] -1) ) ]
                                                    )

        self.GW[self.reward_position[0]][self.reward_position[1]] = self.reward_value
        
        #print('REWARD: ',reward_position)

    def render_step(self):

        plt.matshow(self.GW, 
                    cmap = self.colormap,
                    interpolation = 'none',
                    vmin = 0,
                    vmax = 1
                    )
        # Re-centre pixels such that the grid sepparates them as desired
        # plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
        # plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
        # plt.grid(which='minor')


        # Show Frame
        plt.show()

    def save_step(self):
        plt.matshow(self.GW, 
                    cmap = self.colormap,
                    interpolation = 'none',
                    vmin = 0,
                    vmax = 1
                    )

        # Save Frame
        frame_name = 'Results/Temp/FRAME_'+str(self.frame_number) + '.jpeg'
        plt.savefig(frame_name, bbox_inches='tight')
        plt.close()
        #print('SAVED: ','[',self.state[0],',',self.state[1],'], FRAME: ', self.frame_number)
        self.frame_number += 1
        self.dir_created = False # Reset this boolean each to ensure directory for each run created correctly

    def save_episode(self):

        # Ensure the frames are ordered correctly (sorted naturally so 9 < 10, i.e not 1, 10, 11, ..., 2, 20, 21, ...)
        temp_folder = os.listdir('Results/Temp')
        temp_folder = natsort.natsorted(temp_folder)

        images = []
        for filename in temp_folder: 
            next_frame = 'Results/Temp/' + filename
            
            # Append more than once to slow down GIF
            for i in range(0,3): images.append(imageio.imread(next_frame))
            
            # Keep the folder clean
            os.remove(next_frame)
            #print('APPENDED: ', filename)
        
        gif_name = self.dir_for_run + '/' + 'Episode_'+str(self.episode)+'_'+self.goal_flag+'.gif'
        imageio.mimsave( gif_name, images)

    def log(self, episode_rewards, reward_running_average):


        # Update the stats file
        self.log_dir = self.dir_for_run + '/stats.csv'
        self.log_file = open(self.log_dir,"a")
        self.log_file.write('\nTerminus Occurences:\nGoal Count: {},Boundary Count: {}, Steps Count: {}\r\n'.format(
                            self.terminus_count[0],self.terminus_count[1],self.terminus_count[2])
                            )

        denominator = self.terminus_count[0] + self.terminus_count[1] + self.terminus_count[2]
        self.log_file.write('\nTerminus Statistics:\nGoal %: {},Boundary %: {}, Steps %: {}\r\n'.format(
                            self.terminus_count[0]/denominator,self.terminus_count[1]/denominator,self.terminus_count[2]/denominator)
                            )
        self.log_file.close()

        # Create a new log file
        self.log_dir = self.dir_for_run + '/log.csv'
        self.log_file = open(self.log_dir,"a")
        self.log_file.write('Episode,Starting Position,Steps,Terminal State,Reward,Running Average\n')

        # DEBUG
        # print(len(self.starting_pos), len(self.terminus_states), len(episode_rewards) , len(reward_running_average))
        
        # Fill the file with recorded results
        for i in range(1,self.episode ): 
            self.log_file.write('{},{},{},{},{},{}\n'.format(
                                i,(str(self.starting_pos[i][0])+' '+str(self.starting_pos[i][1])),30 - self.steps_remaining,self.terminus_states[i-1],episode_rewards[i-1],reward_running_average[i-1]))
        self.log_file.close()

        # Create a plot of episode vs reward
        episodes = np.asarray(range(1,self.num_episodes+1)) # 1:num_episodes (inclusive)

        plt.figure(0)
        plt.scatter(episodes , episode_rewards, s = 1)
        plt.plot(episodes, reward_running_average,color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig( (self.dir_for_run+'/'+'Episode vs Rewards') )

        
        plt.figure(1)
        plt.plot(episodes, self.terminus_counts[:,0])
        plt.plot(episodes, self.terminus_counts[:,1])
        plt.plot(episodes, self.terminus_counts[:,2])
        plt.xlabel('Episode')
        plt.ylabel('Total')
        plt.legend(['Goals','Boundaries','Out of Steps'], loc = 'upper left')
        plt.savefig( (self.dir_for_run+'/'+'Terminus State Breakdown with Time') )

    def visited(self,state):
        self.states_visited[self.state[0]][self.state[1]] = 1

    def is_visited(self,state):
        if self.states_visited[self.state[0]][self.state[1]] == 1: 
            return True
        else: 
            return False

    def increment_episode(self):
        self.episode += 1
        self.episodes.append(self.episode)

    def update_terminus_totals(self):
        # Update Terminus running totals
        self.terminus_counts[self.episode-1][0] = self.terminus_count[0]
        self.terminus_counts[self.episode-1][1] = self.terminus_count[1]
        self.terminus_counts[self.episode-1][2] = self.terminus_count[2]