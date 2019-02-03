
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


    def __init__(self):

        # Initialise Attributes
        self.dims = np.array([7,7])
        self.GW = np.ones(self.dims) # default value of grid is 1
        self.colormap = 'nipy_spectral'
        self.boundary_value = 0.95
        self.agent_value = 0.25
        self.reward_value = 0.75

        # Initialise Action Space {Up, Down, Left, Right}
        self.action_space = gym.spaces.Discrete(4)

        # Initialise Observation Space {agent_x, agent_y}
        low = np.array([0,0])
        high = np.array(self.dims)
        self.observation_space = spaces.Box(low,high,dtype=np.float32)

        # See lines 81 - 85, cartpole.py from OpenAI
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_remaining = 20


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        agent_position = self.state
        reward = 0
        
        # Set Tile Agent has Left to Default Value
        self.GW[ self.state[0] ][ self.state[1] ] = 1

        # Take Action to update Agent Position
        if(action==0): self.state = [agent_position[0], agent_position[1]+1] # UP
        if(action==1): self.state = [agent_position[0], agent_position[1]-1] # DOWN
        if(action==2): self.state = [agent_position[0]-1, agent_position[1]] # LEFT
        if(action==3): self.state = [agent_position[0]+1, agent_position[1]] # RIGHT
        print('AGENT: ', self.state)

        # Check if Done
        done = bool( self.GW[self.state[0]][self.state[1]] == self.boundary_value)

        self.steps_remaining -= 1

        if not done:
            # Reached the Goal
            if (self.GW[ self.state[0] ][ self.state[1] ] == self.reward_value):
                reward = 5
                done = True
            # Ran out of Steps
            elif self.steps_remaining == 0:
                done = True
                reward = -10
            # Otherwise just continue onwards
            else:
                reward = -1
        # Agent has gone over the boundary
        else:
            reward = -10

        # Update GridWorld to Show Agent Position
        self.GW[self.state[0]][self.state[1]] = self.agent_value

        return np.array(self.state), reward, done, {}


    def reset(self):
        # Reset GridWorld
        self.GW = np.ones(self.dims) 
        
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

        # Reset Agent to Random Location
        self.state = np.array( [math.trunc( np.random.uniform(low = 1, high = self.dims[0] -1) ),
                                math.trunc( np.random.uniform(low = 1, high = self.dims[1] -1) ) ]
                                )
        self.GW[ self.state[0] ][ self.state[1] ] = self.agent_value
        print('AGENT: ',self.state)

        # Reset Reward Position
        reward_position = self.state

        # Ensure that reward position is different to starting position
        while np.array_equal( reward_position, self.state ):
                reward_position = np.array( [math.trunc( np.random.uniform(low = 1, high = self.dims[0] -1) ),
                                             math.trunc( np.random.uniform(low = 1, high = self.dims[1] -1) ) ]
                                            )
        self.GW[reward_position[0]][reward_position[1]] = self.reward_value
        print('REWARD: ',reward_position)
        # Reset Step Counter 
        self.steps_remaining = 20
        return np.array(self.state)
 

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
        # Re-centre pixels such that the grid sepparates them as desired
        # plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
        # plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
        # plt.grid(which='minor')

        # Save Frame
        frame_name = 'Results/FRAME_'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.jpeg'
        plt.savefig(frame_name, bbox_inches='tight')


    def save_episode(self):

        images = []

        for filename in os.listdir("Results/Test"): 
            next_frame = "Results/Test/" + filename
            images.append(imageio.imread(next_frame))

        for filename in os.listdir("Results/Test"):
            next_frame = "Results/Test/" + filename
            os.remove(next_frame)
            print('DELETED ', filename)
   
        
        #gif_name = 'Results/GIF_'+ str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H:%M')) +'.gif'
        #imageio.mimsave( gif_name, images)