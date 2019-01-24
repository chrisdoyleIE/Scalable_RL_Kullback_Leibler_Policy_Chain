
#---------------------------------------------------
#       Definitions of GridWorld for LLRL
#       Author: Chris Doyle
#---------------------------------------------------


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class gridWorld:

    #---------------------------
    # Objects Attributes
    #---------------------------

    # - colormap = 'spectral'   # Theme for plt.matplot()
    # - GW = np.ones()          # Actual GridWorld; 1.0 = grey in spectral colormap

    # Hidden Attributes:
    # - agent_position = (x,y)
    # - boundary_color = 0.95   # Red in spectral
    # - agent_color = 0.25      # XX in spectral
    # - reward_color = 0.75     # XX in spectral

    #---------------------------
    # Function Attributes
    #---------------------------

    # F1: Constructor
    def __init__(self,dims,colormap,boundary_col,agent_col,reward_col):

        # Initialise Attributes
        m = dims[0]
        n = dims[1]
        self.GW = np.ones(dims)
        self.colormap = colormap
        self.boundary_color = float(boundary_col)
        self.agent_color = float(agent_col)
        self.reward_color = float(reward_col)

        # Initialise Boundary of GridWorld
        for i in range(m): 
            for j in range(n): 
                if i == 0: 
                    self.GW[i][j] = self.boundary_color
                elif i == m-1: 
                    self.GW[i][j] = self.boundary_color
                elif j == 0: 
                    self.GW[i][j] = self.boundary_color
                elif j == n-1: 
                    self.GW[i][j] = self.boundary_color
    
    # F2: Initialise/Reset agent to random location in GridWorld
    def agent__init__(self):
        print('FUNCTION CALL: agent__init__(self)')
        # Ensure random position is inside the boundary
        self.agent_position = (np.random.randint(1,self.GW.shape[0]-1) , # x co-ordinate
                               np.random.randint(1,self.GW.shape[1]-1) ) # y co-ordinate
        
        self.GW[self.agent_position[0]][self.agent_position[1]] = self.agent_color
 
    # F3: Initialise/Reset reward to random location in GridWorld
    def reward__init__(self):
        print('FUNCTION CALL: reward__init__(self)')

        self.reward_position = self.agent_position

        # Ensure that reward position is different to starting position
        while self.reward_position == self.agent_position:
            self.reward_position = (np.random.randint(1,self.GW.shape[0]-1) , # x co-ordinate
                                    np.random.randint(1,self.GW.shape[1]-1) ) # y co-ordinate
        
        self.GW[self.reward_position[0]][self.reward_position[1]] = self.reward_color
    
    # F4: Print the GridWorld
    def print(self):
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
        plt.show()

        
#################
# TEST MAIN
#################

print('IMPORT: grid_world.py')