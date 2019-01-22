#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:31:09 2019

@author: CP
"""

#--------------------------------------------------
# 1. Create a Heatmap / GridWorld with matshow()
#--------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Create a Gridworld
def gridWorld(dims):
   
    # Define the boundaries of the GW to be 0, and all other squares 1
    GW = np.ones(dims)
    m = dims[0]
    n = dims[1]
    for i in range(m): 
        for j in range(n): 
            if i == 0: 
                GW[i][j] = 0
            elif i == m-1: 
                GW[i][j] = 0
            elif j == 0: 
                GW[i][j] = 0
            elif j == n-1: 
                GW[i][j] = 0
    return GW

# Randomly Allocate an Agent in a GridWorld
def placeAgent(GW):
    agent_position = (np.random.randint(1,GW.shape[0]-1) , # x co-ordinate
                      np.random.randint(1,GW.shape[1]-1) ) # y co-ordinate
                      
    GW[agent_position[0]][agent_position[1]] = 0.25        # Set to Red
    return GW, agent_position

# Randomly Allocate a reward in a GridWorld
def placeReward(GW, agent_pos):
    reward_position = agent_pos
    while reward_position == agent_pos:
        reward_position = (np.random.randint(1,GW.shape[0]-1) , # x co-ordinate
                           np.random.randint(1,GW.shape[1]-1) ) # y co-ordinate
        
    GW[reward_position[0]][reward_position[1]] = 0.75           # Set to Yellow
    return GW
    
############
# __MAIN__
############

for i in range(0,5):
    # Initilialise GridWorld
    GW = gridWorld((7,7))
    GW, agent_position = placeAgent(GW)
    GW = placeReward(GW, agent_position)
    
    # Display matrix
    plt.matshow(GW, cmap = 'hot', interpolation='none', vmin=0, vmax=1)
    
    # Re-centre pixels such that the grid sepparates them as desired
    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor')


