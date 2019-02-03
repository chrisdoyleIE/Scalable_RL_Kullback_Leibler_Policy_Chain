
#---------------------------------------------------
#       Test Created Classes for gridWorld & co.
#       Author: Chris Doyle
#---------------------------------------------------

#---------------------------------
# Importing the gridWorld class
#---------------------------------

import grid_world as gw
import numpy as np

state = np.zeros(2)
env = gw.GridWorldEnv()
env.reset()

done = False
while not done:
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()





#env.render()