
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

for i in range(0,2):

    env.reset()
    done = False
    env.save_step()

    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        env.save_step()

    env.save_episode()





#env.render()