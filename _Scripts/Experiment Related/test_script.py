
#---------------------------------------------------
#       Test Created Classes for gridWorld & co.
#       Author: Chris Doyle
#---------------------------------------------------

#---------------------------------
# Importing the gridWorld class
#---------------------------------

import grid_world as gw

env = gw.gridWorld((7,7),
                        colormap = 'nipy_spectral',
                        boundary_col = 0.95,
                        agent_col = 0.25,
                        reward_col = 0.75)

env.agent__init__()
env.reward__init__()
env.render()