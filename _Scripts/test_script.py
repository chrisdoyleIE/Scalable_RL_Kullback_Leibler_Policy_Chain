
#---------------------------------------------------
#       Test Created Classes for gridWorld & co.
#       Author: Chris Doyle
#---------------------------------------------------

#---------------------------------
# Importing the gridWorld class
#---------------------------------

import grid_world as gw

gridWorld = gw.gridWorld((7,7),'nipy_spectral',0.5,0.5,0.5)
gridWorld.agent__init__()
gridWorld.print()