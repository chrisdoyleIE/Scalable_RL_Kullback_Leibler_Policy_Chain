# -------------------------------------
# GridWorld Module
# -------------------------------------

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
%pylab inline

# -------------------------------------
# Define GridWorld Values
# -------------------------------------

DEFAULT_VALUE = -1
EDGE_VALUE = -10
GOAL_VALUE = 10

# -------------------------------------
# Relevant Functions
# -------------------------------------

class GridWorld():
    # Function to initialise GridWorld
    def __init__(self, grid_size = [10,10]):
        self.grid_size = grid_size

    
plt.matshow(10,10)