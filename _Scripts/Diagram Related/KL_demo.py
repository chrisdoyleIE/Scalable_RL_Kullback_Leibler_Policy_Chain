
#-----------------------------------------------
#   IMPORT LIBRARIES & MODULES
#-----------------------------------------------

import matplotlib as plt           # Plotting
import seaborn as sns              # Distribution Plot Properties
import numpy as np                 # Arrays & Normalising
from scipy.stats import uniform

#-----------------------------------------------
#   DEFINE FUNCTIONS
#-----------------------------------------------

def normalize(array):
    
    array = array - np.mean(array)
    
    # Assume non-Uniform as default
    if np.std(array) != 0:
        array = array / np.std(array)
    else:
        array = array / array.shape[0]
    
    return array

#-----------------------------------------------
#   DEFINE ACTION SPACE A
#-----------------------------------------------

A = ['up','down','left','right']

#-----------------------------------------------
#   DEFINE POSSIBLE P(A|s)
#-----------------------------------------------

# Uniform
uniform_policy = normalize([1,1,1,1])

print(uniform_policy)

