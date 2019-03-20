import grid_world as gw
from policy_gradient_keras import PolicyGradient
import numpy as np 

# Bug Patch from https://github.com/MTG/sms-tools/issues/36
import matplotlib
matplotlib.use("TkAgg")
# End of patch 

import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------
#  Initialise Globals
#---------------------------------------------------------------------------------------------------

RENDER = False                  # Saves the episode as a .gif into the Results/Logged Runs/ Run_X
DISPLAY_REWARD = False          # Outputs reward and episode to the console
LOG_RUN = True                  # Whether of not you want to create a log file
render_X_times = 5              # Set to 1 if you do not require to render GIFs
display_reward_X_times = 10   # Set to 1 if you do not want to monitor training progress

env = gw.GridWorldEnv( num_episodes =1000, default_reward_pos =  np.array([3,3]), training= False  )
env.seed(1)     # reproducible, general Policy gradient has high variance


PG = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    gridworld_dims=env.dims,
    learning_rate=0.02,
    reward_decay=0.99,
    dir_for_run = env.dir_for_run
    # output_graph=True,
)

