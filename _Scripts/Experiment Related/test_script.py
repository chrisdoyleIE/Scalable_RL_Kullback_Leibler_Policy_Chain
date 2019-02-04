
#---------------------------------------------------
#       Test Created Classes for gridWorld & co.
#       Author: Chris Doyle
#---------------------------------------------------

#---------------------------------
# Importing the gridWorld class
#---------------------------------

# import grid_world as gw
# import numpy as np

# state = np.zeros(2)
# env = gw.GridWorldEnv()

# for i in range(5):

#     env.reset()
#     done = False
#     while not done:
#         state, reward, done, info = env.step(env.action_space.sample())
#         #env.save_step()
    
#     #env.save_episode()


import grid_world as gw
from RL_brain import PolicyGradient

# Bug Patch from https://github.com/MTG/sms-tools/issues/36
import matplotlib
matplotlib.use("TkAgg")
# End of patch 

import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 0  # renders environment if total episode reward is greater then this threshold
FLAG = False

env = gw.GridWorldEnv()
env.seed(1)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(10000):

    observation = env.reset()

    # Set flag for display option
    flag = False
    if (i_episode % 1000 == 0 and i_episode > 0): flag = True
    #flag = True


    while True:
        action = RL.choose_action(observation,flag)

        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward, (flag and done))

        if flag: env.save_step()
        #env.save_step()

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.01
            if flag: 
                print("episode:", i_episode, "  reward:", int(running_reward),'\n')

            vt = RL.learn()
            break

        observation = observation_


    if (flag): env.save_episode(i_episode)
    #env.save_episode(i_episode)