
#---------------------------------------------------
#       Scalable LLRL Main
#       Author: Chris Doyle
#---------------------------------------------------

#---------------------------------------------------------------------------------------------------
#  Requirements
#---------------------------------------------------------------------------------------------------

import grid_world as gw
from LLRL import PolicyGradient

# Bug Patch from https://github.com/MTG/sms-tools/issues/36
import matplotlib
matplotlib.use("TkAgg")
# End of patch 

import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------
#  Globals
#---------------------------------------------------------------------------------------------------

RENDER = False          # Saves the episode as a .gif into the Results/Logged Runs/ Run_X
DISPLAY_REWARD = False  # Outputs reward and episode to the console
LOG_RUN = True          # Whether of not you want to create a log file

env = gw.GridWorldEnv(100)
env.seed(1)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

LLRL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    gridworld_dims=env.dims,
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range( env.num_episodes ):

    observation = env.reset()

    # Set flags for display & render options
    RENDER, DISPLAY_REWARD = False, False
    if (i_episode % (env.num_episodes / 1) == 0 and i_episode > 0): RENDER = True          
    if (i_episode % (env.num_episodes / 10) == 0 and i_episode > 0): DISPLAY_REWARD = True


    while True:
        action = LLRL.choose_action(observation,RENDER)

        observation_, reward, done, info = env.step(action)
        LLRL.store_transition(observation, action, reward, (RENDER and done))

        if RENDER: env.save_step()

        if done:
            ep_rs_sum = sum(LLRL.ep_rs)

            if i_episode == 0 :       
                running_reward = ep_rs_sum           
            else:
                running_reward_ = running_reward
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05     
                if LOG_RUN: env.log(i_episode,running_reward_, running_reward, ep_rs_sum)
            
            if DISPLAY_REWARD: print("episode:", i_episode, "  reward:", int(running_reward))

            vt = LLRL.learn()
            break

        observation = observation_


    if (RENDER): env.save_episode(i_episode)


# Agent can now be considered trained
trained_policy = LLRL.return_policy()
