
#---------------------------------------------------
#       Scalable PG Main
#       Author: Chris Doyle
#---------------------------------------------------

#---------------------------------------------------------------------------------------------------
#  Requirements
#---------------------------------------------------------------------------------------------------

import grid_world as gw
from policy_gradient import PolicyGradient

# Bug Patch from https://github.com/MTG/sms-tools/issues/36
import matplotlib
matplotlib.use("TkAgg")
# End of patch 

import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------
#  Globals
#---------------------------------------------------------------------------------------------------

RENDER = False                  # Saves the episode as a .gif into the Results/Logged Runs/ Run_X
DISPLAY_REWARD = False          # Outputs reward and episode to the console
LOG_RUN = True                  # Whether of not you want to create a log file
render_X_times = 1              # Set to 1 if you do not require to render GIFs
display_reward_X_times = 10     # Set to 1 if you do not want to monitor training progress

env = gw.GridWorldEnv( num_episodes =500 )
env.seed(1)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

PG = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    gridworld_dims=env.dims,
    learning_rate=0.02,
    reward_decay=0.99,
    dir_for_run = env.dir_for_run
    # output_graph=True,
)

while(env.episode < env.num_episodes-1):

    observation = env.reset()

    # Set flags for display & render options
    RENDER, DISPLAY_REWARD = False, False
    if (env.episode % (env.num_episodes / render_X_times) == 0 and env.episode > 0): RENDER = True          
    if (env.episode % (env.num_episodes / display_reward_X_times) == 0 and env.episode > 0): DISPLAY_REWARD = True


    while True:
        action = PG.choose_action(observation,RENDER)

        observation_, reward, done, info = env.step(action)
        PG.store_transition(observation, action, reward, (RENDER and done))

        if RENDER: env.save_step()

        if done:
            ep_rs_sum = sum(PG.ep_rs)

            if env.episode == 0 :       
                running_reward = ep_rs_sum           
            else:
                running_reward_ = running_reward
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05     
                # gamma should depend on grid size. 0.95 -> 0.99 for smaller -> larger
    
            
            if DISPLAY_REWARD: print("episode:", env.episode, "  reward:", int(ep_rs_sum))

            vt = PG.learn()
            break

        observation = observation_

    if (RENDER): env.save_episode()

# Log episodes if desired
if LOG_RUN: env.log(PG.rewards,PG.running_average)

# Agent can now be considered trained
PG.save_policy()


