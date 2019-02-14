
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

DISPLAY_REWARD_THRESHOLD = 0  # renders environment if total episode reward is greater then this threshold
FLAG = False
CREATE_DIR = True

env = gw.GridWorldEnv()
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

for i_episode in range(10000):

     

    observation = env.reset()

    # Set flag for display option
    flag = False
    if (i_episode % 100 == 0 and i_episode > 0): flag = True


    while True:
        action = LLRL.choose_action(observation,flag)

        observation_, reward, done, info = env.step(action)
        LLRL.store_transition(observation, action, reward, (flag and done))

        if flag: env.save_step()

        if done:
            ep_rs_sum = sum(LLRL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward_ = running_reward
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05     
                env.log(i_episode,running_reward_, running_reward, ep_rs_sum)
            
            if flag: print("episode:", i_episode, "  reward:", int(running_reward),'\n')

            vt = LLRL.learn()
            break

        observation = observation_


    if (flag): 
        env.save_episode(i_episode, CREATE_DIR)
        CREATE_DIR = False      # One directory per running of script


# Agent can now be considered trained
trained_policy = LLRL.return_policy()
