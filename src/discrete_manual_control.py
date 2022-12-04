import gym
import highway_env
from matplotlib import pyplot as plt
import json
import pprint
import os
import keyboard

env = gym.make('highway-v0')

config = json.load(open("highway-env_config.json"))
config['observation']['observation_shape'] = tuple(config['observation']['observation_shape'])
env.configure(config)
env.reset()

# ACTIONS_ALL = {
#         0: 'LANE_LEFT',
#         1: 'IDLE',
#         2: 'LANE_RIGHT',
#         3: 'FASTER',
#         4: 'SLOWER'
#     }



# while True : pass
a = 0
while True:
#   done = truncated = False
#   obs, info = env.reset()
#   while not (done or truncated):
    if keyboard.on_press('left'):
        action = env.action_type.actions_indexes["LANE_LEFT"]
        print(a, "left")
        a+=1
    elif keyboard.on_press('right'):
        action = env.action_type.actions_indexes["LANE_RIGHT"]
        print(a, "right")
        a+=1
    elif keyboard.on_press('up'):
        action = env.action_type.actions_indexes["FASTER"]
        print(a, "up")
        a+=1
    elif keyboard.on_press('down'):
        action = env.action_type.actions_indexes["SLOWER"]
        print(a, "down")
        a+=1
    else:
        action = env.action_type.actions_indexes["IDLE"]
    # obs, reward, done, truncated, info = env.step(action)
    # env.render()
