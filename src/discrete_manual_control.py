import gym
import highway_env
from matplotlib import pyplot as plt
import json
import pprint
import os
import keyboard

env = gym.make('highway-v0')

# config = json.load(open("../configs/highway-env_config.json"))
# config['observation']['observation_shape'] = tuple(config['observation']['observation_shape'])
# env.configure(config)

env.reset()

# ACTIONS_ALL = {
#         0: 'LANE_LEFT',
#         1: 'IDLE',
#         2: 'LANE_RIGHT',
#         3: 'FASTER',
#         4: 'SLOWER'
#     }
a = 0
while True:
  done = truncated = False
  obs, info = env.reset()
  action = env.action_type.actions_indexes["IDLE"]
  while not (done or truncated):
    if keyboard.is_pressed('a'):
        action = env.action_type.actions_indexes["LANE_LEFT"]
        print(a, "left")
        a+=1
    elif keyboard.is_pressed('d'):
        action = env.action_type.actions_indexes["LANE_RIGHT"]
        print(a, "right")
        a+=1
    elif keyboard.is_pressed('w'):
        action = env.action_type.actions_indexes["FASTER"]
        print(a, "faster")
        a+=1
    elif keyboard.is_pressed('s'):
        action = env.action_type.actions_indexes["SLOWER"]
        print(a, "slower")
        a+=1
    else:
        action = env.action_type.actions_indexes["IDLE"]
        print(a, "idle")
        a+=1
    obs, reward, done, truncated, info = env.step(action)
    env.render()
