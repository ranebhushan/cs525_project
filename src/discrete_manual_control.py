import argparse
import highway_env
import numpy as np
import json
import keyboard
import time
import sys, os
sys.path.insert(1, os.getcwd() + '/src')
from common.environment import Environment

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project4")
    parser.add_argument("--env_config_path",
                    type=str,
                    default="configs/highway-env_config.json",
                    help="path for the environment config parameters")
    args = parser.parse_args()
    return args

total_episodes = 10

# ACTIONS_ALL = {
#         0: 'LANE_LEFT',
#         1: 'IDLE',
#         2: 'LANE_RIGHT',
#         3: 'FASTER',
#         4: 'SLOWER'
#     }

if __name__ == '__main__':
    args = parse()
    env_args = json.load(open(args.env_config_path))
    env = Environment("highway-v0", env_args)

    rewards = []
    for i in range(total_episodes):
        print('Starting New Episode')
        time.sleep(2)
        done = truncated = False
        obs, info = env.reset()
        env.render()
        episode_reward = 0.0

        while not (done or truncated):
            env.render()
            action = env.action_type('IDLE')
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            if done or truncated:
                print(f"Episode {i+1} reward: {episode_reward}")
        rewards.append(episode_reward)

    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards',rewards)