import argparse
import numpy as np
import sys, os
sys.path.insert(1, os.getcwd() + '/src')
from common.environment import Environment
import time
from gym.wrappers.monitoring import video_recorder
from tqdm import tqdm
import json
from agent_dqn import Agent_DQN
import yaml

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project4")
    parser.add_argument("--env_config_path",
                    type=str,
                    default="configs/highway-env_config.json",
                    help="path for the environment config parameters")
    parser.add_argument("--train_config_path",
                        type=str,
                        default="configs/DQN.yaml",
                        help="path for the training parameters")
    args = parser.parse_args()
    return args

def load_yaml(yaml_path):
    dict_args = {}
    with open(yaml_path, 'r') as stream:
        dictionary = yaml.safe_load(stream)
        for key, val in dictionary.items():
            dict_args[key] = val
    return dict_args


def test(agent, env, total_episodes=30, record_video=False):
    rewards = []
    if record_video:
        vid = video_recorder.VideoRecorder(env=env.env, path="videos/test_vid.mp4")
    start_time = time.time()
    for i in tqdm(range(total_episodes)):
        frames= 0
        state, _ = env.reset()
        agent.init_game_setting()
        episode_reward = 0.0

        #playing one game
        #frames = [state]
        terminated, truncated = False, False
        while not terminated and not truncated:
            env.render()
            frames += 1
            action = agent.make_action(state, test=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            # env.render()
            #frames.append(state)
            if record_video:
                vid.capture_frame()
            if terminated or truncated:
                ###############################################################################
                ''' May not need to show this part when testing. (Just to show if stop because of 
                Time-limit, i.e., infinite state-action loops)
                ''' 
                if truncated is True:
                    print("Truncated: ", truncated)
                print(f"Episode {i+1} reward: {episode_reward}")
                ###############################################################################
                break
        rewards.append(episode_reward)

    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards',rewards)
    print('running time',time.time()-start_time)

def run(env_args, test_args):
    env = Environment("highway-v0", env_args)
    agent = Agent_DQN(env, test_args)
    test(agent, env, total_episodes=100, record_video=False)


if __name__ == '__main__':
    args = parse()
    test_args = load_yaml(args.train_config_path)
    env_args = json.load(open(args.env_config_path))
    run(env_args, test_args)
