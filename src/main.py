import argparse
import yaml
from common.environment import Environment
from agent_dddqn import Agent_DDDQN
from agent_dqn import Agent_DQN
import sys, os
sys.path.insert(1, os.getcwd() + '/test')
from test import test
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_config_path",
                        type=str,
                        default="configs/DQN.yaml",
                        help="path for the training parameters")
    parser.add_argument("--env_config_path",
                        type=str,
                        default="configs/highway-env_config.json",
                        help="path for the environment config parameters")

    args = parser.parse_args()
    return args

def load_yaml(yaml_path):
    dict_args = {}
    with open(yaml_path, 'r') as stream:
        dictionary = yaml.safe_load(stream)
        for key, val in dictionary.items():
            dict_args[key] = val
    return dict_args


def main():
    args = parse_args()
    agent_args = load_yaml(args.agent_config_path)
    env_args = json.load(open(args.env_config_path))
    env = Environment("highway-v0", env_args)
    agent = None
    if agent_args['model_name'] == 'DDDQN':
        agent = Agent_DDDQN(env, agent_args)
    elif agent_args['model_name'] == 'DQN':
        agent = Agent_DQN(env, agent_args)
    else:
        print('Invalid Model Name')
        sys.exit()
        
    if agent_args['train']:
        agent.train()
    else:
        test(agent, env, total_episodes=100, record_video=False, render=agent_args['render'])

if __name__ == '__main__':
    main()