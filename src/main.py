import argparse
import yaml
from common.environment import Environment
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path",
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
    train_args = load_yaml(args.train_config_path)
    env_args = json.load(open(args.env_config_path))
    env = Environment("highway-v0", env_args)
    from agent_dqn import Agent_DQN
    agent = Agent_DQN(env, train_args)
    agent.train()

if __name__ == '__main__':
    main()