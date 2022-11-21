import argparse
import yaml
from common.environment import Environment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path",
                        type=str,
                        default="configs/DQN.yaml",
                        help="yaml path for the run")

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
    dict_args = load_yaml(args.yaml_path)
    env = Environment("highway-v0")
    from agent_dqn import Agent_DQN
    agent = Agent_DQN(env, dict_args)
    agent.train()

if __name__ == '__main__':
    main()