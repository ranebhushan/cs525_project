import argparse
import yaml

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
    print(dict_args)

if __name__ == '__main__':
    main()