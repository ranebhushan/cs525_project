echo "RL Job running on $(hostname)"

echo "Running Python Code"

python3 test/test.py --train_config_path=configs/DQN.yaml --env_config_path=configs/highway-env_config.json