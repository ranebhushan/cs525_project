import gym
import numpy as np
import highway_env

class Environment():
    def __init__(self, env_name, env_config=None):
        self.env_config = env_config
        self.env = gym.make(env_name)
        self.configure()
        observation, info = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def configure(self):
        if self.env_config:
            try:
                self.env_config['observation']['observation_shape'] = tuple(self.env_config['observation']['observation_shape'])
            except KeyError:
                pass
            self.env.configure(self.env_config)
    
    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)

    def reset(self):
        observation, info = self.env.reset()
        return observation, info
    
    def step(self,action):
        if not self.env.action_space.contains(action):
            raise ValueError('Invalid action!!')
        observation, reward, done, truncated, info = self.env.step(action)
        return observation, reward, done, truncated, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()
    
    def render(self):
        self.env.render()
    
    def action_type(self, action):
        action = self.env.action_type.actions_indexes[action]
        return action
