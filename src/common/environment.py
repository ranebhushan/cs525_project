import gym
import numpy as np
import highway_env

class Environment():
    def __init__(self, env_name, img_width=200, img_height=100,\
                     stack_size=4, scaling=1.75, lanes_count=4):
        self.img_width = img_width
        self.img_height = img_height
        self.stack_size = stack_size
        self.scale = scaling
        self.lanes_count = lanes_count
        self.env = gym.make(env_name)
        self.configure()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def configure(self):
        config = {
            "observation": {
                "type": "GrayscaleObservation",
                "weights": [0.9, 0.1, 0.5],  # weights for RGB conversion
                "stack_size": self.stack_size,
                "observation_shape": (self.img_width, self.img_height)
            },
            "screen_width": self.img_width,
            "screen_height": self.img_height,
            "scaling": self.scale,
            "lanes_count":self.lanes_count,
        }
        self.env.configure(config)
    
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