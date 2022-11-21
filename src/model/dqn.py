#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_shape=[0, 0, 0], num_actions=4, filename='test'):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.channels = in_shape[0]
        self.width = in_shape[1]
        self.height = in_shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_output_size = self.conv_output_dim()

        self.linear = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Save filename for saving model
        self.filename = filename

    # Calculates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, self.channels, self.width, self.height)
        x = self.cnn(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        ###########################
        return x

    # Save a model
    def save_model(self):
        torch.save(self.state_dict(), 'weights/' + self.filename + '.pth')

    # Loads a model
    def load_model(self):
        self.load_state_dict(torch.load('weights/' + self.filename + '.pth'))