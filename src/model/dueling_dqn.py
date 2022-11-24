#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelingDQN(torch.nn.Module):

    def __init__(self, in_shape=[0, 0, 0], num_actions=4, filename='test'):
        super(DuelingDQN, self).__init__()
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
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    # Calculates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, self.channels, self.width, self.height)
        x = self.cnn(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        values = self.value_stream(x)
        advantage = self.advantage_stream(x)
        x = values + (advantage - advantage.mean())
        return x

    # Save a model
    def save_model(self):
        torch.save(self.state_dict(), 'weights/' + self.filename + '.pth')

    # Loads a model
    def load_model(self):
        self.load_state_dict(torch.load('weights/' + self.filename + '.pth'))