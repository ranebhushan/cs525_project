#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os

class DQN(nn.Module):

    def __init__(self, 
        in_shape : np.ndarray = [0, 0, 0], 
        num_actions :int = 4) -> None:

        super(DQN, self).__init__()

        self.channels = in_shape[0]
        self.width = in_shape[1]
        self.height = in_shape[2]
        self.cnn = torch.nn.Sequential(
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

    # Calculates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, self.channels, self.width, self.height)
        x = self.cnn(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

    # # Save a model
    # def save_model(self):
    #     torch.save( self.state_dict(), os.path.join('.', self.directory_path, f'{self.filename}-model-{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pth'))
                                    
    # # Loads a model
    # def load_model(self):
    #     self.load_state_dict(torch.load(self.model_filename))