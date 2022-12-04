#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np

class DuelingDQN(torch.nn.Module):

    def __init__(self, 
        in_shape : np.ndarray = [0, 0, 0], 
        num_actions :int = 4) -> None:

        super(DuelingDQN, self).__init__()

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
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, num_actions)
    
    # Calculates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, self.channels, self.width, self.height)
        x = self.cnn(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = torch.div(x, 255.0)
        x = x.contiguous()
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        V = self.V(x)
        A = self.A(x)
        x = torch.add(V, (A - A.mean(dim=1, keepdim=True)))
        return x
