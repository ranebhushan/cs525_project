#!/usr/bin/env python3

import torch
import numpy as np

class DQN(torch.nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

    def forward(self, x):
        return x