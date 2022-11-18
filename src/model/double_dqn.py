#!/usr/bin/env python3

import torch
import numpy as np

class DoubleDQN(torch.nn.Module):

    def __init__(self):
        super(DoubleDQN, self).__init__()

    def forward(self, x):
        return x