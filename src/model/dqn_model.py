#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.autograd as autograd
# from torchsummary import summary
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, filename='test'):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.filename = filename
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # summary(self.features,self.input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = x.to(device)
        op = self.features(x)
        op = op.reshape(x.size(0), -1)
        x = self.fc(op)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def save_model(self):
        torch.save(self.state_dict(), 'weights/' + self.filename + '.pth')

    # Loads a model
    def load_model(self):
        self.load_state_dict(torch.load('weights/' + self.filename + '.pth'))
    
