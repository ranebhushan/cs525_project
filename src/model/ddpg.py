import os
import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

class ActorNetwork(torch.nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, directory_path= 'weights/',filename=''):

        super(ActorNetwork,self).__init__()
        self.filename = 'ddpg'
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = directory_path
        self.checkpoint = os.path.join(self.checkpoint_dir, name+'_ddpg')
        self.model_filename = filename
        self.fc1 = torch.nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = torch.nn.LayerNorm(self.fc1_dims)
        self.bn2 = torch.nn.LayerNorm(self.fc2_dims)
        self.mu = torch.nn.Linear(self.fc2_dims, self.n_actions)
        # self.action_value = torch.nn.Linear(self.n_actions, self.fc2_dims)
        # self.q = torch.nn.Linear(self.fc2_dims, 1)
        f1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1,f1)
        self.fc1.bias.data.uniform_(-f1,f1)
        f2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2,f2)
        self.fc2.bias.data.uniform_(-f2,f2)
        f3 = 0.003
        self.mu.weight.data.uniform_(-f3,f3)
        self.mu.bias.data.uniform_(-f3,f3)

        self.optimizer = optim.Adam(self.parameters(),lr = lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        state_val = self.fc1(state)
        state_val = self.bn1(state_val)
        state_val = F.relu(state_val)
        state_val = self.fc2(state_val)
        state_val = self.bn2(state_val)
        state_val = F.relu(state_val)
        state_val = torch.tanh(self.mu(state_val))
        return state_val
    
    def save_checkpoint(self):
        print('...........saving checkpoint...................')
        torch.save( self.state_dict(), os.path.join(self.checkpoint_dir, f'{self.filename}-{self.name}-model-{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pth'))
    
    def load_checkpoint(self):
        print('.............loading checkpoint..................')
        self.load_state_dict(torch.load(self.model_filename))
    
class CriticNetwork(torch.nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, directory_path= 'weights/',filename=''):
        super(CriticNetwork,self).__init__()
        self.filename = 'ddpg'
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = directory_path
        self.checkpoint = os.path.join(self.checkpoint_dir, name+'_ddpg')
        self.model_filename = filename
        self.fc1 = torch.nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = torch.nn.LayerNorm(self.fc1_dims)
        self.bn2 = torch.nn.LayerNorm(self.fc2_dims)

        self.action_value = torch.nn.Linear(self.n_actions, self.fc2_dims)
        self.q = torch.nn.Linear(self.fc2_dims, 1)
        f1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1,f1)
        self.fc1.bias.data.uniform_(-f1,f1)
        f2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2,f2)
        self.fc2.bias.data.uniform_(-f2,f2)
        f3 = 0.003
        self.q.weight.data.uniform_(-f3,f3)
        self.q.bias.data.uniform_(-f3,f3)
        f4 = 1.0/np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4,f4)
        self.action_value.bias.data.uniform_(-f4,f4)
        self.optimizer = optim.Adam(self.parameters(),lr = lr, weight_decay = 0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state, action):
        state_val = self.fc1(state)
        state_val = self.bn1(state_val)
        state_val = F.relu(state_val)
        state_val = self.fc2(state_val)
        state_val = self.bn2(state_val)
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_val, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value
    
    def save_checkpoint(self):
        print('...........saving checkpoint...................')
        torch.save( self.state_dict(), os.path.join('.', self.checkpoint_dir, f'{self.filename}-{self.name}-model-{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pth'))
    
    def load_checkpoint(self):
        print('.............loading checkpoint..................')
        self.load_state_dict(torch.load(self.model_filename))