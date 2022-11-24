#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from collections import namedtuple
import math

import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

from agent import Agent
from model.dqn_model import DQN
import time

import os

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
        
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)

# GAMMA = 0.99
# EPISODES = 100000
# EPSILON = 0.02
# EPS_START = EPSILON
# EPS_END = 0.005
# EPS_DECAY = 1000
# batch_size = 32
# ALPHA = 1e-5
# TARGET_UPDATE = 20000

# CAPACITY = 500
# memory = deque(maxlen=CAPACITY)
# storeEpsilon = []
# StartLearning = 100
# LOAD = False
# device = torch.device("cpu")

class Agent_DQN():
    def __init__(self, env, args):
        # Parameters for q-learning

        super(Agent_DQN,self).__init__()

        self.env = env
        state,_ = self.env.reset()
        self.epsilon_start = args['epsilon_start']
        self.epsilon_end = args['epsilon_end']
        self.epsilon_decay = args['epsilon_decay']
        self.epsilon = self.epsilon_start
        self.gamma = args['gamma']
        self.batch_size = args['batch_size']
        self.buffer_size = args['buffer_size']
        self.learning_rate = args['learning_rate']
        self.num_frames = state.shape[0]
        self.steps = 0
        self.target_update_frequency = args['target_update_frequency']
        self.start_learning = args['start_learning']
        self.model_save_frequency = args['model_save_frequency']
        self.load_model = args['load_model']
        self.clip = args['clip_gradients']
        self.reward_save_frequency = args['reward_save_frequency']
        self.train_frequency = args['train_frequency']
        self.csv_file_name = args['csv_file_name']
        try:
            os.remove('logs/' + self.csv_file_name)
        except FileNotFoundError:
            pass
        self.buffer_replay = deque(maxlen=self.buffer_size)
        self.scores = deque(maxlen=100)
        self.rewards = deque(maxlen=self.reward_save_frequency)
        self.storeEpsilon = []
        self.policy_net = DQN(state.shape, self.env.action_space.n,filename=args['model_filename']) 
        self.target_net = DQN(state.shape, self.env.action_space.n,filename=args['model_filename'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if (not args['train']) or self.load_model:
            print('loading trained model')
            self.policy_net.load_model()
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def init_game_setting(self):
        print('loading trained model')
        self.policy_net.load_model()    
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer_replay.append((state, action, reward, next_state, done))
    
    def replay_buffer(self):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer_replay, self.batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
    def make_action(self, observation, test=True):
        if np.random.random() > self.epsilon or test==True:
            observation = Variable(torch.FloatTensor(np.float32(observation)).unsqueeze(0), volatile=True)
            q_value = self.policy_net.forward(observation)
            action  = q_value.max(1)[1].data[0]
            action = int(action.item())            
        else:
            action = random.randrange(4)
        return action

    def optimize_model(self):

        states, actions, next_states, rewards, dones  = self.replay_buffer()

        states_v = Variable(torch.FloatTensor(np.float32(states)))
        next_states_v = Variable(torch.FloatTensor(np.float32(next_states)), volatile=True)
        actions_v = Variable(torch.LongTensor(actions))
        rewards_v = Variable(torch.FloatTensor(rewards))
        done = Variable(torch.FloatTensor(dones))

        state_action_values = self.policy_net(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_net(next_states_v).max(1)[0]
        expected_q_value = rewards_v + next_state_values * self.gamma * (1 - done)

        loss = (state_action_values - Variable(expected_q_value.data)).pow(2).mean()
        return loss
        
        
    def train(self):
        print("TRAINING STARTED")
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        meanScore = 0
        AvgRewards = []
        AllScores = []
        step = 1
        iEpisode = 0

        while True:
                     
            state,_ = self.env.reset()
            done = False
            EpisodeScore = 0
            tBegin = time.time()
            done = False
            print(iEpisode)
            while not done:

                action = self.make_action(state)    
                nextState, reward, done, _, _ = self.env.step(action)
                self.push(state, action, nextState, reward, done)
                state = nextState
                self.env.render()
                if len(self.buffer_replay) > self.start_learning:
                    loss = self.optimize_model()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    iEpisode = 0
                    continue        
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * step/self.epsilon_decay)
                self.storeEpsilon.append(self.epsilon)
                step += 1
                EpisodeScore += reward
                if step % self.target_update_frequency == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            iEpisode += 1
            AllScores.append(EpisodeScore)
            meanScore = np.mean(AllScores[-100:])
            AvgRewards.append(meanScore)
            self.scores.append(EpisodeScore)
            self.rewards.append(EpisodeScore)
            
            if len(self.buffer_replay) > self.start_learning: 
                print('Episode: ', iEpisode+1, ' score:', EpisodeScore, ' Avg Score:',meanScore,' epsilon: ', self.epsilon, ' t: ', time.time()-tBegin, ' loss:', loss.item())
                if iEpisode % self.reward_save_frequency == 0:
                    with open('logs/' + self.csv_file_name, mode='a') as dataFile:
                        rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        rewardwriter.writerow([np.mean(self.rewards)])
            else:
                print('Gathering Data . . .')


            if iEpisode % 500 == 0:
                torch.save({'model_state_dict': self.policy_net.state_dict()}, 'model.pth')

        torch.save({'model_state_dict': self.policy_net.state_dict()}, 'model.pth')

