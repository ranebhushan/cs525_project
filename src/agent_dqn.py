#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import namedtuple, deque
import os
import sys
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from model.dqn import DQN
import csv
import gc
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done'))

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        gc.enable()
        state, _ = self.env.reset()
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
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        # Initialise policy and target networks, set target network to eval mode
        self.online_net = DQN(state.shape, self.env.action_space.n, filename=args['model_filename'])
        self.target_net = DQN(state.shape, self.env.action_space.n, filename=args['model_filename'])
        self.online_net = self.online_net.to(device=self.device)
        self.target_net = self.target_net.to(device=self.device)
        self.target_net.eval()
        
        if (not args['train']) or self.load_model:
            #you can load your model here
            ###########################
            # YOUR IMPLEMENTATION HERE #
            try:
                self.online_net.load_model()
                print('Loading trained model')
            except:
                print('Loading trained model failed')
                pass

        # Set target net to be the same as policy net
        self.replace_target_net(0)

        # Set optimizer & loss function
        self.optim = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()
        #self.loss = torch.nn.SmoothL1Loss()

    # Updates the target net to have same weights as policy net
    def replace_target_net(self, steps):
        if steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            print('Target network replaced')
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    def state_to_tensor(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return state_t

    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if random.random() > self.epsilon or test:
            observation_t = self.state_to_tensor(observation)
            q_values = self.online_net(observation_t.unsqueeze(0))
            max_q_index = torch.argmax(q_values, dim=1)
            action = max_q_index.item()
        else:
            action = random.randint(0, self.env.action_space.n - 1)
        ###########################
        return action
    
    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer_replay.append(Transition(*args))
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        batch = random.sample(self.buffer_replay, self.batch_size)
        # Converts batch of transitions to transitions of batches
        batch = Transition(*zip(*batch))
        # Convert to tensors with correct dimensions
        state = torch.cat([self.state_to_tensor(s).unsqueeze(0) for s in batch.state]).float().to(self.device)
        action = torch.tensor(batch.action).view(-1,1).to(self.device)
        reward = torch.tensor(batch.reward).float().to(self.device)
        next_state = torch.cat([self.state_to_tensor(s).unsqueeze(0) for s in batch.next_state]).float().to(self.device)
        done = torch.tensor(batch.done).float().to(self.device)

        del batch
        ###########################
        return state, action, reward, next_state, done

    def optimize_model(self):
        if len(self.buffer_replay) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer()
        q_values_online = self.online_net(state)
        q_val_next = self.online_net(next_state).detach()
        q_val_next_target = self.target_net(next_state).detach()
        with torch.no_grad():
            q_actions = torch.max(q_val_next, dim=1)[1].unsqueeze(1)
            q_max = q_val_next_target.gather(dim=1, index=q_actions).squeeze()
            q_target = reward + self.gamma * q_max * (1 - done)
        
        q_val = q_values_online.gather(dim=1, index=action).squeeze()

        # # Calculate the value of the action taken
        # q_eval = self.online_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        # # Calculate best next action value from the target net and detach from graph
        # q_next = self.target_net(next_state).detach().max(1)[0]
        # # Using q_next and reward, calculate q_target
        # # (1-done) ensures q_target is reward if transition is in a terminating state
        # q_target = reward + q_next * self.gamma * (1 - done)
        # Compute the loss
        loss = self.loss(q_val, q_target).to(self.device)
        # Perform backward propagation and optimization step
        self.optim.zero_grad()
        self.online_net.zero_grad()
        loss.backward()

        if self.clip:
            for param in self.online_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optim.step()

        return loss.item()

    # Decrement epsilon 
    def dec_eps(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps / self.epsilon_decay)
        self.steps += 1

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        mean_score = 0
        max_score = 0
        i_episode = 0

        while mean_score < 60:
            # Initialize the environment and state
            current_state, _ = self.env.reset()
            done = False
            episode_score = 0
            loss = 0
            while not done:
                # Select and perform an action
                action = self.make_action(current_state, False)
                next_state, reward, done, _, _ = self.env.step(action)
                self.push(current_state, action, reward, next_state, int(done))
                current_state = next_state
                # print(len(self.buffer_replay))
                if len(self.buffer_replay) > self.start_learning:
                    # Decay epsilon
                    self.dec_eps()
                    if self.steps % self.train_frequency == 0:
                        loss = self.optimize_model()
                else:
                    i_episode = 0
                    continue
                
                # Add the reward to the previous score
                episode_score += reward
                # Update target network
                self.replace_target_net(self.steps)

            i_episode += 1
            max_score = episode_score if episode_score > max_score else max_score
            self.scores.append(episode_score)
            self.rewards.append(episode_score)
            mean_score = np.mean(self.scores)

            if len(self.buffer_replay) > self.start_learning:
                print('Episode: ', i_episode, ' Score:', episode_score, ' Avg Score:',round(mean_score,4),' Epsilon: ', round(self.epsilon,4), ' Loss:', round(loss,4), ' Max Score:', max_score)
                if i_episode % self.reward_save_frequency == 0:
                    with open('logs/' + self.csv_file_name, mode='a') as dataFile:
                        rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        rewardwriter.writerow([np.mean(self.rewards)])
            else:
                print('Gathering Data . . .')

            if i_episode % self.model_save_frequency == 0:
                # Save model
                self.online_net.save_model()

        print('======== Complete ========')
        self.online_net.save_model()
        ###########################