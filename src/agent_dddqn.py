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
from model.dueling_dqn import DuelingDQN
import csv
import gc
from datetime import datetime
from torchsummary import summary

Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done'))

class Memory(object):
    def __init__(self, memory_size: int, device) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        self.device = device

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indexes]
        # Converts batch of transitions to transitions of batches
        batch = Transition(*zip(*batch))
        # Convert to tensors with correct dimensions
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        actions = torch.FloatTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)
        del indexes
        del batch
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()

class Agent_DDDQN(Agent):
    def __init__(self, env, args):
        super(Agent_DDDQN,self).__init__(env)
        ###########################
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
        self.total_episodes = args['total_episodes']
        self.model_save_frequency = args['model_save_frequency']
        self.load_model = args['load_model']
        self.clip = args['clip_gradients']
        self.reward_save_frequency = args['reward_save_frequency']
        self.train_frequency = args['train_frequency']
        self.model_name = args['model_name']
        self.trained_model_name = args['trained_model_folder_and_filename']
       
        self.current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.directory_path = os.path.join("weights", f"{self.model_name}-{self.current_time}")
        self.csv_filename = os.path.join('logs', f'{self.model_name}-{self.current_time}.csv')

        if(os.path.exists(self.directory_path) == False):
            os.mkdir(self.directory_path)

        with open(self.csv_filename, mode='w', newline="") as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(["Date Time", "Episode", "Reward", "Epsilon", "Loss", "Max. Reward", "Mean Reward"])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.buffer_replay = Memory(self.buffer_size, self.device)
        self.scores = deque(maxlen=100)
        self.rewards = deque(maxlen=self.reward_save_frequency)
        # Initialise policy and target networks
        self.online_net = DuelingDQN(state.shape, self.env.action_space.n)
        # summary(self.online_net, (4, 600, 150))
        self.target_net = DuelingDQN(state.shape, self.env.action_space.n)

        self.online_net = self.online_net.to(device=self.device)
        self.target_net = self.target_net.to(device=self.device)
        
        if (not args['train']) or self.load_model:
            try:
                self.online_net.load_state_dict(torch.load(os.path.join("weights", f'{self.trained_model_name}')))
                print('Loaded trained model')
            except:
                print('Loading trained model failed')
                pass

        # Set target net to be the same as policy net
        self.replace_target_net(0)

        # Set optimizer & loss function
        self.optim = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

    # Updates the target net to have same weights as policy net
    def replace_target_net(self, steps):
        if steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            print('Target network replaced')
            
    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    def state_to_tensor(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state_t

    def make_action(self, observation, test=False):
        ###########################
        if random.random() > self.epsilon or test:
            observation_t = self.state_to_tensor(observation)
            with torch.no_grad():
                q_values = self.online_net(observation_t)
                max_q_index = torch.argmax(q_values, dim=1)
            action = max_q_index.item()
        else:
            action = self.env.get_random_action()
        ###########################
        return action

    def optimize_model(self):
        if self.buffer_replay.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer_replay.sample(self.batch_size)
        with torch.no_grad():
            q_next_target = self.target_net(next_states)
            q_next_online = self.online_net(next_states)
            online_max_action = torch.argmax(q_next_online, dim=1, keepdim=True)
            q_max = q_next_target.gather(1, online_max_action.long())
            # Using q_max and reward, calculate q_target
            # (1-done) ensures q_target is reward if transition is in a terminating state
            q_target = rewards + self.gamma * q_max * (1 - dones)
        q_pred = self.online_net(states).gather(1, actions.long())
        # Compute the loss
        loss = self.loss(q_pred, q_target)
        # Perform backward propagation and optimization step
        self.optim.zero_grad()
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
        ###########################
        mean_score = 0
        max_score = 0
        i_episode = 0
        while i_episode <= self.total_episodes:
            # Initialize the environment and state
            current_state, _ = self.env.reset()
            done = False
            truncated = False
            episode_score = 0
            loss = 0
            while not (done or truncated):
                # Select and perform an action
                action = self.make_action(current_state, False)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.buffer_replay.push(current_state, action, reward, next_state, int(done or truncated))
                current_state = next_state
                self.env.render()
                if self.buffer_replay.size() > self.start_learning:
                    # Decay epsilon
                    self.dec_eps()
                    # Update target network
                    self.replace_target_net(self.steps)
                    if self.steps % self.train_frequency == 0: # Need to check if its required
                        loss = self.optimize_model()
                else:
                    i_episode = 0
                    continue
                
                # Add the reward to the previous score
                episode_score += reward

            if self.buffer_replay.size() > self.start_learning:
                print('Episode: {:5d},\tScore: {:3.4f},\tAvg.Score: {:3.4f},\tEpislon: {:1.3f},\tLoss: {:8.4f},\tMax.Score: {:3.4f}'
                .format(i_episode, episode_score, mean_score, self.epsilon, loss, max_score))
                
                csvData = [ datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
                            i_episode,
                            episode_score,
                            self.epsilon,
                            loss,
                            max_score,
                            mean_score
                            ]
                with open(self.csv_filename, mode='a') as csvFile:
                    csvWriter = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csvWriter.writerow(csvData)
            else:
                print('Gathering Data: {0}/{1}'.format(self.buffer_replay.size(),self.start_learning))

            if (i_episode > 1) and (i_episode % self.model_save_frequency == 0):
                # Save model
                self.current_moment = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                torch.save( self.online_net.state_dict(), os.path.join('.', self.directory_path, f'{self.model_name}-model-{self.current_moment}.pth'))
                print('-'*100)
                print("Model Saved :", os.path.join(self.directory_path, f'{self.model_name}-model-{self.current_moment}.pth'))
                print('-'*100)

            i_episode += 1
            max_score = episode_score if episode_score > max_score else max_score
            self.scores.append(episode_score)
            self.rewards.append(episode_score)
            mean_score = np.mean(self.scores)

        print('='*50, 'Complete', '='*50)
        torch.save( self.online_net.state_dict(), os.path.join('.', self.directory_path, f'{self.model_name}-model-{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pth'))
        print("Final Model Saved :", os.path.join(self.directory_path, f'{self.model_name}-model-{self.current_moment}.pth'))
        ###########################
