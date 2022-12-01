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

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done'))

class Agent_DDDQN(Agent):
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

        self.buffer_replay = deque(maxlen=self.buffer_size)
        self.scores = deque(maxlen=100)
        self.rewards = deque(maxlen=self.reward_save_frequency)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialise policy and target networks, set target network to eval mode
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
        self.optim = optim.RMSprop(self.online_net.parameters(), lr=self.learning_rate)
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
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return state_t

    def make_action(self, observation, test=False):
        ###########################
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
        states, actions, rewards, next_states, dones = self.replay_buffer()

        q_pred = self.online_net.forward(states).gather(dim=1, index=actions).squeeze()
        q_next = self.target_net.forward(next_states)
        q_eval = self.online_net.forward(next_states)
        with torch.no_grad():
            max_actions = torch.max(q_eval, dim=1)[1].unsqueeze(1)
            q_max = q_next.gather(dim=1, index=max_actions).squeeze()
            # Using q_max and reward, calculate q_target
            # (1-done) ensures q_target is reward if transition is in a terminating state
            q_target = rewards + self.gamma * q_max * (1 - dones)

        # q_values_online = self.online_net(state)
        # q_val_next = self.online_net(next_state).detach()
        # q_val_next_target = self.target_net(next_state).detach()
        # with torch.no_grad():
        #     q_actions = torch.max(q_val_next, dim=1)[1].unsqueeze(1)
        #     q_max = q_val_next_target.gather(dim=1, index=q_actions).squeeze()
        #     q_target = reward + self.gamma * q_max * (1 - done)
        # q_val = q_values_online.gather(dim=1, index=action).squeeze()

        # Compute the loss
        loss = self.loss(q_pred, q_target).to(self.device)
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
        ###########################
        mean_score = 0
        max_score = 0
        i_episode = 0

        while i_episode <= self.total_episodes:
            # Initialize the environment and state
            current_state, _ = self.env.reset()
            done = False
            episode_score = 0
            loss = 0
            while not done or truncated:
                # Select and perform an action
                action = self.make_action(current_state, False)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.push(current_state, action, reward, next_state, int(done or truncated))
                current_state = next_state
                # self.env.render()
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

            if len(self.buffer_replay) > self.start_learning:
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
                print('Gathering Data: {0}/{1}'.format(len(self.buffer_replay),self.start_learning))

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
