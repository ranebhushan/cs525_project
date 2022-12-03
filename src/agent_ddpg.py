#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import namedtuple, deque
import os
import sys
import math

import torch
# import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from model.ddpg import ActorNetwork, CriticNetwork
from helper.noise import OUActionNoise
from helper.buffer import ReplayBuffer
import csv
import gc
from datetime import datetime

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DDPG(Agent):
    def __init__(self,env,args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DDPG,self).__init__(env)
        # print(self.env.config())
        gc.enable()
        state , _ = self.env.reset()
        state=state.reshape(state.shape[0]*state.shape[1],)
        # print(state)
        print(self.env.action_space)
        print(self.env.action_space.low)
        print(self.env.action_space.high)
        # print(self.env.action_space.shape)
        self.n_actions = self.env.action_space.shape[0]
        self.epsilon_start = args['epsilon_start']
        self.epsilon_end = args['epsilon_end']
        self.epsilon_decay = args['epsilon_decay']
        self.epsilon = self.epsilon_start

        self.gamma = args['gamma']
        self.batch_size = args['batch_size']
        self.buffer_size = args['buffer_size']
        self.learning_rate_actor = args['learning_rate_actor']
        self.learning_rate_critic = args['learning_rate_critic']
        self.num_frames = state.shape[0]
        self.steps = 0
        self.target_update_frequency = args['target_update_frequency']
        self.start_learning = args['start_learning']
        self.model_save_frequency = args['model_save_frequency']
        self.load_model = args['load_model']
        self.clip = args['clip_gradients']
        self.reward_save_frequency = args['reward_save_frequency']
        self.train_frequency = args['train_frequency']
        self.model_name = args['model_name']
        self.input_dims = state.shape
        self.tau = args['tau']
        self.fc1_dims = args['fc1_dims']
        self.fc2_dims = args['fc2_dims']
        self.current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.directory_path = os.path.join("weights", f"{self.model_name}-{self.current_time}")
        self.csv_filename = os.path.join('logs', f'{self.model_name}-{self.current_time}.csv')
        self.scores = deque(maxlen=100)
        self.rewards = deque(maxlen=self.reward_save_frequency)
        # print("INPUT_DIMS:",self.input_dims)
        if(os.path.exists(self.directory_path) == False):
            os.mkdir(self.directory_path)

        with open(self.csv_filename, mode='w', newline="") as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(["Date Time", "Episode", "Reward", "Epsilon", "Loss", "Max. Reward", "Mean Reward"])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = ReplayBuffer(self.buffer_size,self.input_dims,self.n_actions)
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))
        self.actor = ActorNetwork(self.learning_rate_actor, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions, name = "actor",filename='actor_ddpg')
        self.critic = CriticNetwork(self.learning_rate_critic, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions, name = "critic",filename='critic_ddpg')
        self.target_actor = ActorNetwork(self.learning_rate_actor, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions, name = "target_actor",filename='target_actor_ddpg')
        self.target_critic = CriticNetwork(self.learning_rate_critic, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions, name = "target_critic",filename='target_critic_ddpg')
        self.actor = self.actor.to(device=self.device)
        self.critic = self.critic.to(device=self.device)
        self.target_actor = self.target_actor.to(device=self.device)
        self.target_critic = self.target_critic.to(device=self.device)
        # self.actor.load_checkpoint()
        # self.critic.load_checkpoint()
        # self.target_actor.load_checkpoint()
        # self.target_critic.load_checkpoint()
        self.update_network_params()
    
    def make_action(self,obs):
        self.actor.eval()
        state = torch.as_tensor([obs],dtype=torch.float, device= self.device)
        # print("STATE:",state.shape)
        mu = self.actor.forward(state).to(self.device)
        mu_prime = mu + torch.as_tensor(self.noise(),dtype = torch.float, device = self.device)
        # print(mu_prime)
        self.actor.train()
        return mu_prime.to(self.device).detach().numpy()[0]
    
    def remember(self, state, action, reward, next_state , done):
        self.memory.store_transition(state,action, reward, next_state, done)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def learn(self):
        if self.memory.counter < self.batch_size:
            return
        
        states, actions, rewards, next_states, done = self.memory.sample_buffer(self.batch_size)
        states = torch.as_tensor(states, dtype = torch.float, device = self.device)
        next_states = torch.as_tensor(next_states, dtype= torch.float, device= self.device)
        actions = torch.as_tensor(actions, dtype= torch.float, device= self.device)
        rewards = torch.as_tensor(rewards, dtype= torch.float, device= self.device)
        done = torch.as_tensor(done, device= self.device)

        target_actions = self.target_actor.forward(next_states)
        critic_value_ = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma * critic_value_
        target = target.view(self.batch_size,1)

        self.critic.optimizer.zero_grad()
        critic_loss = torch.nn.functional.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_params()

    def update_network_params(self, tau=None):
        if tau is None:
            tau= self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)* target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def train(self):
        mean_score = 0
        max_score = 0
        i_episode = 0

        while mean_score < 2000:
            # Initialize the environment and state
            current_state= self.env.reset()
            current_state=current_state[0].reshape(current_state[0].shape[0]*current_state[0].shape[1],)
            # print(current_state.shape)
            done = False
            truncated = False
            episode_score = 0
            loss = 0
            while not (done or truncated):
                # Select and perform an action
                action = self.make_action(current_state)
                # print(action)
                # throttle = np.clip(action[0],-1,1)
                # steering = np.clip(action[1],-0.05,0.05)
                action = np.clip(action,-1,1)
                # action = torch.tensor([throttle,steering]).numpy()
                # print(type(action))
                # action = np.clip(action[0])
                # action[0] = throttle
                # action[1] = steering
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = next_state.reshape(next_state.shape[0]*next_state.shape[1],)
                self.remember(current_state, action, reward, next_state, done)
                self.learn()
                # self.push(current_state, action, reward, next_state, int(done))
                current_state = next_state
                self.env.render()
                # Add the reward to the previous score
                episode_score += reward
                # Update target network
                # self.replace_target_net(self.steps)

            if self.memory.counter > self.start_learning:
                # print('Episode: ', i_episode, ' Score:', episode_score, ' Avg Score:',round(mean_score,4),' Epsilon: ', round(self.epsilon,4), ' Loss:', round(loss,4), ' Max Score:', max_score)
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
                print('Gathering Data . . .')

            if (i_episode > 1) and (i_episode % self.model_save_frequency == 0):
                # Save model
                # self.online_net.save_model()
                self.current_moment = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                self.save_models()
                # torch.save( self.online_net.state_dict(), os.path.join('.', self.directory_path, f'{self.model_name}-model-{self.current_moment}.pth'))
                print('-'*100)
                print("Model Saved :", os.path.join(self.directory_path, f'{self.model_name}-model-{self.current_moment}.pth'))
                print('-'*100)

            i_episode += 1
            max_score = episode_score if episode_score > max_score else max_score
            self.scores.append(episode_score)
            self.rewards.append(episode_score)
            mean_score = np.mean(self.scores)

        print('='*50, 'Complete', '='*50)
        # self.online_net.save_model()
        self.save_models()
        # torch.save( self.online_net.state_dict(), os.path.join('.', self.directory_path, f'{self.model_name}-model-{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pth'))
        # print("Final Model Saved :", os.path.join(self.directory_path, f'{self.model_name}-model-{self.current_moment}.pth'))
        ###########################