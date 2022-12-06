import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
import gym
import numpy as np
import os
import random
from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from gym.spaces import box, Discrete, discrete

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def remember(self, state, action, logprob, reward, is_terminal):
        state = torch.from_numpy(state).float().to(device)
        action = torch.tensor(action, dtype=torch.float16).to(device)
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprob)
        # if reward==0:
        #     reward = 50
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, checkpoint):
#         super(Actor, self).__init__()

#         self.action_dim = action_dim
#         self.state_dim = state_dim
#         self.checkpoint = checkpoint+'/actor_ddpg'
        
#         self.fc1 = nn.Linear(state_dim, 32)
#         self.fc2 = nn.Linear(32,64)
#         self.fc3 = nn.Linear(64,32)
#         self.mu = nn.Linear(32,action_dim)

#         self.bn1 = nn.LayerNorm(32)
#         self.bn2 = nn.LayerNorm(64)
#         self.bn3 = nn.LayerNorm(32)

#         fc1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])
#         fc2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])
#         fc3 = 1.0/np.sqrt(self.fc3.weight.data.size()[0])
#         mu = 0.03

#         torch.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc3.weight.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.fc3.bias.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.mu.weight.data, -mu, mu)
#         torch.nn.init.uniform_(self.mu.bias.data, -mu, mu)


#     def forward(self, x, act=False):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = F.softmax(self.mu(x))

#         return x.to(device)
    
#     def save(self):
#         torch.save(self.state_dict(), self.checkpoint)

#     def load(self):
#         self.load_state_dict(torch.load(self.checkpoint))


# class Critic(nn.Module):
#     def __init__(self, state_dim, checkpoint):
#         super(Critic, self).__init__()

#         self.state_dim = state_dim
#         self.checkpoint = checkpoint+'/critic_ddpg'

#         self.fc1 = nn.Linear(state_dim, 64)
#         self.bn1 = nn.LayerNorm(64)
#         self.fc2 = nn.Linear(64,128)
#         self.bn2 = nn.LayerNorm(128)
#         self.fc3 = nn.Linear(128,32)
#         self.bn3 = nn.LayerNorm(32)
#         self.pi = nn.Linear(32,1)

#         fc1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])
#         fc2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])
#         fc3 = 1.0/np.sqrt(self.fc3.weight.data.size()[0])
#         pi = 0.03

#         torch.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc3.weight.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.fc3.bias.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.pi.weight.data, -pi, pi)
#         torch.nn.init.uniform_(self.pi.bias.data, -pi, pi)


#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = self.pi(x)

#         return x.to(device)

#     def save(self):
#         torch.save(self.state_dict(), self.checkpoint)

#     def load(self):
#         self.load_state_dict(torch.load(self.checkpoint))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, checkpoint='tmp/ppo'):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, 1),
                nn.Tanh()
                )
        self.checkpoint = checkpoint
        self.dist = None

        
    def forward(self):
        raise NotImplementedError

        
    def act(self, state):
        # state = torch.from_numpy(state).float().to(device)
        # action_probs = self.old_actor.forward(state)
        # dist = Categorical(action_probs)
        # action = dist.sample()
        # print(state)

        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device) 
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            self.dist = dist
            
            # memory.states.append(state)
            # memory.actions.append(action)
            # memory.logprobs.append(dist.log_prob(action))
            return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs = self.action_layer.forward(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer.forward(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self):
        print('saving into {}'.format(self.checkpoint))
        if not os.path.isdir(self.checkpoint):
            os.makedirs(self.checkpoint)
        torch.save(self.action_layer.state_dict(), self.checkpoint+'/action.pt')
        torch.save(self.value_layer.state_dict(), self.checkpoint+'/value.pt')

    def load(self):
        print('loading from {}'.format(self.checkpoint))
        if not os.path.isdir(self.checkpoint):
            return
        self.action_layer.load_state_dict(torch.load(self.checkpoint+'/action.pt'))
        self.value_layer.load_state_dict(torch.load(self.checkpoint+'/value.pt'))

    def transfer(self):
        self.old_actor.load_state_dict(self.action_layer.state_dict())


class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, checkpoint='tmp/ppo'):
        super(ActorCriticContinuous, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim),
                nn.Tanh()
                )
        self.std = nn.Parameter(torch.zeros(1,action_dim)).to(device)
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, 1),
                nn.Tanh()
                )
        self.checkpoint = checkpoint
        self.dist = None
        self.action_var = torch.full((action_dim,), 0.1*0.1).to(device)

        
    def forward(self):
        raise NotImplementedError

        
    def act(self, state):
        # state = torch.from_numpy(state).float().to(device)
        # action_probs = self.old_actor.forward(state)
        # dist = Categorical(action_probs)
        # action = dist.sample()
        # print(state)

        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device) 
            action_probs = self.action_layer(state)
            cov_mat = torch.diag(self.action_var).to(device)
            # print(cov_mat)
            # print(action_probs)
            dist = MultivariateNormal(action_probs, cov_mat)
            action = dist.sample()
            # print(action, action_probs)
            log_prob = dist.log_prob(action)
            self.dist = dist
            return action.detach().cpu().numpy(), log_prob
        
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        action_var = self.action_var.expand_as(action_probs)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_probs, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self):
        print('saving into {}'.format(self.checkpoint))
        if not os.path.isdir(self.checkpoint):
            os.makedirs(self.checkpoint)
        torch.save(self.action_layer.state_dict(), self.checkpoint+'/action.pt')
        torch.save(self.value_layer.state_dict(), self.checkpoint+'/value.pt')

    def load(self):
        print('loading from {}'.format(self.checkpoint))
        if not os.path.isdir(self.checkpoint):
            return
        self.action_layer.load_state_dict(torch.load(self.checkpoint+'/action.pt'))
        self.value_layer.load_state_dict(torch.load(self.checkpoint+'/value.pt'))

    def transfer(self):
        self.old_actor.load_state_dict(self.action_layer.state_dict())
        
class PPO:
    def __init__(self, env, 
    n_latent_var, 
    lr, 
    betas, 
    gamma, 
    K_epochs, 
    eps_clip, 
    checkpoint='../models'):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.env = env
        self.memory = Memory()
        self.checkpoint = checkpoint
        # self.policy = ActorCritic(env.reset().reshape(-1).shape[0], env.action_space.n, n_latent_var, checkpoint+'/policy').to(device)
        self.MseLoss = nn.MSELoss()
        self.state_dim,_ = self.env.reset()
        self.state_dim = self.state_dim.flatten()
        self.state_dim = self.state_dim.shape[0]
        # self.policy_old = ActorCritic(env.reset().reshape(-1).shape[0], env.action_space.n, n_latent_var, checkpoint+'/old_policy').to(device)
        # self.policy_old.load_state_dict(self.policy.state_dict())
        if isinstance(env.action_space, Discrete):
            self.policy = ActorCritic(self.state_dim, env.action_space.n, n_latent_var, checkpoint+'/policy').to(device)
            self.policy_old = ActorCritic(self.state_dim, env.action_space.n, n_latent_var, checkpoint+'/old_policy').to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy = ActorCriticContinuous(self.state_dim, env.action_space.shape[0], n_latent_var, checkpoint+'/policy').to(device)
            self.policy_old = ActorCriticContinuous(self.state_dim, env.action_space.shape[0], n_latent_var, checkpoint+'/old_policy').to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

    def select_action(self, state):
        return self.policy_old.act(state)

    def remember(self, state, action, log_prob, rewards, is_terminal):
        self.memory.remember(state, action, log_prob, rewards, is_terminal)
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        # print('update')
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards:
        # print(rewards)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # print(rewards)
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        self.policy.action_layer.train()
        self.policy.value_layer.train()
        self.optimizer.zero_grad()
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # print(logprobs-old_logprobs)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # print(ratios)
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            # print(advantages)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.mean(torch.min(surr1, surr2)) + 0.5*torch.mean(self.MseLoss(state_values, rewards)) - 0.001*torch.mean(dist_entropy)
            
            # take gradient step
            # print(loss)
            loss.backward()
            self.optimizer.step()

        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, scores):
        print('saving to checkpoint................................')
        self.policy.save()
        self.policy_old.save()
        np.savetxt(os.path.abspath(self.checkpoint)+'/data.csv', scores, delimiter=',')

    def load(self):
        print('loading data........................................')
        self.policy.load()
        self.policy_old.load()

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_change = 0 if action==2 or action==0 else 1
        neighbours = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
        lane = self.env.vehicle.target_lane_index[2] if isinstance(self.env.vehicle, ControlledVehicle) \
            else self.env.vehicle.lane_index[2]
        # if self.env.vehicle.crashed:
        #     reward = self.env.config["collision_reward"]*100
        # else:
        #     reward = 10
                # + self.env.RIGHT_LANE_REWARD * lane*10 / max(len(neighbours) - 1, 1) \
        scaled_speed = utils.lmap(self.env.vehicle.speed, self.env.config["reward_speed_range"], [0, 1])
        # print(scaled_speed, self.env.config["reward_speed_range"])
        # print(scaled_speed)
        # print(self.env.RIGHT_LANE_REWARD*lane)
        # print(self.env.config["collision_reward"])
        crashed = -3 if self.env.vehicle.crashed else 1
        reward = crashed*1
            # + 1 * scaled_speed \
            # + crashed*1
            # + 0.01*lane_change \
            # + 0.24 * lane / max(len(neighbours) - 1, 1) \
            

        # print(reward)
        # reward = self.env.config["collision_reward"] if self.env.vehicle.crashed else reward
        # reward = utils.lmap(reward,
        #                   [-3, 2],
        #                   [-1, 1])
        # print(reward)
        # reward = utils.lmap(reward,
        #                 [self.env.config["collision_reward"]*5, (self.env.HIGH_SPEED_REWARD + self.env.RIGHT_LANE_REWARD)*5],
        #                 [0, 5])
        # print(reward)
        # print(self.env.vehicle.speed, reward)
        reward = 0 if not self.env.vehicle.on_road else reward
        return reward


    def runner(self, i_episode, render=False, train=True):
        # env = gym.make(env_name)
        scores=[]
        # training loop
        # np.random.seed(0)
        # self.load()
        if not train:
            self.load()
        success=0
        # t=0
        total=0
        count_rand=0
        for i in range(1,i_episode+1):
            state = self.env.reset()
            running_reward = 0.0
            done=False
            t=0
            # print(state)
            while not done:
                t+=1
                state = state.reshape(-1)
                action, log_prob = self.select_action(state)
                # val=random.random()
                # if val<0.03:
                #     action = self.env.action_space.sample()
                #     log_prob = self.policy_old.dist.log_prob(torch.tensor(action))
                #     print('random')
                new_state, reward, done, info = self.env.step(action)
                # reward_new = self._reward(action)
                self.remember(state, action, log_prob, reward, done)
                state = new_state
                running_reward += reward
                # self.update()
                if render:
                    self.env.render()
                # if val<0.3 and done and t<100:
                #     count_rand+=1
                #     print('crashed because of random')
            total+=t
            # print(reward_new)
            # self.update()
            if i%2==0 and train==True:
                self.update()
                # self.memory.clear_memory()
            if i%10==0:
                self.memory.clear_memory()
            scores.append(running_reward)
            print('episode ', i, 'score %.2f' % running_reward,
                'trailing 50 games avg %.3f' % np.mean(scores[-50:]), 'finished after ', t, ' steps', 'average steps: ', total/i,
                'crash because of random: ', count_rand)
            if i%25==0 and train==True:
                self.save(scores)
        # scores.append(running_reward)
        # np.savetxt(self.checkpoint+'/data.csv', scores, delimiter=',')
        if train:
            self.save(scores)
        self.env.close()
        return scores