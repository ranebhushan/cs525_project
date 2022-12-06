import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import sys, os
sys.path.insert(1, os.getcwd() + '/logs')


# File format: [Date Time,Episode,Reward,Epsilon,Loss,Max. Reward,Mean Reward]

file_name = 'DQN-2022-12-04--03-30-14.csv'

def read_csv(filename):
    logs_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/logs/'
    df = pd.read_csv(logs_path + filename, sep=',', header=0)
    return df

def plot_loss(data):
    plt.plot(data['Episode'].to_numpy(), data['Loss'].to_numpy())
    plt.title("Loss vs. No.of Episodes")
    plt.xlabel("No.of Episodes")
    plt.ylabel("Loss")
    plt.savefig('DQN_loss.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_mean_reward(data):
    # scores = deque(maxlen=100)
    # mean_reward = np.array([])
    # for reward in data['Reward'].to_numpy():
    #     scores.append(reward)
    #     mean_reward = np.append(mean_reward, np.mean(scores))
    # plt.plot(data['Episode'].to_numpy(), mean_reward) 
    plt.plot(data['Episode'].to_numpy(), data['Mean Reward'].to_numpy()) 
    plt.title("Mean Reward vs. No.of Episodes")
    plt.xlabel("No.of Episodes")
    plt.ylabel("Mean Reward")
    plt.savefig('DQN_mean_reward.png',dpi=300,bbox_inches='tight')
    plt.show() 

def plot_reward(data):
    plt.plot(data['Episode'].to_numpy(), data['Reward'].to_numpy()) 
    plt.title("Reward vs. No.of Episodes")
    plt.xlabel("No.of Episodes")
    plt.ylabel("Reward")   
    plt.savefig('DQN_reward.png',dpi=300,bbox_inches='tight') 
    plt.show() 

def plot_epsilon(data):
    plt.plot(data['Episode'].to_numpy(), data['Epsilon'].to_numpy()) 
    plt.title("Epsilon vs. No.of Episodes")
    plt.xlabel("No.of Episodes")
    plt.ylabel("Epsilon")  
    plt.savefig('DQN_epsilon.png',dpi=300,bbox_inches='tight')  
    plt.show() 

def main():
    data = read_csv(file_name)
    plot_mean_reward(data)
    plot_reward(data)
    plot_loss(data)
    plot_epsilon(data)

if __name__ == '__main__':
    main()
