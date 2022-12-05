import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(1, os.getcwd() + '/logs')


# File format: [Date Time,Episode,Reward,Epsilon,Loss,Max. Reward,Mean Reward]

file_name = 'DDDQN-2022-12-03--18-44-20.csv'

def read_csv(filename):
    logs_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/logs/'
    df = pd.read_csv(logs_path + filename, sep=',', header=0)
    return df

def plot_loss(data):
    plt.plot(data['Episode'].to_numpy(), data['Loss'].to_numpy())
    plt.title("Episode v/s Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()

def plot_mean_reward(data):
    plt.plot(data['Episode'].to_numpy(), data['Mean Reward'].to_numpy()) 
    plt.title("Episode v/s Mean Reward")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.show() 

def plot_reward(data):
    plt.plot(data['Episode'].to_numpy(), data['Reward'].to_numpy()) 
    plt.title("Episode v/s Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")    
    plt.show() 

def plot_epsilon(data):
    plt.plot(data['Episode'].to_numpy(), data['Epsilon'].to_numpy()) 
    plt.title("Episode v/s Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")    
    plt.show() 

def main():
    data = read_csv(file_name)
    plot_mean_reward(data)
    plot_reward(data)
    plot_loss(data)
    plot_epsilon(data)

if __name__ == '__main__':
    main()

# y = np.array([])



# x = np.arange(start=30, stop=30*(y.shape[0]+1), step=30)

# plt.plot(x, y)
# plt.show()