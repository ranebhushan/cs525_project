import csv
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(1, os.getcwd() + '/logs')

file_name = 'DDDQN-2022-11-30--02-48-34.csv'

def read_csv(filename):
    logs_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/logs/'
    data = list()
    with open(logs_path + filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            data.append([row])
    return data

def plot_loss(data):
    pass

def plot_mean_reward():
    pass

def plot_reward():
    pass

def plot_epsilon():
    pass

def main():
    data = read_csv(file_name)
    from IPython import embed; embed()
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