"""
Simple DQN Training Script
Matches the simplicity of the PPO training script
"""

import numpy as np
import torch
import time, datetime

import gymnasium as gym
import gym_env

from dqn import DQN

# Load datasets
dataset1 = np.genfromtxt("./dataset/dataset1.csv", delimiter=',', skip_header=1)
dataset2 = np.genfromtxt("./dataset/dataset2.csv", delimiter=',', skip_header=1)
dataset3 = np.genfromtxt("./dataset/dataset3.csv", delimiter=',', skip_header=1)
dataset4 = np.genfromtxt("./dataset/dataset4.csv", delimiter=',', skip_header=1)
dataset5 = np.genfromtxt("./dataset/dataset5.csv", delimiter=',', skip_header=1)

# Create environment with first dataset
env = gym.make("gym_env:gym_env/PriorityScheduler-v0", 
               data=dataset1, 
               encoder_context=30, 
               max_priority=10)

# Create DQN model
model = DQN(env, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, 
            epsilon_decay=0.9995, batch_size=64, buffer_capacity=100000, target_update=1000)

n_steps = 5000000  # 5 million steps per dataset

print('Training DQN model with', n_steps, 'per dataset')
start_time = time.time()

print('Starting time:', datetime.datetime.now())
model.learn(n_steps)
print('Training on first dataset complete after', time.time() - start_time, 'seconds')

start_time = time.time()
env.reset(options={'new_data': dataset2})
model.learn(n_steps)
print('Training on second dataset complete after', time.time() - start_time, 'seconds')

start_time = time.time()
env.reset(options={'new_data': dataset3})
model.learn(n_steps)
print('Training on third dataset complete after', time.time() - start_time, 'seconds')

start_time = time.time()
env.reset(options={'new_data': dataset4})
model.learn(n_steps)
print('Training on fourth dataset complete after', time.time() - start_time, 'seconds')

#start_time = time.time()
#env.reset(options={'new_data': dataset5})
#model.learn(n_steps)
#print('Training on fifth dataset complete after', time.time() - start_time, 'seconds')

print(model.q_net)
torch.save(model.q_net.state_dict(), 'model_weights/dqn_scheduler_5mil_30context.pt')