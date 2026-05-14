import numpy as np
import torch
import time, datetime

import gymnasium as gym
import gym_env

from ppo import PPO

dataset1 = np.genfromtxt("./dataset/train/dataset1.csv", delimiter=',', skip_header=1)
dataset2 = np.genfromtxt("./dataset/train/dataset2.csv", delimiter=',', skip_header=1)
dataset3 = np.genfromtxt("./dataset/train/dataset3.csv", delimiter=',', skip_header=1)
dataset4 = np.genfromtxt("./dataset/train/dataset4.csv", delimiter=',', skip_header=1)
dataset5 = np.genfromtxt("./dataset/train/dataset5.csv", delimiter=',', skip_header=1)

env = gym.make("gym_env:gym_env/PriorityScheduler-v0", data=dataset1, encoder_context=30, max_priority=10, time_quantum=4)
model = PPO(env, 64)

n_steps = 100000

print('Training PPO model with', n_steps, 'per dataset')
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

start_time = time.time()
env.reset(options={'new_data': dataset5})
model.learn(n_steps)
print('Training on fifth dataset complete after', time.time() - start_time, 'seconds')

print(model.actor)
torch.save(model.actor.state_dict(), 'model_weights/ppo_trained_model.pt')

