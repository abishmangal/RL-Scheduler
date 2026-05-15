"""
Simple DPO Training Script
Matches the simplicity of the PPO training script
"""

import numpy as np
import torch
import time, datetime

import gymnasium as gym
import gym_env

from dpo import DPO, collect_preferences

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

# Create DPO model (optionally load PPO as reference)
model = DPO(env, ref_actor_path=None, beta=0.1, lr=1e-4, batch_size=64)

n_pairs = 2000  # Number of preference pairs per dataset
n_epochs = 10   # DPO training epochs

print('Training DPO model with', n_pairs, 'preference pairs per dataset')
start_time = time.time()

print('Starting time:', datetime.datetime.now())

# Dataset 1
print('\n--- Training on first dataset ---')
prefs = collect_preferences(env, n_pairs=n_pairs, horizon=15, n_candidates=4)
model.train(prefs, n_epochs=n_epochs)
print('Training on first dataset complete after', time.time() - start_time, 'seconds')

# Dataset 2
start_time = time.time()
env.reset(options={'new_data': dataset2})
print('\n--- Training on second dataset ---')
prefs = collect_preferences(env, n_pairs=n_pairs, horizon=15, n_candidates=4)
model.train(prefs, n_epochs=n_epochs)
print('Training on second dataset complete after', time.time() - start_time, 'seconds')

# Dataset 3
start_time = time.time()
env.reset(options={'new_data': dataset3})
print('\n--- Training on third dataset ---')
prefs = collect_preferences(env, n_pairs=n_pairs, horizon=15, n_candidates=4)
model.train(prefs, n_epochs=n_epochs)
print('Training on third dataset complete after', time.time() - start_time, 'seconds')

# Dataset 4
start_time = time.time()
env.reset(options={'new_data': dataset4})
print('\n--- Training on fourth dataset ---')
prefs = collect_preferences(env, n_pairs=n_pairs, horizon=15, n_candidates=4)
model.train(prefs, n_epochs=n_epochs)
print('Training on fourth dataset complete after', time.time() - start_time, 'seconds')

#Dataset 5
#start_time = time.time()
#env.reset(options={'new_data': dataset5})
#print('\n--- Training on fifth dataset ---')
#prefs = collect_preferences(env, n_pairs=n_pairs, horizon=15, n_candidates=4)
#model.train(prefs, n_epochs=n_epochs)
#print('Training on fifth dataset complete after', time.time() - start_time, 'seconds')

print(model.policy)
torch.save(model.policy.state_dict(), 'model_weights/dpo_scheduler_5mil_30context.pt')