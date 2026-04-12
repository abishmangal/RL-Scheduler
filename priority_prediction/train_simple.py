import numpy as np
import torch
import time
import datetime

import gymnasium as gym
import gym_env

from ppo import PPO

# Load dataset 3
dataset3 = np.genfromtxt("./dataset/dataset3.csv", delimiter=',', skip_header=1)

# Create environment with dataset 3
env = gym.make("gym_env:gym_env/PriorityScheduler-v0", 
               data=dataset3, 
               encoder_context=30, 
               max_priority=10, time_quantum=4)

# Initialize PPO model
model = PPO(env, 64)

# Training parameters
n_steps = 5000000

print('=' * 50)
print('Training PPO model on Dataset 3')
print(f'Total steps: {n_steps:,}')
print(f'Start time: {datetime.datetime.now()}')
print('=' * 50)

# Train the model
start_time = time.time()
model.learn(n_steps)
training_time = time.time() - start_time

print('=' * 50)
print('Training Complete!')
print(f'Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)')
print(f'End time: {datetime.datetime.now()}')
print('=' * 50)

# Save the model
model_path = 'model_weights/ml_priority_scheduler_dataset3_5mil_30context.pt'
torch.save(model.actor.state_dict(), model_path)
print(f'Model saved to: {model_path}')

# Optional: Print model architecture
print('\nModel Architecture:')
print(model.actor)