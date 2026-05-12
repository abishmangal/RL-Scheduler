import numpy as np
import torch
import time
import datetime
import os

import gymnasium as gym
import gym_env

from dqn import DQN

# ---------------- LOAD DATASET 3 ----------------
dataset3 = np.genfromtxt("./dataset/dataset3.csv", delimiter=',', skip_header=1)

if dataset3.ndim == 1:
    dataset3 = dataset3.reshape(1, -1)

# ---------------- ENV ----------------
env = gym.make(
    "gym_env:gym_env/PriorityScheduler-v0",
    data=dataset3,
    encoder_context=30,
    max_priority=10,
    time_quantum=4,
)

# ---------------- AGENT ----------------
agent = DQN(
    env,
    lr              = 1e-3,
    gamma           = 0.99,
    epsilon_start   = 1.0,
    epsilon_end     = 0.05,
    epsilon_decay   = 0.995,
    batch_size      = 64,
    buffer_capacity = 50_000,
    target_update   = 500,
)

# ---------------- TRAINING PARAMETERS ----------------
n_steps = 2_000_000

print('=' * 50)
print('Training DQN model on Dataset 3')
print(f'Total steps:    {n_steps:,}')
print(f'LR:             {agent.optimizer.param_groups[0]["lr"]}')
print(f'Gamma:          {agent.gamma}')
print(f'Epsilon start:  {agent.epsilon}')
print(f'Epsilon end:    {agent.epsilon_end}')
print(f'Batch size:     {agent.batch_size}')
print(f'Target update:  {agent.target_update}')
print(f'Start time:     {datetime.datetime.now()}')
print('=' * 50)

# ---------------- TRAIN ----------------
start_time = time.time()
losses = agent.learn(n_steps)
training_time = time.time() - start_time

print('=' * 50)
print('Training Complete!')
print(f'Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)')
print(f'End time: {datetime.datetime.now()}')
if losses:
    print(f'Final avg loss (last 100): {np.mean(losses[-100:]):.4f}')
print('=' * 50)

# ---------------- SAVE ----------------
os.makedirs("model_weights", exist_ok=True)
model_path = "model_weights/dqn_scheduler_dataset3_30context.pt"
agent.save(model_path)
print(f'Model saved to: {model_path}')

# ---------------- MODEL ARCHITECTURE ----------------
print('\nModel Architecture:')
print(agent.q_net)

# ---------------- SANITY CHECK ----------------
print('\nRunning sanity check...')
obs, _ = env.reset()
flat = obs.ravel().astype(np.float32)

done = False
actions = []

while not done:
    a = agent.get_action(flat, greedy=True)
    actions.append(a)
    obs, _, done, _, _ = env.step(a)
    flat = obs.ravel().astype(np.float32)

print(f'Actions taken: {len(actions)}')
print(f'Action distribution: {dict(sorted({a: actions.count(a) for a in set(actions)}.items()))}')
print('\nDone')