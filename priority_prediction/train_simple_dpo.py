import numpy as np
import torch
import time
import datetime
import os

import gymnasium as gym
import gym_env

from dpo import DPO, collect_preferences

# ---------------- LOAD DATASET 3 ----------------
dataset3 = np.genfromtxt("./dataset/dataset3.csv", delimiter=',', skip_header=1)

if dataset3.ndim == 1:
    dataset3 = dataset3.reshape(1, -1)

print(f"Dataset shape: {dataset3.shape} ({int(dataset3.shape[0])} processes)")

# ---------------- ENV ----------------
env = gym.make(
    "gym_env:gym_env/PriorityScheduler-v0",
    data=dataset3,
    encoder_context=30,
    max_priority=10,
    time_quantum=4,
)

# ---------------- TRAINING PARAMETERS ----------------
N_PAIRS      = 1000
HORIZON      = 15
N_CANDIDATES = 4
BETA         = 0.1
EPOCHS       = 5
BATCH_SIZE   = 64
LR           = 1e-4
PPO_WEIGHTS  = "model_weights/ml_priority_scheduler_dataset3_5mil_30context.pt"

print('=' * 50)
print('Training DPO model on Dataset 3')
print(f'Pairs:        {N_PAIRS:,}')
print(f'Horizon:      {HORIZON}')
print(f'Candidates:   {N_CANDIDATES}')
print(f'Beta:         {BETA}')
print(f'Epochs:       {EPOCHS}')
print(f'Batch size:   {BATCH_SIZE}')
print(f'LR:           {LR}')
print(f'Ref policy:   {PPO_WEIGHTS}')
print(f'Start time:   {datetime.datetime.now()}')
print('=' * 50)

# ---------------- COLLECT PREFERENCES ----------------
print('\nCollecting preference pairs...')
pref_start = time.time()

pref_dataset = collect_preferences(
    env,
    n_pairs=N_PAIRS,
    horizon=HORIZON,
    n_candidates=N_CANDIDATES,
    seed=42,
)

print(f'Preference collection took {time.time() - pref_start:.2f}s')
print(f'Collected {len(pref_dataset)} preference pairs')

# ---------------- DPO TRAIN ----------------
trainer = DPO(
    env=env,
    ref_actor_path=PPO_WEIGHTS,
    beta=BETA,
    lr=LR,
    batch_size=BATCH_SIZE,
)

print('\nStarting DPO training...')
start_time = time.time()
trainer.train(pref_dataset, n_epochs=EPOCHS)
training_time = time.time() - start_time

print('=' * 50)
print('Training Complete!')
print(f'Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)')
print(f'End time: {datetime.datetime.now()}')
print('=' * 50)

# ---------------- SAVE ----------------
os.makedirs("model_weights", exist_ok=True)
model_path = "model_weights/dpo_scheduler_dataset3_30context.pt"
trainer.save(model_path)
print(f'Model saved to: {model_path}')

# ---------------- MODEL ARCHITECTURE ----------------
print('\nModel Architecture:')
print(trainer.policy)

# ---------------- SANITY CHECK ----------------
print('\nRunning sanity check...')
obs, _ = env.reset()
flat = np.ravel(obs).astype(np.float32)

done = False
actions = []

while not done:
    a = trainer.get_action(flat)
    actions.append(a)
    obs, _, done, _, _ = env.step(a)
    flat = np.ravel(obs).astype(np.float32)

print(f'Actions taken: {len(actions)}')
print(f'Action distribution: {dict(sorted({a: actions.count(a) for a in set(actions)}.items()))}')
print('\nDone')