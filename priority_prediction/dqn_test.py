"""
Quick Test for Trained DQN - Single Episode (FIXED)
"""

import numpy as np
import torch
import gymnasium as gym
import gym_env
from dqn import DQN

# Configuration
DATASET_PATH = "./dataset/dataset1.csv"
MODEL_PATH = "model_weights/dqn_scheduler_dataset1.pt"  # Change this

# Load dataset
dataset = np.genfromtxt(DATASET_PATH, delimiter=',', skip_header=1)
print(f"Loaded {len(dataset)} processes")

# Create environment
env = gym.make("gym_env:gym_env/PriorityScheduler-v0",
               data=dataset,
               encoder_context=30,
               max_priority=10,
               time_quantum=4)

# Access the unwrapped environment to get internal attributes
unwrapped_env = env.unwrapped

# Load trained model
print(f"\nLoading model from {MODEL_PATH}")
dqn = DQN(env)  # Create DQN instance
dqn.load(MODEL_PATH)  # Load weights

# Run one evaluation episode
print("\nRunning evaluation episode...")
eval_results = dqn.evaluate(n_episodes=1)
print(f"Results: {eval_results}")

# Run full episode with details
print("\nRunning detailed episode...")
obs, _ = env.reset()
done = False
total_reward = 0
step = 0

while not done:
    flat_obs = obs.ravel().astype(np.float32)
    action = dqn.get_action(flat_obs, greedy=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    step += 1
    
    if step % 500 == 0:
        # Use unwrapped environment to access completed_processes
        completed = len(unwrapped_env.completed_processes) if hasattr(unwrapped_env, 'completed_processes') else 0
        print(f"  Step {step}: Reward={total_reward:.2f}, "
              f"Completed={completed}/{len(dataset)}")

print(f"\n{'='*50}")
print(f"Episode Complete!")
print(f"{'='*50}")
print(f"Total Steps:        {step}")
print(f"Total Reward:       {total_reward:,.2f}")

# Get final stats from unwrapped environment
if hasattr(unwrapped_env, 'completed_processes'):
    completed = len(unwrapped_env.completed_processes)
    print(f"Completed:          {completed}/{len(dataset)}")
    
    if completed > 0:
        turnarounds = [t for _, t in unwrapped_env.completed_processes]
        print(f"Avg Turnaround:     {np.mean(turnarounds):.2f}")
        print(f"Min Turnaround:     {np.min(turnarounds)}")
        print(f"Max Turnaround:     {np.max(turnarounds)}")