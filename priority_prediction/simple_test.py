import torch
import numpy as np
import gymnasium as gym
import gym_env

from ppo import PPO

print("Starting test...")

dataset = np.genfromtxt("./dataset/dataset3.csv", delimiter=',', skip_header=1)

env = gym.make("gym_env/PriorityScheduler-v0", data=dataset, encoder_context=30, max_priority=10, time_quantum=4)

model = PPO(env, 64)
model.actor.load_state_dict(torch.load("model_weights/ml_priority_scheduler_dataset3_5mil_30context.pt"))
model.actor.eval()

obs, _ = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    steps += 1

    obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32)
    action = torch.argmax(model.actor(obs_flat)).item()

    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    if steps % 100 == 0:
        print("Step:", steps)

    if steps > 20000:
        print("Breaking loop (safety)")
        break

print("Finished!")
print("Steps:", steps)
print("Total reward:", total_reward)
print("Completed:", len(env.unwrapped.completed_processes))