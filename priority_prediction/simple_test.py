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

# Track scheduling decisions
priority_assignments = {}   # pid -> assigned priority
scheduling_log = []         # list of (step, pid, action, queue_state)

print("\n" + "=" * 70)
print("SCHEDULING TRACE")
print("=" * 70)
print(f"{'Step':>6} {'Action':>8} {'PID Assigned':>14} {'Queue Size':>12} {'Reward':>10}")
print("-" * 70)

while not done:
    steps += 1

    obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32)
    with torch.no_grad():
        action = torch.argmax(model.actor(obs_flat)).item()

    # Extract next arriving process from obs (row 0)
    arriving_pid = int(obs[0, 0]) if obs[0, 0] != -1 else None
    queue_size   = sum(1 for i in range(1, obs.shape[0]) if obs[i, 0] != -1)

    # Record priority assignment
    if arriving_pid is not None and arriving_pid not in priority_assignments:
        priority_assignments[arriving_pid] = action
        print(f"{steps:>6} {action:>8} {arriving_pid:>14} {queue_size:>12}", end="")

    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    if arriving_pid is not None and arriving_pid not in [log[0] for log in scheduling_log]:
        print(f" {reward:>10.2f}")
        scheduling_log.append((arriving_pid, action, reward))

    if steps > 20000:
        print("Breaking loop (safety)")
        break

# ── Summary ──────────────────────────────────────────────────────────────────
completed = env.unwrapped.completed_processes

print("\n" + "=" * 70)
print("PRIORITY ASSIGNMENTS")
print("=" * 70)
print(f"{'PID':>6} {'Priority':>10} {'Arrival':>10} {'Instructions':>14}")
print("-" * 70)
for pid in sorted(priority_assignments.keys()):
    arrival = int(dataset[pid, 1])
    instr   = int(dataset[pid, 2])
    prio    = priority_assignments[pid]
    print(f"{pid:>6} {prio:>10} {arrival:>10} {instr:>14}")

print("\n" + "=" * 70)
print("COMPLETION STATS")
print("=" * 70)
print(f"{'PID':>6} {'Priority':>10} {'Turnaround':>12} {'Instructions':>14}")
print("-" * 70)

turnarounds = []
for pid, turnaround in sorted(completed, key=lambda x: x[0]):
    prio  = priority_assignments.get(pid, '?')
    instr = int(dataset[pid, 2])
    turnarounds.append(turnaround)
    print(f"{pid:>6} {str(prio):>10} {turnaround:>12} {instr:>14}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total steps:        {steps}")
print(f"Total reward:       {total_reward:.2f}")
print(f"Processes completed:{len(completed)}/{len(dataset)}")
if turnarounds:
    print(f"Avg turnaround:     {np.mean(turnarounds):.2f}")
    print(f"Min turnaround:     {np.min(turnarounds):.2f}  (PID {completed[np.argmin(turnarounds)][0]})")
    print(f"Max turnaround:     {np.max(turnarounds):.2f}  (PID {completed[np.argmax(turnarounds)][0]})")
print("=" * 70)