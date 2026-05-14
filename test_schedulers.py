"""
Scheduler Comparison Script (Time Quantum Enabled)
Tests: FIFO, Round Robin, CFS, MLQ, MFQ, PPO, DPO
Environment: 9 features, WITH time quantum, NO aging
"""

import os
import csv
import sys
import numpy as np

from schedulers.round_robin import RoundRobin
from schedulers.fifo import FIFO
from schedulers.cfs import CFS
from schedulers.ml_prio import MLPriority
from schedulers.mlq import MLQ
from schedulers.mfq import MFQ
from schedulers.dpo_prio import DPOPriority
# from schedulers.dqn_prio import DQNPriority  # Uncomment when DQN is ready

# -----------------------------
# CONFIGURATION
# -----------------------------
CSV = "./dataset/train/dataset1.csv"

ENCODER_CONTEXT = 30
MAX_PRIORITY = 10
TIME_QUANTUM = 4  # Time quantum for preemptive scheduling

# Model paths (update these to your actual model paths)
MODEL_PATHS = {
    'PPO': 'model_weights/ppo_trained_model.pt',
    'DPO': 'model_weights/dpo_trained_model.pt',
    'DQN': 'model_weights/dqn_trained_model.pt'
}

# -----------------------------
# TEST FUNCTION
# -----------------------------
def test_scheduler(name, scheduler_class, data, **kwargs):
    """Test a single scheduler and return results"""
    print("=" * 50)
    print(f"Testing: {name}")
    print("=" * 50)
    
    try:
        sched = scheduler_class(data, **kwargs)
        sched.time_run()
        sched.calc_stats()
        sched.print_stats()
        print()
        return sched
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Skipping {name}\n")
        return None

# -----------------------------
# LOAD DATA
# -----------------------------
print("=" * 60)
print("SCHEDULER COMPARISON - INITIAL ENVIRONMENT")
print("=" * 60)

data = np.genfromtxt(CSV, delimiter=',', skip_header=1)
if data.ndim == 1:
    data = data.reshape(1, -1)

print(f"\nDataset: {CSV}")
print(f"Processes: {data.shape[0]}")
print(f"Environment: 9 features (with time quantum={TIME_QUANTUM}, no aging)")
print()

# -----------------------------
# RUN ALL SCHEDULERS
# -----------------------------
results = {}

# Classical schedulers (no additional params needed)
results['FIFO'] = test_scheduler("FIFO", FIFO, data)
results['RoundRobin'] = test_scheduler("Round Robin", RoundRobin, data)
results['CFS'] = test_scheduler("CFS", CFS, data)
results['MLQ'] = test_scheduler("MLQ", MLQ, data)
results['MFQ'] = test_scheduler("MFQ", MFQ, data)

# ML-based schedulers (need encoder_context and max_priority)
results['PPO'] = test_scheduler(
    "ML Priority (PPO)", MLPriority, data,
    encoder_context=ENCODER_CONTEXT,
    max_priority=MAX_PRIORITY,
    time_quantum=TIME_QUANTUM,
    model_path=MODEL_PATHS['PPO']
)

results['DPO'] = test_scheduler(
    "DPO Priority", DPOPriority, data,
    encoder_context=ENCODER_CONTEXT,
    max_priority=MAX_PRIORITY,
    time_quantum=TIME_QUANTUM,
    model_path=MODEL_PATHS['DPO']
)

# Uncomment when DQN is ready
# results['DQN'] = test_scheduler(
#     "DQN Priority", DQNPriority, data,
#     encoder_context=ENCODER_CONTEXT,
#     max_priority=MAX_PRIORITY,
#     model_path=MODEL_PATHS['DQN']
# )

# Remove None results (failed schedulers)
results = {k: v for k, v in results.items() if v is not None}

# -----------------------------
# FINAL COMPARISON TABLE
# -----------------------------
print("\n" + "=" * 85)
print("FINAL COMPARISON TABLE")
print("=" * 85)
print(f"{'Scheduler':<18} {'CPU%':>8} {'Thruput':>10} {'Turnaround':>12} {'Waiting':>10} {'Response':>10} {'Runtime(s)':>10}")
print("-" * 85)

for name, s in results.items():
    print(f"{name:<18} "
          f"{s.stat_cpu_util*100:>8.1f} "
          f"{s.stat_throughput:>10.4f} "
          f"{s.stat_turnaround_time:>12.2f} "
          f"{s.stat_waiting_time:>10.2f} "
          f"{s.stat_response_time:>10.2f} "
          f"{s.stat_runtime:>10.4f}")
print("=" * 85)

# -----------------------------
# SAVE TO CSV
# -----------------------------
os.makedirs('results', exist_ok=True)
out_csv = 'results/scheduler_comparison_initial_env.csv'

with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Scheduler', 'CPU_Util', 'Throughput', 'Turnaround',
                     'Waiting', 'Response', 'Runtime'])
    for name, s in results.items():
        writer.writerow([
            name,
            round(s.stat_cpu_util, 4),
            round(s.stat_throughput, 4),
            round(s.stat_turnaround_time, 2),
            round(s.stat_waiting_time, 2),
            round(s.stat_response_time, 2),
            round(s.stat_runtime, 6)
        ])

print(f"\n✅ Results saved to {out_csv}")

# -----------------------------
# PRINT BEST PERFORMERS
# -----------------------------
print("\n" + "=" * 60)
print("BEST PERFORMERS BY METRIC")
print("=" * 60)

# Find best for each metric
if results:
    best_turnaround = min(results.items(), key=lambda x: x[1].stat_turnaround_time)
    best_waiting = min(results.items(), key=lambda x: x[1].stat_waiting_time)
    best_response = min(results.items(), key=lambda x: x[1].stat_response_time)
    best_throughput = max(results.items(), key=lambda x: x[1].stat_throughput)
    best_cpu = max(results.items(), key=lambda x: x[1].stat_cpu_util)

    print(f"🏆 Best Turnaround:  {best_turnaround[0]} ({best_turnaround[1].stat_turnaround_time:.2f})")
    print(f"🏆 Best Waiting:     {best_waiting[0]} ({best_waiting[1].stat_waiting_time:.2f})")
    print(f"🏆 Best Response:    {best_response[0]} ({best_response[1].stat_response_time:.2f})")
    print(f"🏆 Best Throughput:  {best_throughput[0]} ({best_throughput[1].stat_throughput:.4f})")
    print(f"🏆 Best CPU Util:    {best_cpu[0]} ({best_cpu[1].stat_cpu_util*100:.1f}%)")

print("\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)