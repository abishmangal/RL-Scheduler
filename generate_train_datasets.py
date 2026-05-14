#!/usr/bin/env python3
"""
Training Dataset Generator for CPU Scheduling RL Training
Generates diverse, realistic datasets suitable for training PPO/DQN/DPO agents.
Each dataset contains a mix of all edge cases so the agent learns robust policies.
"""

import numpy as np
import os


def generate_train_dataset(n_processes=10000, max_arrival=30000, seed=42):
    """
    Generate a diverse training dataset with realistic patterns.

    Features:
    - Mixed instruction lengths (no single-value patterns)
    - Poisson-distributed arrivals with varying rates (bursts + idle)
    - Early long jobs (FIFO blocking test)
    - Late urgent jobs (response time test)
    - Simultaneous burst arrivals (queue pressure test)
    - Sequential PIDs for correct statistics
    """
    rng = np.random.RandomState(seed)

    # =============================================================
    # INSTRUCTION COUNTS: diverse distribution
    # =============================================================
    instr_bins = [
        (1, 5, 0.25),     # 25% short interactive
        (6, 15, 0.30),    # 30% medium
        (16, 40, 0.25),   # 25% long
        (41, 100, 0.15),  # 15% very long (starvation candidates)
        (101, 200, 0.05), # 5% extremely long
    ]

    instructions = []
    for low, high, prob in instr_bins:
        count = int(n_processes * prob)
        instr = rng.randint(low, high + 1, count)
        instructions.extend(instr)

    while len(instructions) < n_processes:
        instructions.append(rng.randint(1, 20))
    while len(instructions) > n_processes:
        instructions.pop()

    instructions = np.array(instructions, dtype=np.int32)

    # =============================================================
    # ARRIVAL TIMES: Poisson process with bursts
    # =============================================================
    arrivals = []

    # 1. Early long job for FIFO blocking test (arrives at time 5)
    arrivals.append(5)

    # 2. Late urgent jobs for response time test
    for t in [max_arrival - 20, max_arrival - 15, max_arrival - 10, max_arrival - 5]:
        arrivals.append(t)

    # 3. Burst arrivals (multiple jobs at same time, queue pressure)
    burst_times = [
        (int(max_arrival * 0.1), 8),   # 8 jobs at 10% mark
        (int(max_arrival * 0.3), 12),  # 12 jobs at 30% mark
        (int(max_arrival * 0.6), 6),   # 6 jobs at 60% mark
        (int(max_arrival * 0.8), 15),  # 15 jobs at 80% mark
    ]
    for t, count in burst_times:
        for _ in range(count):
            arrivals.append(t)

    # 4. Poisson arrivals for remaining processes
    remaining = n_processes - len(arrivals)
    # Vary arrival rate: clusters of activity with idle periods
    lam = remaining / max_arrival * 2.5  # base rate
    interarrival_times = rng.exponential(1.0 / lam, remaining * 2)
    poisson_arrivals = np.cumsum(interarrival_times).astype(np.int32)
    poisson_arrivals = poisson_arrivals[poisson_arrivals < max_arrival]
    poisson_arrivals = poisson_arrivals[:remaining]
    while len(poisson_arrivals) < remaining:
        extra = rng.exponential(1.0 / lam, remaining - len(poisson_arrivals))
        poisson_arrivals = np.append(poisson_arrivals, poisson_arrivals[-1] + np.cumsum(extra).astype(np.int32))
        poisson_arrivals = poisson_arrivals[poisson_arrivals < max_arrival]
    poisson_arrivals = poisson_arrivals[:remaining]
    arrivals.extend(poisson_arrivals.tolist())

    arrivals = np.array(arrivals, dtype=np.int32)

    # Sort by arrival time
    sorted_idx = np.argsort(arrivals)
    instructions = instructions[sorted_idx]
    arrivals = arrivals[sorted_idx]

    # Make the earliest-arriving job very long (FIFO blocking test)
    early_idx = np.where(arrivals <= 10)[0]
    if len(early_idx) > 0:
        instructions[early_idx[0]] = max(80, int(instructions.max()))

    # Sequential PIDs (required for correct scheduler statistics)
    pids = np.arange(len(arrivals))

    data = np.column_stack([pids.astype(np.int32), arrivals, instructions])
    return data


def save_dataset(data, filename, directory="./dataset/"):
    os.makedirs(directory, exist_ok=True)
    savepath = os.path.join(directory, filename + '.csv')
    np.savetxt(savepath, data, fmt='%i', delimiter=',',
               header='PID,ArrivalTime,InstructionCount', comments='')
    print(f"  Saved to {savepath}")
    return savepath


def print_statistics(data, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"{name} Statistics")
    print(f"{'='*50}")
    print(f"  Processes: {len(data)}")
    print(f"  Arrivals: [{data[:,1].min()} - {data[:,1].max()}]")
    print(f"  Instructions: min={data[:,2].min()}, max={data[:,2].max()}, avg={data[:,2].mean():.1f}")

    short = np.sum(data[:,2] <= 5)
    medium = np.sum((data[:,2] > 5) & (data[:,2] <= 15))
    long_ = np.sum((data[:,2] > 15) & (data[:,2] <= 40))
    very_long = np.sum(data[:,2] > 40)

    print(f"  Short (1-5):     {short:>5} ({short/len(data)*100:5.1f}%)")
    print(f"  Medium (6-15):   {medium:>5} ({medium/len(data)*100:5.1f}%)")
    print(f"  Long (16-40):    {long_:>5} ({long_/len(data)*100:5.1f}%)")
    print(f"  Very Long (41+): {very_long:>5} ({very_long/len(data)*100:5.1f}%)")

    early_long = data[(data[:,1] < 20) & (data[:,2] > 50)]
    if len(early_long) > 0:
        print(f"  Edge: {len(early_long)} early long job(s) — FIFO blocking test")

    late_urgent = data[(data[:,1] > data[:,1].max() - 50) & (data[:,2] <= 3)]
    if len(late_urgent) > 0:
        print(f"  Edge: {len(late_urgent)} late urgent job(s) — Response time test")

    unique_times, counts = np.unique(data[:,1], return_counts=True)
    bursts = {int(t): int(c) for t, c in zip(unique_times, counts) if c > 5}
    if bursts:
        print(f"  Edge: Burst arrivals at {bursts} — Queue pressure test")
    print(f"  {'='*50}\n")


# ============================================================
# Main: Generate 5 training datasets (replaces datasets 1-5)
# ============================================================

if __name__ == "__main__":
    output_dir = "./dataset/train"
    n_processes = 1000
    max_arrival = 3000

    print("=" * 60)
    print("GENERATING TRAINING DATASETS FOR RL SCHEDULING")
    print("=" * 60)

    # Generate 5 datasets with different seeds for diversity
    for i in range(1, 6):
        seed = 100 + i
        print(f"\nGenerating dataset{i} ({n_processes} processes, seed={seed})...")
        data = generate_train_dataset(
            n_processes=n_processes,
            max_arrival=max_arrival,
            seed=seed,
        )
        save_dataset(data, f"dataset{i}", directory=output_dir)
        print_statistics(data, f"Training Dataset {i}")

    print("=" * 60)
    print("All 5 training datasets generated successfully!")
    print(f"Output directory: {output_dir}/")
    print("=" * 60)
