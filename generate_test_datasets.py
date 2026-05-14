#!/usr/bin/env python3
"""
Dataset Generator for CPU Scheduling Comparison
Generates datasets that expose weaknesses of classical schedulers
All datasets use sequential PIDs (0,1,2,...) for correct statistics calculation
"""

import numpy as np
import os

def generate_challenging_dataset(n_processes=500, max_arrival=500, max_instructions=50, seed=42):
    """
    Generate a dataset that specifically exposes classical scheduler weaknesses.
    All PIDs are sequential (0, 1, 2, ...) for correct statistics.
    
    Key test scenarios:
    1. FIFO blocking: Long job arriving early blocks short jobs
    2. Starvation: Very long jobs that may never run in SJF
    3. Response time: Urgent short jobs arriving late
    4. Burst handling: Multiple jobs arriving simultaneously
    5. Mixed workload: Interactive vs batch jobs
    """
    np.random.seed(seed)
    
    # Create varied instruction counts to test different scenarios
    instruction_probs = [
        (1, 5, 0.35),    # 35% short interactive jobs (1-5 instructions)
        (6, 15, 0.35),   # 35% medium jobs (6-15 instructions)
        (16, 30, 0.20),  # 20% long jobs (16-30 instructions)
        (31, 100, 0.10)  # 10% very long jobs (31-100 instructions)
    ]
    
    instructions = []
    for low, high, prob in instruction_probs:
        count = int(n_processes * prob)
        instr = np.random.randint(low, high + 1, count)
        instructions.extend(instr)
    
    # Adjust if we have extra processes due to rounding
    while len(instructions) < n_processes:
        instructions.append(np.random.randint(1, 10))
    while len(instructions) > n_processes:
        instructions.pop()
    
    instructions = np.array(instructions)
    
    # Create arrivals with patterns that expose scheduler weaknesses
    arrivals = []
    
    # 1. FIFO blocking test: Long job arrives at time 5
    long_job_arrival = 5
    
    # 2. Response time test: Urgent jobs arriving late
    urgent_arrivals = [480, 485, 490, 495]
    
    # 3. Burst arrival test: Multiple jobs at same time
    burst_times = [100, 200, 300]
    
    # 4. Normal distribution for remaining
    remaining_count = n_processes - 1 - len(urgent_arrivals) - (len(burst_times) * 5)
    normal_arrivals = np.random.uniform(0, max_arrival, remaining_count)
    normal_arrivals = np.clip(normal_arrivals, 1, max_arrival - 10)
    
    # Build arrival list
    arrivals.append(long_job_arrival)
    arrivals.extend(normal_arrivals.tolist())
    arrivals.extend(urgent_arrivals)
    for burst in burst_times:
        for _ in range(5):  # 5 jobs at each burst time
            arrivals.append(burst)
    
    arrivals = np.array(arrivals)
    
    # Sort by arrival time
    sorted_indices = np.argsort(arrivals)
    instructions = instructions[sorted_indices]
    arrivals = arrivals[sorted_indices]
    
    # Make sure the long job (first in sorted order) has long instructions
    # Find the job that arrived at time 5 (or earliest)
    early_indices = np.where(arrivals <= 10)[0]
    if len(early_indices) > 0:
        instructions[early_indices[0]] = 80  # Very long job for FIFO test
    
    # Add sequential PIDs (0, 1, 2, ...)
    pids = np.arange(n_processes)
    
    # Create final dataset
    data = np.column_stack([pids, arrivals.astype(np.int32), instructions.astype(np.int32)])
    
    return data


def generate_test_dataset(size=500, scenario="balanced", seed=42):
    """
    Generate specialized datasets for different test scenarios.
    All PIDs are sequential (0, 1, 2, ...) for correct statistics.
    
    scenarios:
    - "balanced": Mixed workload
    - "fifo_test": Long job early to test FIFO blocking
    - "starvation_test": Very long jobs to test starvation prevention
    - "response_test": Late urgent jobs to test response time
    - "burst_test": Many simultaneous arrivals
    """
    np.random.seed(seed)
    
    if scenario == "fifo_test":
        # Many short jobs after a long job - tests FIFO blocking
        instructions = np.ones(size, dtype=np.int32) * 2  # Most are short
        instructions[0] = 100  # First process is very long
        arrivals = np.random.uniform(0, 500, size)
        arrivals[0] = 10  # Long job arrives early
        arrivals = arrivals.astype(np.int32)
        
    elif scenario == "starvation_test":
        # Mix with very long jobs that may starve in SJF
        instructions = np.where(np.random.random(size) < 0.2, 
                                np.random.randint(50, 100, size),  # 20% very long
                                np.random.randint(1, 10, size))    # 80% short
        instructions = instructions.astype(np.int32)
        arrivals = np.random.uniform(0, 500, size).astype(np.int32)
        
    elif scenario == "response_test":
        # Urgent jobs arriving late - tests response time
        instructions = np.ones(size, dtype=np.int32) * 5  # All medium
        urgent_indices = np.random.choice(size, size//10, replace=False)
        instructions[urgent_indices] = 1  # 10% are urgent (1 instruction)
        arrivals = np.random.uniform(0, 480, size)
        # Make some urgent jobs arrive later
        for idx in urgent_indices[:5]:
            arrivals[idx] = np.random.uniform(480, 500)
        arrivals = arrivals.astype(np.int32)
        
    elif scenario == "burst_test":
        # Many jobs arriving simultaneously - tests queue management
        instructions = np.random.randint(1, 20, size).astype(np.int32)
        # Create bursts
        arrivals = np.zeros(size, dtype=np.int32)
        for i in range(0, size, 20):
            arrivals[i:i+20] = i * 10 + np.random.randint(-5, 5, 20)
        arrivals = np.clip(arrivals, 0, 500).astype(np.int32)
        
    else:  # "balanced" - default
        # Mixed: 40% short, 35% medium, 25% long
        instructions = []
        for _ in range(int(size * 0.4)):
            instructions.append(np.random.randint(1, 5))
        for _ in range(int(size * 0.35)):
            instructions.append(np.random.randint(6, 15))
        for _ in range(size - len(instructions)):
            instructions.append(np.random.randint(16, 50))
        instructions = np.array(instructions, dtype=np.int32)
        arrivals = np.random.uniform(0, 500, size).astype(np.int32)
    
    # Sort by arrival time
    sorted_idx = np.argsort(arrivals)
    instructions = instructions[sorted_idx]
    arrivals = arrivals[sorted_idx]
    
    # Add sequential PIDs (0, 1, 2, ...)
    pids = np.arange(size)
    
    # Create final dataset
    data = np.column_stack([pids, arrivals, instructions])
    
    return data


def save_dataset(data, filename, directory="./dataset/"):
    """Save dataset in the required format"""
    directory = os.path.join(directory, "test")
    os.makedirs(directory, exist_ok=True)
    savepath = os.path.join(directory, filename + '.csv')
    np.savetxt(savepath, data, fmt='%i', delimiter=',', 
               header='PID,ArrivalTime,InstructionCount', comments='')
    print(f"Saved to {savepath}")
    return savepath


def print_statistics(data, name="Dataset"):
    """Print dataset statistics"""
    print(f"\n{'='*50}")
    print(f"{name} Statistics")
    print(f"{'='*50}")
    print(f"Total processes: {len(data)}")
    print(f"PID range: {data[:,0].min()} - {data[:,0].max()} (sequential)")
    print(f"Arrival time range: {data[:,1].min()} - {data[:,1].max()}")
    print(f"Instructions: min={data[:,2].min()}, max={data[:,2].max()}, avg={data[:,2].mean():.1f}")
    
    # Instruction distribution
    short = np.sum(data[:,2] <= 5)
    medium = np.sum((data[:,2] > 5) & (data[:,2] <= 15))
    long = np.sum((data[:,2] > 15) & (data[:,2] <= 30))
    very_long = np.sum(data[:,2] > 30)
    
    print(f"Short jobs (1-5 instr): {short} ({short/len(data)*100:.1f}%)")
    print(f"Medium jobs (6-15 instr): {medium} ({medium/len(data)*100:.1f}%)")
    print(f"Long jobs (16-30 instr): {long} ({long/len(data)*100:.1f}%)")
    print(f"Very long jobs (31+ instr): {very_long} ({very_long/len(data)*100:.1f}%)")
    
    # Check for specific test scenarios
    early_long = data[(data[:,1] < 20) & (data[:,2] > 50)]
    if len(early_long) > 0:
        print(f"⚠️  Early long jobs (arrival<20, instr>50): {len(early_long)} - FIFO starvation risk")
    
    late_urgent = data[(data[:,1] > 400) & (data[:,2] <= 3)]
    if len(late_urgent) > 0:
        print(f"⚠️  Late urgent jobs (arrival>400, instr<=3): {len(late_urgent)} - Response time test")
    
    # Check burst arrivals
    unique_times, counts = np.unique(data[:,1], return_counts=True)
    bursts = {t: c for t, c in zip(unique_times, counts) if c > 5}
    if bursts:
        print(f"⚠️  Burst arrivals: {bursts} - Queue management test")


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    
    # Generate main challenging dataset (500 processes)
    print("Generating Challenging Dataset (500 processes)...")
    dataset = generate_challenging_dataset(
        n_processes=500,
        max_arrival=500,
        max_instructions=50,
        seed=42
    )
    
    # Save
    save_dataset(dataset, "dataset_challenging_500")
    print_statistics(dataset, "Challenging Dataset (500)")
    
    # Generate additional test datasets
    print("\n" + "="*60)
    print("Generating Test Datasets for Specific Scenarios")
    print("="*60)
    
    # 1. FIFO blocking test
    fifo_test = generate_test_dataset(200, scenario="fifo_test", seed=42)
    save_dataset(fifo_test, "dataset_fifo_test")
    print_statistics(fifo_test, "FIFO Blocking Test")
    
    # 2. Starvation test
    starvation_test = generate_test_dataset(200, scenario="starvation_test", seed=42)
    save_dataset(starvation_test, "dataset_starvation_test")
    print_statistics(starvation_test, "Starvation Test")
    
    # 3. Response time test
    response_test = generate_test_dataset(200, scenario="response_test", seed=42)
    save_dataset(response_test, "dataset_response_test")
    print_statistics(response_test, "Response Time Test")
    
    # 4. Burst test
    burst_test = generate_test_dataset(200, scenario="burst_test", seed=42)
    save_dataset(burst_test, "dataset_burst_test")
    print_statistics(burst_test, "Burst Arrival Test")
    
    # 5. Small balanced dataset for quick testing (50 processes)
    small_dataset = generate_challenging_dataset(50, max_arrival=200, max_instructions=30, seed=42)
    save_dataset(small_dataset, "dataset_small_50")
    
    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("="*60)
    
    # Display first 20 lines of the main dataset
    print("\n=== First 20 lines of challenging dataset ===")
    print("PID,ArrivalTime,InstructionCount")
    for i in range(min(20, len(dataset))):
        print(f"{dataset[i,0]},{dataset[i,1]},{dataset[i,2]}")
    
    # Verify PIDs are sequential
    print(f"\n✅ PID verification: 0 to {dataset[-1,0]} (sequential)")