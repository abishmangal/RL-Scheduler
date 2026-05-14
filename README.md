# CPU Scheduling with Deep Reinforcement Learning

## Introduction

This project extends [Kyle O'Donnell's](https://github.com/kyle-odonnell) PPO-based CPU scheduler (original code [here](https://github.com/kyle-odonnell/PPO-for-CPU-Scheduling)) by adding **multiple RL training algorithms** (PPO, DPO, DQN) and **expanded scheduler comparisons** against classical algorithms. The gym environment has been updated to 9 observation features with time quantum support and response-time-focused reward weighting.

## Environment Setup

Install the Anaconda environment:
``` 
conda env create -f environment.yml
conda activate ml
```

Install the custom Gymnasium environment:
```
pip install -e gym_env
```

Verify registration:
```python
import gymnasium as gym
gym.envs.registry.keys().__contains__('gym_env/PriorityScheduler-v0')
```

## Dataset Generation

### Training Datasets
Generate 5 training datasets (with diverse edge cases: FIFO blocking, starvation, response time, burst arrivals):
```
python generate_train_datasets.py
```
Datasets are saved to `dataset/train/dataset{1-5}.csv` with sequential PIDs and columns: `PID,ArrivalTime,InstructionCount`.

### Test Datasets
Generate specialized test datasets for evaluating scheduler weaknesses:
```
python generate_dataset.py
```
Outputs to `dataset/test/` with scenarios like `fifo_test`, `starvation_test`, `response_test`, `burst_test`, and `challenging_500`.

## Training

### PPO (Proximal Policy Optimization)
```
python priority_prediction/train_ppo_priority_scheduler.py
```

### DQN (Deep Q-Network)
```
python priority_prediction/train_dqn.py
```

### DPO (Direct Preference Optimization)
```
python priority_prediction/train_dpo.py
```

Model weights are saved to `model_weights/` as `.pt` files.

A basic PPO example is at `priority_prediction/run_ppo_example.py`.

## Schedulers

Available schedulers in `schedulers/`:

| Scheduler | File | Type |
|-----------|------|------|
| FIFO | `fifo.py` | Classical |
| Round Robin | `round_robin.py` | Classical (time quantum) |
| CFS | `cfs.py` | Classical (Completely Fair) |
| MLQ | `mlq.py` | Classical (Multilevel Queue) |
| MFQ | `mfq.py` | Classical (Multilevel Feedback Queue) |
| PPO Priority | `ml_prio.py` | RL-trained priority prediction |
| DPO Priority | `dpo_prio.py` | RL-trained (Direct Preference Optimization) |
| DQN Priority | `dqn_prio.py` | RL-trained (Deep Q-Network) |

All RL schedulers use an **encoder context of 30** and **max priority of 10** with the updated 9-feature environment.

## Evaluation

### Quick Demo (with Gantt charts)
```
python demo_scheduling.py
```
Compares all schedulers side-by-side on a chosen dataset with live results and optional Gantt chart visualization.

### Full Test Suite
```
python test_schedulers.py
```
Runs all schedulers on a specified dataset and prints detailed statistics including turnaround time, waiting time, response time, CPU utilization, and throughput.

### Test Datasets (pre-generated)
- `dataset_challenging_500.csv` - Mixed workload with edge cases
- `dataset_fifo_test.csv` - Long job early (FIFO blocking)
- `dataset_starvation_test.csv` - Very long jobs (starvation risk)
- `dataset_response_test.csv` - Late urgent jobs (response time)
- `dataset_burst_test.csv` - Simultaneous burst arrivals

## Tracking Statistics

All schedulers track:
- **CPU Utilization**
- **Throughput**
- **Average Turnaround Time**
- **Average Waiting Time**
- **Average Response Time**
- **Overhead**

## Results

- Tabulated results: `results_report.ipynb` -> `results/test{1-3}_results.csv`
- Figures: `plot_figures.ipynb` -> `figures/test{1-3}/`
- Pre-generated Gantt charts: `gantt_*.png`

## References

1. Eric Yu - [PPO from Scratch](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) ([GitHub](https://github.com/ericyangyu/PPO-for-Beginners))
2. Sanchith Hegde - [Completely Fair Scheduler](https://github.com/SanchithHegde/completely-fair-scheduler)
3. Kyle O'Donnell - [Original PPO-for-CPU-Scheduling](https://github.com/kyle-odonnell/PPO-for-CPU-Scheduling)
4. Rafailov et al. - [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

