import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from schedulers.fifo import FIFO
from schedulers.cfs import CFS
from schedulers.round_robin import RoundRobin
from schedulers.ml_prio import MLPriority

DATASET = "./dataset/dataset5.csv"

def test_scheduler(scheduler_cls, name, **kwargs):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    data = np.genfromtxt(DATASET, delimiter=',', skip_header=1)
    sched = scheduler_cls(data, **kwargs)
    sched.time_run()
    sched.calc_stats()
    sched.print_stats()
    return sched

# Run all schedulers
fifo    = test_scheduler(FIFO,       "FIFO")
rr      = test_scheduler(RoundRobin, "Round Robin")
cfs     = test_scheduler(CFS,        "CFS (Linux)")
ml_prio = test_scheduler(MLPriority, "ML Priority (PPO)",
                         encoder_context=30,
                         max_priority=10,
                         time_quantum=4,
                         model_path="model_weights/ml_priority_scheduler_dataset3_5mil_30context.pt")

# Summary comparison table
print(f"\n{'='*50}")
print("  COMPARISON SUMMARY — Dataset 2")
print(f"{'='*50}")
print(f"{'Metric':<20} {'FIFO':>10} {'RR':>10} {'CFS':>10} {'ML PPO':>10}")
print(f"{'-'*60}")

metrics = [
    ("CPU Util",       "stat_cpu_util"),
    ("Throughput",     "stat_throughput"),
    ("Turnaround",     "stat_turnaround_time"),
    ("Waiting Time",   "stat_waiting_time"),
    ("Response Time",  "stat_response_time"),
    ("Runtime (s)",    "stat_runtime"),
]

schedulers = [fifo, rr, cfs, ml_prio]

for label, attr in metrics:
    values = [getattr(s, attr) for s in schedulers]
    print(f"{label:<20} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f} {values[3]:>10.4f}")

print(f"\n{'='*50}")
print("Lower turnaround/waiting/response = better")
print("Higher CPU util/throughput = better")
print(f"{'='*50}")