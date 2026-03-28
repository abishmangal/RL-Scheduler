# CPU Scheduler — Update Changelog
**Date:** 2026-03-28

---

## 1. `priority_scheduler_env.py` — RL Environment

### Time Quantum / Preemption (New Feature)
- Added `time_quantum=4` parameter to `__init__`
- Observation space shape changed from `(encoder_context+1, 5)` to `(encoder_context+1, 6)` — added `quantum_remaining` as 6th feature
- Heap tuple expanded from 5-tuple to 6-tuple: `(priority, pid, arrival, instr, remaining, quantum_remaining)`
- `_execute_for_time()` now preempts a process when its quantum hits 0 and pushes it back with a fresh quantum
- Three exit cases in execution loop: process completes, quantum exhausted (preempt), time ran out mid-quantum

### `_get_obs()`
- Updated unpack to match new 6-tuple heap
- Row 0 (arriving process) now shows `time_quantum` in column 5
- Queue rows now include `quantum_rem` in column 5

### `reset()`
- Heap tuple comment updated to reflect new 6-tuple shape

### `step()`
- `heappush` now includes `self.time_quantum` as 6th element on process arrival

### `render()`
- Now shows `Quantum left` per process

---

## 2. `ppo.py` — PPO Training

### Action Space Fix (Critical)
- Replaced `MultivariateNormal` with `Categorical` distribution — env uses `spaces.Discrete`, not continuous actions
- Removed `cov_var` and `cov_mat` entirely
- `get_action()` now uses `Categorical(logits=logits)` and returns a scalar integer via `.item()`
- `evaluate()` now uses `Categorical.log_prob()` for discrete action log probabilities

### Batch Actions dtype Fix
- Changed `batch_acts` from `dtype=torch.float` to `dtype=torch.long` — required by `Categorical.log_prob()`

### Observation dtype Fix
- Added `.astype(np.float32)` when flattening obs in rollout

### Hyperparameters (Tuned)
| Parameter | Old | New |
|---|---|---|
| `timesteps_per_batch` | 100,000 | 4,096 |
| `max_timesteps_per_episode` | 10,000 | 500 |
| `gamma` | 0.95 | 0.99 |
| `n_updates_per_iteration` | 5 | 10 |
| `clip` | 0.2 | 0.15 |
| `lr` | 0.005 | 3e-4 |

### Added `save()` and `load()` methods
- `save()` stores both actor and critic state dicts
- `load()` restores both from checkpoint

### Training Logging
- Added per-iteration print: timesteps, actor loss, critic loss, avg episode length

---

## 3. `schedulers/ml_prio.py` — ML Scheduler Wrapper

### Obs Shape Fix (Critical)
- Input dim changed from `(encoder_context+1) * 5` to `(encoder_context+1) * 6`
- `_get_observation()` now builds a 6-column obs matrix matching the trained model

### Heap Fix
- Replaced `PriorityQueue` with `heapq` — removes mutex overhead and heap invariant bugs
- Heap tuple: `(priority, pid, arrival, instr, remaining, quantum_remaining)`

### Time Quantum / Preemption Added
- `run()` now preempts processes when quantum exhausted
- Processes pushed back with fresh `self.time_quantum` on preemption

### Import Fix
- Removed `from ppo import PPO` — only `FeedForwardNN` needed for inference
- Added `sys.path.append` to resolve `priority_prediction/` directory
- Changed to `from network import FeedForwardNN`

### Model Path Fix
- Now uses `kwargs.get('model_path', ...)` instead of hardcoded path
- Absolute path resolution using `os.path` so script works from any directory

### Inference Fix
- Added `torch.no_grad()` during inference
- Added `model.eval()` after loading weights
- Renamed `get_priority` → `_get_priority`, `get_observation` → `_get_observation`

---

## 4. `schedulers/scheduler.py` — Base Scheduler

### `waiting_time()` — Two Bugs Fixed
**Bug 1 — Wrong variable:**
```python
# Old (wrong — used arrivals for both)
instruction_count = self.arrivals[pid]

# Fixed
instr = self.instr_count[pid]
```

**Bug 2 — Formula broken for preemptive schedulers:**
```python
# Old (assumes continuous execution — wrong with time slices)
finish_time - arrival_time - instruction_count

# Fixed (time in system minus actual execution time)
(finish - arrival) - instr
```

---

## 5. `testing_codes/compare_schedulers.py` — New File

- Added `sys.path.append` with `os.path.abspath` for portable imports
- All file paths made absolute using `os.path` so script runs from any directory
- Tests FIFO, Round Robin, CFS, and ML PPO on the same dataset
- Prints individual stats per scheduler
- Prints side-by-side comparison summary table

---

## Summary of What Was NOT Changed

| File | Reason |
|---|---|
| `schedulers/fifo.py` | FIFO has no time slicing by design |
| `schedulers/round_robin.py` | Already implements time slicing correctly |
| `schedulers/cfs.py` | Already implements dynamic time slicing with vruntime |
| `network.py` | Input dim flows in automatically from PPO — no change needed |
| Dataset format | `[pid, arrival, instructions]` unchanged |