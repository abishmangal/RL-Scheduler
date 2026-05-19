"""
ml_prio.py — ML-Based Priority Scheduler (PPO Inference)
=========================================================

This scheduler uses a trained PPO actor network to assign priorities to
processes at submission time.  It replicates the observation-building logic
from the training environment so the model sees the same input format.

At each process arrival:
  1. Build the 9-feature observation matrix from the current queue state.
  2. Run the PPO actor (feed-forward network) to get priority logits.
  3. Pick argmax (highest logit) as the priority.
  4. Push the process into a min-heap ready queue.

The queue is preemptive with a time quantum: each process runs for at most
`time_quantum` ticks before being re-queued (unless it finishes first).

Dictionary tracking (wait_since, total_wait, last_run_time, first_run_time)
is maintained to populate the dynamic features of the observation —
matching the training environment's _get_obs().
"""

import os
import heapq
import numpy as np
import torch
from .scheduler import Scheduler
from priority_prediction.network import FeedForwardNN


class MLPriority(Scheduler):
    """
    ML-based Priority Scheduler using trained PPO policy.

    Inherits from the abstract Scheduler base class, which provides
    statistics calculation (turnaround, waiting, response time, etc.)
    after run() completes.

    The model is a FeedForwardNN trained via PPO in priority_prediction/.
    """
    
    def __init__(self, data, **kwargs):
        super().__init__(data)
        
        # Store raw data
        self.raw_data = data
        self.pids = data[:, 0].astype(int)
        self.arrivals = data[:, 1]
        self.instr_count = data[:, 2]
        
        # Required kwargs
        try:
            self.encoder_context = kwargs['encoder_context']
            self.max_priority = kwargs['max_priority']
        except KeyError:
            raise ValueError(
                "MLPriority requires 'encoder_context' and 'max_priority' kwargs."
            )
        
        # Optional kwargs
        self.model_path = kwargs.get('model_path', 'model_weights/ppo_trained_model.pt')
        self.time_quantum = kwargs.get('time_quantum', 4)
        
        # Priority history for Gantt chart visualization
        self.gantt_priority = []
        
        # ============================================================
        # DICTIONARY TRACKING FOR DYNAMIC STATE (9-feature observation)
        # ============================================================
        self.wait_since = {}      # When each process started waiting (timestamp)
        self.total_wait = {}      # Accumulated waiting time (ticks)
        self.last_run_time = {}   # When each process last ran (timestamp)
        self.first_run_time = {}  # When each process first ran (for response time)
        
        # Build observation dimension: (encoder_context+1) * 9 features
        obs_dim = (self.encoder_context + 1) * 9
        
        # Load model
        self.model = FeedForwardNN(obs_dim, self.max_priority)
        
        # Handle absolute/relative path
        if not os.path.isabs(self.model_path):
            project_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..'
            )
            self.model_path = os.path.normpath(
                os.path.join(project_root, self.model_path)
            )
        
        # Load weights and set to eval mode
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if 'actor' in checkpoint:
            self.model.load_state_dict(checkpoint['actor'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"[MLPriority] Loaded model from: {self.model_path}")
        print(f"[MLPriority] Observation dim: {obs_dim} (9 features with time_quantum={self.time_quantum})")
        print(f"[MLPriority] Dictionary tracking enabled: wait_time, last_run_time")
    
    def run(self):
        """
        Run the scheduler simulation.

        Main loop:
          1. At each time tick, admit all processes whose arrival time == now.
             For each admitted process, query the PPO model for a priority.
          2. Pop the highest-priority process from the min-heap.
          3. Execute it for 1 tick (or until preemption/completion).
          4. If it still has work remaining, push it back into the heap
             (with a fresh quantum if the current one expired).
          5. If nothing is ready, record an idle tick (-1 in gantt).

        Dictionary tracking is updated at each step to reflect current
        wait times, last-run times, and first-run detection.
        """
        
        # Build process list: [pid, arrival, total_instructions, remaining_instructions]
        processes = []
        for i in range(self.raw_data.shape[0]):
            pid = int(self.raw_data[i, 0])
            arrival = int(self.raw_data[i, 1])
            instr = int(self.raw_data[i, 2])
            processes.append([pid, arrival, instr, instr])
        processes.sort(key=lambda x: x[1])
        
        # Priority queue as min-heap: (priority, pid, arrival, total, remaining, quantum_rem)
        self.execution_queue = []
        data_pointer = 0
        time = 0
        self.gantt = []
        
        self.gantt_priority.clear()
        
        # Clear tracking dictionaries for new run
        self.wait_since.clear()
        self.total_wait.clear()
        self.last_run_time.clear()
        self.first_run_time.clear()
        
        while self.execution_queue or data_pointer < len(processes):
            # Add all processes arriving at current time
            while (data_pointer < len(processes) and 
                   processes[data_pointer][1] == time):
                proc = processes[data_pointer]
                pid = proc[0]
                
                # Get priority from model
                priority = self._get_priority(data_pointer, processes, time)
                
                # Initialize dictionary tracking for new process
                self.wait_since[pid] = time
                self.total_wait[pid] = 0
                self.last_run_time[pid] = 0
                self.first_run_time[pid] = 0
                
                heapq.heappush(
                    self.execution_queue,
                    (priority, pid, proc[1], proc[2], proc[3], self.time_quantum)
                )
                data_pointer += 1
            
            if self.execution_queue:
                # Get highest priority process (lowest number)
                priority, pid, arrival, total, remaining, quantum_rem = heapq.heappop(self.execution_queue)
                
                # Accumulate wait time since this process was last enqueued
                if pid in self.wait_since:
                    wait_accum = time - self.wait_since[pid]
                    self.total_wait[pid] = self.total_wait.get(pid, 0) + wait_accum
                
                self.last_run_time[pid] = time
                
                if self.first_run_time.get(pid, 0) == 0:
                    self.first_run_time[pid] = time
                
                # Run for 1 time unit
                self.gantt.append(pid)
                self.gantt_priority.append(priority)
                remaining -= 1
                quantum_rem -= 1
                time += 1
                
                # If process still has work, start a new wait period
                if remaining > 0:
                    self.wait_since[pid] = time
                
                if remaining > 0:
                    if quantum_rem == 0:
                        # Quantum exhausted — recharge
                        heapq.heappush(
                            self.execution_queue,
                            (priority, pid, arrival, total, remaining, self.time_quantum)
                        )
                    else:
                        heapq.heappush(
                            self.execution_queue,
                            (priority, pid, arrival, total, remaining, quantum_rem)
                        )
                else:
                    # Process completed — clean up tracking dictionaries
                    self.wait_since.pop(pid, None)
                    self.total_wait.pop(pid, None)
                    self.last_run_time.pop(pid, None)
            else:
                # No processes ready — idle tick
                self.gantt.append(-1)
                self.gantt_priority.append(-1)
                time += 1
    
    def _get_priority(self, data_pointer: int, processes: list, current_time: int) -> int:
        """
        Query the PPO model for a priority assignment.

        Builds the 9-feature observation matching the training environment,
        flattens it, and runs it through the actor network.  Returns the
        argmax priority (discrete action).
        """
        obs = self._get_observation(data_pointer, processes, current_time)
        flat = obs.ravel().astype(np.float32)
        tensor = torch.tensor(flat, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(tensor)
            priority = int(torch.argmax(logits).item())
        
        return priority
    
    def _get_observation(self, data_pointer: int, processes: list, current_time: int) -> np.ndarray:
        """
        Build the 9-feature observation that matches the training environment.

        Shape: (encoder_context + 1, 9)
          Row 0       = next arriving process (not yet in queue)
          Rows 1+     = processes currently in the ready queue

        Features per row:
          [0] PID
          [1] Arrival time
          [2] Total instructions
          [3] Remaining instructions
          [4] Current priority (or -1 for the arriving process)
          [5] Quantum remaining
          [6] Queue position (0 = front)
          [7] Total wait time (accumulated)
          [8] Time since last run

        This mirrors gym_env/envs/priority_scheduler.py:_get_obs().
        """
        obs = np.full((self.encoder_context + 1, 9), -1, dtype=np.float32)
        
        # ============================================================
        # ROW 0: NEXT ARRIVING PROCESS (not yet in queue)
        # ============================================================
        if data_pointer < len(processes):
            proc = processes[data_pointer]
            obs[0, 0] = proc[0]                    # PID
            obs[0, 1] = proc[1]                    # Arrival time
            obs[0, 2] = proc[2]                    # Total instructions
            obs[0, 3] = proc[3]                    # Remaining instructions
            obs[0, 4] = -1                         # No priority yet
            obs[0, 5] = self.time_quantum          # Will get full quantum
            obs[0, 6] = len(self.execution_queue)  # Queue position (how many ahead)
            obs[0, 7] = 0                          # Wait time (not in queue)
            obs[0, 8] = 0                          # Time since last run (N/A)
        
        # ============================================================
        # ROWS 1+: PROCESSES IN READY QUEUE
        # ============================================================
        for i, (priority, pid, arrival, total, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            
            # ============================================================
            # CALCULATE TOTAL WAIT TIME (from dictionaries)
            # ============================================================
            total_wait = self.total_wait.get(pid, 0)
            if pid in self.wait_since:
                total_wait += current_time - self.wait_since[pid]
            
            # ============================================================
            # CALCULATE TIME SINCE LAST RUN (from dictionaries)
            # ============================================================
            last_run = self.last_run_time.get(pid, 0)
            time_since_last = current_time - last_run if last_run > 0 else 0
            
            # ============================================================
            # BUILD 9-FEATURE OBSERVATION
            # ============================================================
            obs[i + 1, 0] = pid                    # PID
            obs[i + 1, 1] = arrival                # Arrival time
            obs[i + 1, 2] = total                  # Total instructions
            obs[i + 1, 3] = remaining              # Remaining instructions
            obs[i + 1, 4] = priority               # Current priority
            obs[i + 1, 5] = quantum_rem            # Quantum remaining
            obs[i + 1, 6] = i                      # Queue position (0 = front)
            obs[i + 1, 7] = total_wait             # Total accumulated wait time
            obs[i + 1, 8] = time_since_last        # Time since last execution
        
        return obs
    
    def get_stats(self):
        """Return tracking statistics for debugging"""
        return {
            'wait_since': dict(self.wait_since),
            'total_wait': dict(self.total_wait),
            'last_run_time': dict(self.last_run_time),
            'first_run_time': dict(self.first_run_time)
        }