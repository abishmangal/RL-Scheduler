"""
dpo_prio.py — DPO-Based Priority Scheduler (Inference)
========================================================

This scheduler uses a trained DPO policy network to assign priorities.

It is structurally identical to ml_prio.py (PPO inference), but loads
weights from a DPO-trained model instead.  The observation-building and
queue-management logic is shared across all ML-based schedulers.

See ml_prio.py for detailed algorithm notes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'priority_prediction'))

from .scheduler import Scheduler
import heapq
import numpy as np
import torch
from priority_prediction.network import FeedForwardNN


class DPOPriority(Scheduler):
    """
    Priority scheduler driven by a DPO-trained policy network.

    At each process arrival, the policy network (FeedForwardNN) maps the
    current 9-feature observation to action logits; argmax gives the
    assigned priority.

    Parameters
    ----------
    data            : np.ndarray — process table [PID, ArrivalTime, Instructions]
    encoder_context : int        — queue slots in observation
    max_priority    : int        — discrete priority levels
    time_quantum    : int        — ticks before preemption (default 4)
    model_path      : str        — path to saved DPO weights (.pt)
    """

    def __init__(self, data, **kwargs):
        super().__init__(data)
        
        self.raw_data = data

        try:
            self.encoder_context = kwargs['encoder_context']
            self.max_priority = kwargs['max_priority']
        except KeyError:
            raise ValueError(
                "DPOPriority requires 'encoder_context' and 'max_priority' kwargs."
            )

        self.time_quantum = kwargs.get('time_quantum', 4)
        
        # Priority history for Gantt chart visualization
        self.gantt_priority = []
        
        # Dictionary tracking mirrors the training environment's internal
        # state, populating the dynamic features of the observation
        # (wait_time, last_run_time, time_since_last_run).
        self.wait_since = {}
        self.total_wait = {}
        self.last_run_time = {}
        self.first_run_time = {}

        obs_dim = (self.encoder_context + 1) * 9
        self.model = FeedForwardNN(obs_dim, self.max_priority)

        model_path = kwargs.get('model_path', 'model_weights/dpo_trained_model.pt')
        if not os.path.isabs(model_path):
            project_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..'
            )
            model_path = os.path.normpath(
                os.path.join(project_root, model_path)
            )

        # Load model with error handling
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'policy' in checkpoint:
            self.model.load_state_dict(checkpoint['policy'])
        elif 'q_net_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['q_net_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"[DPOPriority] Loaded from: {model_path}")
        print(f"[DPOPriority] Observation dim: {obs_dim} (9 features, time_quantum={self.time_quantum})")
        print(f"[DPOPriority] Dictionary tracking enabled: wait_time, last_run_time")
        print(f"[DPOPriority] Processes: {len(self.pids)}")

    def run(self):
        """
        Run the scheduler simulation.

        Simulation loop mirrors ml_prio.py: admit arriving processes (querying
        the DPO model for each priority), execute the highest-priority process
        for 1 tick, preempt/re-queue or complete it, and track wait times
        and run times via dictionaries.
        """
        processes = []
        for i in range(self.raw_data.shape[0]):
            pid = int(self.raw_data[i, 0])
            arrival = int(self.raw_data[i, 1])
            instr = int(self.raw_data[i, 2])
            processes.append([pid, arrival, instr, instr])

        processes.sort(key=lambda x: x[1])
        
        self.execution_queue = []
        data_pointer = 0
        time = 0
        self.gantt = []
        
        self.gantt_priority.clear()
        self.wait_since.clear()
        self.total_wait.clear()
        self.last_run_time.clear()
        self.first_run_time.clear()

        while self.execution_queue or data_pointer < len(processes):
            while (data_pointer < len(processes) and 
                   processes[data_pointer][1] == time):
                proc = processes[data_pointer]
                pid = proc[0]
                
                priority = self._get_priority(data_pointer, processes, time)
                
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
                priority, pid, arrival, total, remaining, quantum_rem = heapq.heappop(self.execution_queue)
                
                if pid in self.wait_since:
                    wait_accum = time - self.wait_since[pid]
                    self.total_wait[pid] = self.total_wait.get(pid, 0) + wait_accum
                
                self.last_run_time[pid] = time
                
                if self.first_run_time.get(pid, 0) == 0:
                    self.first_run_time[pid] = time
                
                self.gantt.append(pid)
                self.gantt_priority.append(priority)
                remaining -= 1
                quantum_rem -= 1
                time += 1
                
                if remaining > 0:
                    self.wait_since[pid] = time

                if remaining > 0:
                    if quantum_rem == 0:
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
                    self.wait_since.pop(pid, None)
                    self.total_wait.pop(pid, None)
                    self.last_run_time.pop(pid, None)
            else:
                self.gantt.append(-1)
                self.gantt_priority.append(-1)
                time += 1

    def _get_priority(self, data_pointer: int, processes: list, current_time: int) -> int:
        """Query the DPO policy network for a priority assignment."""
        obs = self._get_observation(data_pointer, processes, current_time)
        flat = obs.ravel().astype(np.float32)
        tensor = torch.tensor(flat, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(tensor)
        
        return int(torch.argmax(logits).item())

    def _get_observation(self, data_pointer: int, processes: list, current_time: int) -> np.ndarray:
        """
        Build the 9-feature observation matching the training environment.

        See ml_prio.py:_get_observation() for the full feature layout.
        """
        obs = np.full((self.encoder_context + 1, 9), -1, dtype=np.float32)

        # ============================================================
        # ROW 0: next arriving process (not yet in queue)
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
        # ROWS 1+: current queue snapshot
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
        """Return tracking statistics for debugging."""
        return {
            'wait_since': dict(self.wait_since),
            'total_wait': dict(self.total_wait),
            'last_run_time': dict(self.last_run_time),
            'first_run_time': dict(self.first_run_time)
        }


# For testing compatibility
if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    # Create dummy data
    test_data = np.array([
        [0, 0, 10],
        [1, 1, 5],
        [2, 2, 8]
    ])
    
    # Test scheduler
    scheduler = DPOPriority(
        test_data,
        encoder_context=30,
        max_priority=10,
        time_quantum=4,
        model_path="model_weights/dpo_trained_model.pt"
    )
    
    scheduler.time_run()
    scheduler.calc_stats()
    scheduler.print_stats()