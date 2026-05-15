"""
schedulers/dpo_prio.py
======================
CPU scheduler using DPO-trained policy for initial environment (5 features)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'priority_prediction'))

from .scheduler import Scheduler
import heapq
import numpy as np
import torch
from network import FeedForwardNN


class DPOPriority(Scheduler):
    """
    Priority scheduler driven by a DPO-trained policy.
    For initial environment: 5 features, NO time quantum, NO aging.
    
    Parameters
    ----------
    data            : np.ndarray — process table [PID, ArrivalTime, Instructions]
    encoder_context : int        — queue slots in observation
    max_priority    : int        — discrete priority levels
    model_path      : str        — path to saved DPO weights (.pt)
    """

    def __init__(self, data, **kwargs):
        # Initialize parent Scheduler with data (this sets pids, arrivals, instr_count)
        super().__init__(data)
        
        # Store raw data (already in self.data from parent)
        self.raw_data = data

        try:
            self.encoder_context = kwargs['encoder_context']
            self.max_priority = kwargs['max_priority']
        except KeyError:
            raise ValueError(
                "DPOPriority requires 'encoder_context' and 'max_priority' kwargs."
            )

        # Observation dimension: (encoder_context+1) * 5 (5 features)
        obs_dim = (self.encoder_context + 1) * 5
        self.model = FeedForwardNN(obs_dim, self.max_priority)

        model_path = kwargs.get('model_path', 'model_weights/dpo_scheduler.pt')
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
        print(f"[DPOPriority] Observation dim: {obs_dim} (5 features)")
        print(f"[DPOPriority] Processes: {len(self.pids)}")

    def run(self):
        """
        Run the scheduler simulation.
        This method is required by the abstract Scheduler class.
        """
        # Build process list: [pid, arrival, total, remaining]
        processes = []
        for i in range(self.raw_data.shape[0]):
            pid = int(self.raw_data[i, 0])
            arrival = int(self.raw_data[i, 1])
            instr = int(self.raw_data[i, 2])
            processes.append([pid, arrival, instr, instr])

        # Sort by arrival time
        processes.sort(key=lambda x: x[1])
        
        # Priority queue as min-heap: (priority, pid, arrival, total, remaining)
        self.execution_queue = []
        data_pointer = 0
        time = 0
        self.gantt = []  # This is used by parent class for statistics

        while self.execution_queue or data_pointer < len(processes):
            # Add all processes arriving at current time
            while (data_pointer < len(processes) and 
                   processes[data_pointer][1] == time):
                proc = processes[data_pointer]
                priority = self._get_priority(data_pointer, processes, time)
                heapq.heappush(
                    self.execution_queue,
                    (priority, proc[0], proc[1], proc[2], proc[3])
                )
                data_pointer += 1

            if self.execution_queue:
                priority, pid, arrival, total, remaining = heapq.heappop(self.execution_queue)
                
                # Record in gantt chart
                self.gantt.append(pid)
                remaining -= 1
                time += 1

                if remaining > 0:
                    # Push back with updated remaining instructions
                    heapq.heappush(
                        self.execution_queue,
                        (priority, pid, arrival, total, remaining)
                    )
            else:
                # No processes ready - idle tick
                self.gantt.append(-1)
                time += 1

    def _get_priority(self, data_pointer: int, processes: list, current_time: int) -> int:
        """Use DPO model to select priority."""
        obs = self._get_observation(data_pointer, processes, current_time)
        flat = obs.ravel().astype(np.float32)
        tensor = torch.tensor(flat, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(tensor)
        
        return int(torch.argmax(logits).item())

    def _get_observation(self, data_pointer: int, processes: list, current_time: int) -> np.ndarray:
        """
        Build observation for initial environment (5 features).
        Shape: (encoder_context + 1, 5)
        Features: [PID, arrival, total_instructions, remaining_instructions, priority]
        """
        obs = np.full((self.encoder_context + 1, 5), -1, dtype=np.float32)

        # Row 0: next arriving process
        if data_pointer < len(processes):
            proc = processes[data_pointer]
            obs[0, :4] = [proc[0], proc[1], proc[2], proc[3]]
            # obs[0, 4] stays -1 (no priority yet)

        # Rows 1+: current queue snapshot
        for i, (priority, pid, arrival, total, remaining) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            obs[i + 1] = [pid, arrival, total, remaining, priority]

        return obs


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
        model_path="model_weights/dpo_scheduler.pt"  # Update path
    )
    
    scheduler.time_run()
    scheduler.calc_stats()
    scheduler.print_stats()