"""
schedulers/dqn_prio.py
======================
CPU scheduler using a trained DQN Q-network to assign priorities.
Drop-in replacement for MLPriority.
"""

from .scheduler import Scheduler
import heapq
import numpy as np
import torch
from priority_prediction.network import FeedForwardNN


class DQNPriority(Scheduler):
    """
    Priority scheduler driven by a DQN-trained Q-network.
    Environment: 6 features (PID, arrival, total, remaining, priority, quantum_rem).
    Uses time quantum preemption. Models must be trained on 6-feature env.

    Parameters
    ----------
    data            : np.ndarray  — process table [PID, ArrivalTime, Instructions]
    encoder_context : int         — queue slots in observation (match training)
    max_priority    : int         — discrete priority levels (match training)
    time_quantum    : int         — time slice before preemption
    model_path      : str         — path to saved DQN weights (.pt)
    """

    def __init__(self, data, **kwargs):
        # FIX: Initialize parent with data (sets pids, arrivals, instr_count)
        super().__init__(data)
        self.raw_data = data
        
        try:
            self.encoder_context = kwargs['encoder_context']
            self.max_priority = kwargs['max_priority']
        except KeyError:
            raise ValueError(
                "DQNPriority requires 'encoder_context' and 'max_priority' kwargs."
            )

        self.time_quantum = kwargs.get('time_quantum', 4)

        # Observation dimension: (encoder_context+1) * 6 features (matches gym env)
        obs_dim = (self.encoder_context + 1) * 6
        self.model = FeedForwardNN(obs_dim, self.max_priority)

        model_path = kwargs.get('model_path', 'model_weights/dqn_scheduler.pt')
        if not os.path.isabs(model_path):
            project_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..'
            )
            model_path = os.path.normpath(
                os.path.join(project_root, model_path)
            )

        checkpoint = torch.load(model_path, map_location='cpu')
        # Handle both full checkpoint and just state_dict
        if 'q_net_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['q_net_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"[DQNPriority] Loaded weights from: {model_path}")
        print(f"[DQNPriority] Observation dim: {obs_dim} (6 features, context={self.encoder_context}, time_quantum={self.time_quantum})")
        print(f"[DQNPriority] Processes: {len(self.pids)}")

    def run(self):
        """Run the scheduler simulation."""
        # Build process list: [pid, arrival, total, remaining]
        processes = []
        for i in range(self.raw_data.shape[0]):
            pid = int(self.raw_data[i, 0])
            arrival = int(self.raw_data[i, 1])
            instr = int(self.raw_data[i, 2])
            processes.append([pid, arrival, instr, instr])
        
        # Sort by arrival time
        processes.sort(key=lambda x: x[1])

        self.execution_queue = []
        data_pointer = 0
        time = 0
        self.gantt = []  # Track which process runs at each time step

        while self.execution_queue or data_pointer < len(processes):
            # Add all processes arriving at current time
            while (data_pointer < len(processes) and 
                   processes[data_pointer][1] == time):
                proc = processes[data_pointer]
                priority = self._get_priority(data_pointer, processes)
                heapq.heappush(
                    self.execution_queue,
                    (priority, proc[0], proc[1], proc[2], proc[3], self.time_quantum)
                )
                data_pointer += 1

            if self.execution_queue:
                priority, pid, arrival, total, remaining, quantum_rem = heapq.heappop(self.execution_queue)
                
                self.gantt.append(pid)
                remaining -= 1
                quantum_rem -= 1
                time += 1

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
                # No processes ready - idle tick
                self.gantt.append(-1)
                time += 1

    def _get_priority(self, data_pointer: int, processes: list) -> int:
        """Use the DQN model to select priority for the next process."""
        obs = self._get_observation(data_pointer, processes)
        flat = obs.ravel().astype(np.float32)
        tensor = torch.tensor(flat, dtype=torch.float32)
        
        with torch.no_grad():
            q_vals = self.model(tensor)
        
        return int(torch.argmax(q_vals).item())

    def _get_observation(self, data_pointer: int, processes: list) -> np.ndarray:
        """
        Build observation (6 features — matches the gym env with time quantum).
        Shape: (encoder_context + 1, 6)
        Features: [PID, arrival, total_instructions, remaining_instructions, priority, quantum_rem]
        """
        obs = np.full((self.encoder_context + 1, 6), -1, dtype=np.float32)

        # Row 0: next arriving process (not yet in queue)
        if data_pointer < len(processes):
            proc = processes[data_pointer]
            obs[0, :4] = [proc[0], proc[1], proc[2], proc[3]]
            obs[0, 5] = self.time_quantum  # quantum for arriving process

        # Rows 1+: current queue snapshot
        for i, (priority, pid, arrival, total, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            obs[i + 1] = [pid, arrival, total, remaining, priority, quantum_rem]

        return obs

    def get_average_turnaround(self) -> float:
        """Calculate average turnaround time from completed processes."""
        if not hasattr(self, 'completed_processes') or not self.completed_processes:
            return 0.0
        
        total_turnaround = sum(t for _, t in self.completed_processes)
        return total_turnaround / len(self.completed_processes)