import os
import heapq
import numpy as np
import torch
from .scheduler import Scheduler
from priority_prediction.network import FeedForwardNN


class MLPriority(Scheduler):
    """
    ML-based Priority Scheduler using trained PPO policy.
    For initial environment: 5 features, NO time quantum, NO aging.
    """
    
    def __init__(self, data, **kwargs):
        super().__init__(data=None)
        
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
        self.model_path = kwargs.get('model_path', 'model_weights/ml_priority_scheduler_5mil_30context.pt')
        
        # Build observation dimension: (encoder_context+1) * 5 features
        obs_dim = (self.encoder_context + 1) * 5
        
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
        print(f"[MLPriority] Observation dim: {obs_dim}")
    
    def run(self):
        """Run the scheduler simulation"""
        
        # Build process list: [pid, arrival, total_instructions, remaining_instructions]
        processes = []
        for pid in range(self.raw_data.shape[0]):
            arrival = int(self.raw_data[pid, 1])
            instr = int(self.raw_data[pid, 2])
            processes.append([pid, arrival, instr, instr])
        
        # Priority queue as min-heap: (priority, pid, arrival, total, remaining)
        self.execution_queue = []
        data_pointer = 0
        time = 0
        self.gantt = []
        
        while self.execution_queue or data_pointer < len(processes):
            # Add all processes arriving at current time
            while (data_pointer < len(processes) and 
                   processes[data_pointer][1] == time):
                proc = processes[data_pointer]
                priority = self._get_priority(data_pointer, processes)
                heapq.heappush(
                    self.execution_queue,
                    (priority, proc[0], proc[1], proc[2], proc[3])
                )
                data_pointer += 1
            
            if self.execution_queue:
                # Get highest priority process (lowest number)
                priority, pid, arrival, total, remaining = heapq.heappop(self.execution_queue)
                
                # Run for 1 time unit
                self.gantt.append(pid)
                remaining -= 1
                time += 1
                
                # If not finished, push back to queue
                if remaining > 0:
                    heapq.heappush(
                        self.execution_queue,
                        (priority, pid, arrival, total, remaining)
                    )
            else:
                # No processes ready - idle tick
                self.gantt.append(-1)
                time += 1
    
    def _get_priority(self, data_pointer: int, processes: list) -> int:
        """Get priority prediction from ML model"""
        obs = self._get_observation(data_pointer, processes)
        flat = obs.ravel().astype(np.float32)
        tensor = torch.tensor(flat, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(tensor)
            priority = int(torch.argmax(logits).item())
        
        return priority
    
    def _get_observation(self, data_pointer: int, processes: list) -> np.ndarray:
        """
        Build observation for initial environment (5 features).
        Shape: (encoder_context + 1, 5)
        Features: [PID, arrival, total_instructions, remaining_instructions, priority]
        """
        obs = np.full((self.encoder_context + 1, 5), -1, dtype=np.float32)
        
        # Row 0: next arriving process (not yet in queue)
        if data_pointer < len(processes):
            proc = processes[data_pointer]
            obs[0, :4] = [proc[0], proc[1], proc[2], proc[3]]
            # obs[0, 4] stays -1 (no priority assigned yet)
        
        # Rows 1+: current queue snapshot (in priority order)
        for i, (priority, pid, arrival, total, remaining) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            obs[i + 1] = [pid, arrival, total, remaining, priority]
        
        return obs