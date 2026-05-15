from .scheduler import Scheduler
from collections import deque
import numpy as np


class RoundRobin(Scheduler):
    """
    Round Robin Scheduler - Clean implementation
    """
    
    def __init__(self, data, **kwargs):
        super().__init__(data=data)
        self.quantum = kwargs.get('time_quantum', 4)
    
    def run(self):
        # Build process list [pid, arrival, remaining]
        processes = []
        for i in range(self.data.shape[0]):
            processes.append([
                int(self.data[i, 0]),  # pid
                int(self.data[i, 1]),  # arrival
                int(self.data[i, 2])   # remaining instructions
            ])
        processes.sort(key=lambda x: x[1])
        
        queue = deque()
        ptr = 0
        time = 0
        self.gantt = []
        
        # Track quantum for current process
        current_pid = None
        quantum_left = 0
        
        while ptr < len(processes) or queue or current_pid is not None:
            # Add arriving processes
            while ptr < len(processes) and processes[ptr][1] == time:
                queue.append(processes[ptr])
                ptr += 1
            
            # Start new process if none running
            if current_pid is None and queue:
                current_pid = queue.popleft()
                quantum_left = self.quantum
            
            if current_pid is not None:
                # Run current process
                self.gantt.append(current_pid[0])
                current_pid[2] -= 1  # Decrement remaining
                quantum_left -= 1
                time += 1
                
                # Check if completed
                if current_pid[2] == 0:
                    current_pid = None
                    continue
                
                # Check if quantum expired
                if quantum_left == 0:
                    queue.append(current_pid)
                    current_pid = None
            else:
                # Idle
                self.gantt.append(-1)
                time += 1