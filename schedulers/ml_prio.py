import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'priority_prediction'))

from .scheduler import Scheduler
import heapq
import numpy as np
import torch
from network import FeedForwardNN


class MLPriority(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data=None)
        self.raw_data = data
        self.pids = data[:, 0].astype(int)
        self.arrivals = data[:, 1]
        self.instr_count = data[:, 2]

        try:
            self.encoder_context = kwargs['encoder_context']
            self.max_priority = kwargs['max_priority']
        except KeyError:
            print("MLPriority must be instantiated with 'encoder_context' and 'max_priority' kwargs")

        self.time_quantum = kwargs.get('time_quantum', 4)

        # FIX 1: * 6 instead of * 5 (added quantum_remaining as 6th feature)
        self.model = FeedForwardNN((self.encoder_context + 1) * 6, self.max_priority)

        # FIX 2: use model_path kwarg with absolute path resolution
        model_path = kwargs.get('model_path', 'model_weights/ml_priority_scheduler_dataset3_5mil_30context.pt')
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        full_path = os.path.join(project_root, model_path)
        self.model.load_state_dict(torch.load(full_path, map_location='cpu'))
        self.model.eval()

    def run(self):
        # Build process list [pid, arrival, instr, remaining]
        processes = []
        for pid in range(self.raw_data.shape[0]):
            arrival = int(self.raw_data[pid, 1])
            instr = int(self.raw_data[pid, 2])
            processes.append([pid, arrival, instr, instr])

        # FIX 3: heapq instead of PriorityQueue (no mutex, no heap invariant bugs)
        # tuple: (priority, pid, arrival, instr, remaining, quantum_remaining)
        self.execution_queue = []
        data_pointer = 0
        time = 0

        while self.execution_queue or data_pointer < len(processes):

            # Add all processes arriving at current time
            while data_pointer < len(processes) and processes[data_pointer][1] == time:
                proc = processes[data_pointer]
                priority = self._get_priority(data_pointer, processes)
                heapq.heappush(self.execution_queue,
                               (priority, proc[0], proc[1], proc[2], proc[3], self.time_quantum))
                data_pointer += 1

            if self.execution_queue:
                priority, pid, arrival, instr, remaining, quantum_rem = heapq.heappop(self.execution_queue)
                self.gantt.append(pid)
                remaining -= 1
                quantum_rem -= 1

                if remaining == 0:
                    # Process complete
                    time += 1
                    continue
                elif quantum_rem == 0:
                    # FIX 4: preempt - push back with fresh quantum
                    heapq.heappush(self.execution_queue,
                                   (priority, pid, arrival, instr, remaining, self.time_quantum))
                else:
                    heapq.heappush(self.execution_queue,
                                   (priority, pid, arrival, instr, remaining, quantum_rem))
            else:
                self.gantt.append(-1)

            time += 1

    def _get_priority(self, data_pointer, processes):
        obs = self._get_observation(data_pointer, processes).ravel().astype(np.float32)
        with torch.no_grad():
            logits = self.model(obs)
        return int(torch.argmax(logits).item())

    def _get_observation(self, data_pointer, processes):
        # FIX 5: obs shape (encoder_context+1, 6) to match trained model
        obs = np.full((self.encoder_context + 1, 6), -1, dtype=np.float32)

        # Row 0: next arriving process
        proc = processes[data_pointer]
        obs[0, :4] = [proc[0], proc[1], proc[2], proc[3]]
        obs[0, 4] = -1                  # no priority assigned yet
        obs[0, 5] = self.time_quantum   # will get full quantum when scheduled

        # Rows 1+: current queue snapshot
        for i, (priority, pid, arrival, instr, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            obs[i + 1] = [pid, arrival, instr, remaining, priority, quantum_rem]

        return obs