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
        self.aging_threshold = kwargs.get('aging_threshold', 50)
        self.aging_boost = kwargs.get('aging_boost', 1)
        self.aging_interval = kwargs.get('aging_interval', 100)

        # FIX: * 7 (added wait_time as 7th feature)
        self.model = FeedForwardNN((self.encoder_context + 1) * 7, self.max_priority)

        model_path = kwargs.get('model_path', 'model_weights/ml_priority_scheduler_dataset3_5mil_30context.pt')
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        full_path = os.path.join(project_root, model_path)
        self.model.load_state_dict(torch.load(full_path, map_location='cpu'))
        self.model.eval()

    def run(self):
        processes = []
        for pid in range(self.raw_data.shape[0]):
            arrival = int(self.raw_data[pid, 1])
            instr = int(self.raw_data[pid, 2])
            processes.append([pid, arrival, instr, instr])

        self.execution_queue = []
        data_pointer = 0
        time = 0

        # Wait time tracking (matches env)
        self.wait_since = {}
        self.total_wait = {}
        self.assigned_priority = {}
        self.last_aging_time = 0

        while self.execution_queue or data_pointer < len(processes):

            # Add all processes arriving at current time
            while data_pointer < len(processes) and processes[data_pointer][1] == time:
                proc = processes[data_pointer]
                priority = self._get_priority(data_pointer, processes, time)
                self.wait_since[proc[0]] = time
                self.total_wait[proc[0]] = 0
                self.assigned_priority[proc[0]] = priority
                heapq.heappush(self.execution_queue,
                               (priority, proc[0], proc[1], proc[2], proc[3], self.time_quantum))
                data_pointer += 1

            # Apply aging periodically
            self._apply_aging(time)

            if self.execution_queue:
                priority, pid, arrival, instr, remaining, quantum_rem = heapq.heappop(self.execution_queue)

                # Accumulate wait time before running
                wait_accumulated = time - self.wait_since.get(pid, time)
                self.total_wait[pid] = self.total_wait.get(pid, 0) + wait_accumulated

                self.gantt.append(pid)
                remaining -= 1
                quantum_rem -= 1
                time += 1

                if remaining == 0:
                    # Process complete — clean up
                    self.wait_since.pop(pid, None)
                    self.total_wait.pop(pid, None)
                    self.assigned_priority.pop(pid, None)
                    continue
                elif quantum_rem == 0:
                    # Preempt — restore original priority, aging reapplies separately
                    original_priority = self.assigned_priority.get(pid, priority)
                    self.wait_since[pid] = time
                    heapq.heappush(self.execution_queue,
                                   (original_priority, pid, arrival, instr, remaining, self.time_quantum))
                else:
                    original_priority = self.assigned_priority.get(pid, priority)
                    heapq.heappush(self.execution_queue,
                                   (original_priority, pid, arrival, instr, remaining, quantum_rem))
            else:
                self.gantt.append(-1)
                time += 1

    def _apply_aging(self, current_time):
        """Apply priority aging periodically — matches env logic."""
        if current_time - self.last_aging_time < self.aging_interval:
            return

        self.last_aging_time = current_time
        aged = False
        new_queue = []

        for priority, pid, arrival, instr, remaining, quantum_rem in self.execution_queue:
            current_wait = self.total_wait.get(pid, 0) + (
                current_time - self.wait_since.get(pid, current_time)
            )
            if current_wait >= self.aging_threshold:
                original = self.assigned_priority.get(pid, priority)
                boost_amount = (current_wait // self.aging_threshold) * self.aging_boost
                new_priority = max(0, original - boost_amount)
                if new_priority != priority:
                    aged = True
                new_queue.append((new_priority, pid, arrival, instr, remaining, quantum_rem))
            else:
                new_queue.append((priority, pid, arrival, instr, remaining, quantum_rem))

        if aged:
            heapq.heapify(new_queue)
            self.execution_queue = new_queue

    def _get_priority(self, data_pointer, processes, current_time):
        obs = self._get_observation(data_pointer, processes, current_time).ravel().astype(np.float32)
        with torch.no_grad():
            logits = self.model(obs)
        return int(torch.argmax(logits).item())

    def _get_observation(self, data_pointer, processes, current_time):
        # FIX: obs shape (encoder_context+1, 7) — added wait_time as 7th feature
        obs = np.full((self.encoder_context + 1, 7), -1, dtype=np.float32)

        # Row 0: next arriving process
        proc = processes[data_pointer]
        obs[0, :4] = [proc[0], proc[1], proc[2], proc[3]]
        obs[0, 4] = -1                  # no priority yet
        obs[0, 5] = self.time_quantum   # full quantum when scheduled
        obs[0, 6] = 0                   # wait time = 0, not in queue yet

        # Rows 1+: current queue snapshot
        for i, (priority, pid, arrival, instr, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            wait_time = self.total_wait.get(pid, 0) + (
                current_time - self.wait_since.get(pid, current_time)
            )
            obs[i + 1] = [pid, arrival, instr, remaining, priority, quantum_rem, wait_time]

        return obs