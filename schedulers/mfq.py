from .scheduler import Scheduler
from collections import deque
import numpy as np

class MFQ(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data)
        raw = np.array(data) if not isinstance(data, np.ndarray) else data
        self.processes = [[int(r[0]), int(r[1]), int(r[2]), int(r[2])] for r in raw]
        self.queues = [deque() for _ in range(3)]
        self.quantums = [4, 8, 15]
        self.quantum_used = {}

    def run(self):
        time = 0
        ptr = 0

        while any(self.queues) or ptr < len(self.processes):
            while ptr < len(self.processes) and self.processes[ptr][1] <= time:
                proc = self.processes[ptr]
                self.queues[0].append(proc)
                self.quantum_used[proc[0]] = 0
                ptr += 1

            executed = False
            for i in range(len(self.queues)):
                if self.queues[i]:
                    executed = True
                    proc = self.queues[i].popleft()
                    self.gantt.append(int(proc[0]))
                    proc[3] -= 1
                    self.quantum_used[proc[0]] += 1

                    if proc[3] == 0:
                        del self.quantum_used[proc[0]]
                    elif self.quantum_used[proc[0]] >= self.quantums[i]:
                        self.quantum_used[proc[0]] = 0
                        next_q = min(i + 1, len(self.queues) - 1)
                        self.queues[next_q].append(proc)
                    else:
                        self.queues[i].append(proc)
                    break

            if not executed:
                self.gantt.append(-1)

            time += 1
