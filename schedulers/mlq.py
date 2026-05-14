import numpy as np
from .scheduler import Scheduler
from collections import deque

class MLQ(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data)
        raw = np.array(data) if not isinstance(data, np.ndarray) else data
        self.processes = [[int(r[0]), int(r[1]), int(r[2]), int(r[2])] for r in raw]
        self.queues = [deque(), deque(), deque()]

    def run(self):
        time = 0
        ptr = 0
        while any(self.queues) or ptr < len(self.processes):
            while ptr < len(self.processes) and self.processes[ptr][1] <= time:
                priority = np.random.randint(0, 3)
                self.queues[priority].append(self.processes[ptr])
                ptr += 1

            worked = False
            for queue in self.queues:
                if queue:
                    worked = True
                    proc = queue.popleft()
                    self.gantt.append(int(proc[0]))
                    proc[3] -= 1
                    if proc[3] > 0:
                        queue.append(proc)
                    break

            if not worked:
                self.gantt.append(-1)

            time += 1
