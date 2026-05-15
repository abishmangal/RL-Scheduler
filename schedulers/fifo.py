from .scheduler import Scheduler
from collections import deque
import numpy as np

class FIFO(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data=data)

    def run(self):
        # ✅ Work on a local copy, don't modify self.data
        # Build process list from self.data (don't modify original)
        processes = []
        for i in range(self.data.shape[0]):
            processes.append([
                int(self.data[i, 0]),  # pid
                int(self.data[i, 1]),  # arrival
                int(self.data[i, 2])   # instructions
            ])
        
        # Sort by arrival time
        processes.sort(key=lambda x: x[1])
        
        deq = deque()
        time = 0
        ptr = 0  # Pointer to track which processes have been added
        self.gantt = []

        while len(deq) != 0 or ptr < len(processes):
            # Add all processes arriving at current time (non-destructive)
            while ptr < len(processes) and processes[ptr][1] == time:
                deq.append(processes[ptr])
                ptr += 1  # ✅ Just increment pointer, don't delete!

            if deq:
                process = deq.popleft()
                self.gantt.append(process[0])  # Store PID
                process[2] -= 1  # Decrease InstructionCount
                
                if process[2] == 0:
                    # Process complete - don't add back
                    time += 1
                    continue
                else:
                    # Add back to front of queue
                    deq.appendleft(process)
            else:
                # Queue empty - idle
                self.gantt.append(-1)

            time += 1
        return