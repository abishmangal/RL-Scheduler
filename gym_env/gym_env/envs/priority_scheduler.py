import gymnasium as gym
from gymnasium import spaces
import heapq
import numpy as np


class PrioritySchedulerEnv(gym.Env):
    def __init__(self, data, encoder_context, max_priority, time_quantum=4):
        super().__init__()

        self.data = data
        self.encoder_context = encoder_context
        self.max_priority = max_priority
        self.time_quantum = time_quantum  # NEW: ticks before forced preemption

        self.observation_space = spaces.Box(
            low=-1, high=np.inf,
            shape=(encoder_context + 1, 6),  # NEW: +1 col for quantum_remaining
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(max_priority)

        self.reset()

    def _get_info(self):
        return {}

    def _get_obs(self):
        obs = np.full((self.encoder_context + 1, 6), -1, dtype=np.int32)

        # Next arriving process (quantum_remaining = full quantum, not yet scheduled)
        if self.data_pointer < len(self.processes):
            pid, arrival, instr, remaining = self.processes[self.data_pointer]
            obs[0, :4] = [pid, arrival, instr, remaining]
            obs[0, 4] = -1                   # no priority yet
            obs[0, 5] = self.time_quantum    # will get a full quantum when added

        # Current queue: (priority, pid, arrival, instr, remaining, quantum_remaining)
        for i, (priority, pid, arrival, instr, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            obs[i + 1] = [pid, arrival, instr, remaining, priority, quantum_rem]

        return obs

    def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            if options and "new_data" in options:
                self.data = options["new_data"]

            self.processes = []
            for pid in range(self.data.shape[0]):
                arrival = int(self.data[pid, 1])
                instr = int(self.data[pid, 2])
                self.processes.append([pid, arrival, instr, instr])

            self.current_time = 0
            self.data_pointer = 0
            self.execution_queue = []       # (priority, pid, arrival, instr, remaining, quantum_remaining)
            self.completed_processes = []

            self.total_completed = 0
            self.total_turnaround = 0

            # NEW: response time tracking
            self.first_run_time = {}
            self.avg_instructions = np.mean([p[2] for p in self.processes])

            return self._get_obs(), self._get_info()

    def step(self, action):
        new_completions = 0
        new_turnarounds = 0
        new_response_times = 0  # NEW

        if self.data_pointer < len(self.processes):
            proc = self.processes[self.data_pointer]

            # Execute until this process arrives
            delta_time = proc[1] - self.current_time
            if delta_time > 0:
                completed, turnarounds, response = self._execute_for_time(delta_time)  # NEW: unpack 3
                new_completions += completed
                new_turnarounds += turnarounds
                new_response_times += response  # NEW

            # Add arriving process with chosen priority and full quantum
            if isinstance(action, np.ndarray):
                priority = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                priority = int(action)

            heapq.heappush(
                self.execution_queue,
                (priority, proc[0], proc[1], proc[2], proc[3], self.time_quantum)
            )

            self.data_pointer += 1

        else:
            # No more arrivals -> finish everything
            remaining_time = sum(p[4] for p in self.execution_queue)
            if remaining_time > 0:
                completed, turnarounds, response = self._execute_for_time(remaining_time)  # NEW: unpack 3
                new_completions += completed
                new_turnarounds += turnarounds
                new_response_times += response  # NEW

        self.total_completed += new_completions
        self.total_turnaround += new_turnarounds

        queue_pressure = len(self.execution_queue)

        reward = (
            - new_turnarounds / self.avg_instructions        # turnaround (main signal)
            - 0.5 * new_response_times / self.avg_instructions  # response time (half weight)
            - 0.01 * queue_pressure                          # queue pressure (tiny weight)
        )

        terminated = (
            self.data_pointer == len(self.processes)
            and len(self.execution_queue) == 0
        )

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _execute_for_time(self, time_available):
        """
        Execute processes for up to time_available ticks with preemption.
        A process is preempted when its quantum_remaining hits 0.
        Returns (completed_count, sum_of_turnarounds, sum_of_response_times).
        """
        completed = 0
        total_turnaround = 0
        new_response = 0        # NEW
        remaining_time = time_available

        while remaining_time > 0 and self.execution_queue:
            priority, pid, arrival, instr, remaining, quantum_rem = heapq.heappop(self.execution_queue)

            # NEW: track first run time for response time — before current_time advances
            if pid not in self.first_run_time:
                self.first_run_time[pid] = self.current_time
                new_response += self.current_time - arrival  # response = first_run - arrival

            # Run for the minimum of: time left, instructions left, quantum left
            run_time = min(remaining_time, remaining, quantum_rem)

            self.current_time += run_time
            remaining_time -= run_time
            remaining -= run_time
            quantum_rem -= run_time

            if remaining == 0:
                # Process fully completed
                turnaround = self.current_time - arrival
                self.completed_processes.append((pid, turnaround))
                completed += 1
                total_turnaround += turnaround

            elif quantum_rem == 0:
                # Quantum exhausted - preempt and push back with fresh quantum
                heapq.heappush(
                    self.execution_queue,
                    (priority, pid, arrival, instr, remaining, self.time_quantum)
                )

            else:
                # Time ran out mid-quantum - push back with remaining quantum intact
                heapq.heappush(
                    self.execution_queue,
                    (priority, pid, arrival, instr, remaining, quantum_rem)
                )

        return completed, total_turnaround, new_response  # NEW: return 3 values

    def render(self):
        print(f"\nTime: {self.current_time}")
        print("Queue:")
        for p in self.execution_queue:
            print(f"  Priority {p[0]}: PID {p[1]}, Remaining: {p[4]}/{p[3]}, Quantum left: {p[5]}")
        print(f"Completed: {len(self.completed_processes)} processes")
        if self.completed_processes:
            avg_turnaround = sum(t for _, t in self.completed_processes) / len(self.completed_processes)
            print(f"Average Turnaround: {avg_turnaround:.2f}")