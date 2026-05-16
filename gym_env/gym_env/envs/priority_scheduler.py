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
        self.time_quantum = time_quantum  # ADDED

        # Changed from 5 to 6 features (added quantum_rem)
        self.observation_space = spaces.Box(
            low=-1, high=np.inf,
            shape=(encoder_context + 1, 6),  # 5 → 6
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(max_priority)

        self.reset()

    def _get_info(self):
        return {}

    def _get_obs(self):
        # Changed from 5 to 6 features
        obs = np.full((self.encoder_context + 1, 6), -1, dtype=np.int32)

        # Next arriving process
        if self.data_pointer < len(self.processes):
            pid, arrival, instr, remaining = self.processes[self.data_pointer]
            obs[0, :4] = [pid, arrival, instr, remaining]
            obs[0, 4] = -1  # no priority yet
            obs[0, 5] = self.time_quantum  # ADDED: quantum for new process

        # Current queue - now includes quantum_rem
        for i, (priority, pid, arrival, instr, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            obs[i + 1] = [pid, arrival, instr, remaining, priority, quantum_rem]  # 6 features

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
        self.execution_queue = []  # Now stores (priority, pid, arrival, instr, remaining, quantum_rem)
        self.completed_processes = []
        self.total_completed = 0
        self.total_turnaround = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        new_completions = 0
        new_turnarounds = 0

        if self.data_pointer < len(self.processes):
            proc = self.processes[self.data_pointer]

            delta_time = proc[1] - self.current_time
            if delta_time > 0:
                completed, turnarounds = self._execute_for_time(delta_time)
                new_completions += completed
                new_turnarounds += turnarounds

            # Add process with chosen priority
            if isinstance(action, np.ndarray):
                priority = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                priority = int(action)

            # Queue entry now includes quantum_rem
            heapq.heappush(
                self.execution_queue,
                (priority, proc[0], proc[1], proc[2], proc[3], self.time_quantum)
            )

            self.data_pointer += 1

        else:
            remaining_time = sum(p[4] for p in self.execution_queue)
            if remaining_time > 0:
                completed, turnarounds = self._execute_for_time(remaining_time)
                new_completions += completed
                new_turnarounds += turnarounds

        self.total_completed += new_completions
        self.total_turnaround += new_turnarounds

        reward = 100 * new_completions - new_turnarounds

        terminated = (
            self.data_pointer == len(self.processes)
            and len(self.execution_queue) == 0
        )

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _execute_for_time(self, time_available):
        completed = 0
        total_turnaround = 0
        remaining_time = time_available

        while remaining_time > 0 and self.execution_queue:
            # Queue now has 6 elements (added quantum_rem)
            priority, pid, arrival, instr, remaining, quantum_rem = heapq.heappop(self.execution_queue)

            # Calculate run time (limited by quantum as well)
            run_time = min(remaining_time, remaining, quantum_rem)

            self.current_time += run_time
            remaining_time -= run_time
            remaining -= run_time
            quantum_rem -= run_time

            if remaining == 0:
                # Process completed
                turnaround = self.current_time - arrival
                self.completed_processes.append((pid, turnaround))
                completed += 1
                total_turnaround += turnaround
            elif quantum_rem == 0:
                # Quantum exhausted - preempt, restore full quantum
                heapq.heappush(
                    self.execution_queue,
                    (priority, pid, arrival, instr, remaining, self.time_quantum)
                )
            else:
                # Still has quantum - push back with remaining quantum
                heapq.heappush(
                    self.execution_queue,
                    (priority, pid, arrival, instr, remaining, quantum_rem)
                )

        return completed, total_turnaround

    def render(self):
        print(f"\nTime: {self.current_time}")
        print("Queue:")
        for p in self.execution_queue:
            print(f"  Priority {p[0]}: PID {p[1]}, Remaining: {p[4]}/{p[3]}, Quantum left: {p[5]}")
        print(f"Completed: {len(self.completed_processes)} processes")
        if self.completed_processes:
            avg_turnaround = sum(t for _, t in self.completed_processes) / len(self.completed_processes)
            print(f"Average Turnaround: {avg_turnaround:.2f}")