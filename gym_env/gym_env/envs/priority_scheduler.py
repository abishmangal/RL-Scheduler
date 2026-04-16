import gymnasium as gym
from gymnasium import spaces
import heapq
import numpy as np


class PrioritySchedulerEnv(gym.Env):
    def __init__(self, data, encoder_context, max_priority, time_quantum=4,
                 aging_threshold=50, aging_boost=1, aging_interval=100):
        super().__init__()

        self.data = data
        self.encoder_context = encoder_context
        self.max_priority = max_priority
        self.time_quantum = time_quantum
        self.aging_threshold = aging_threshold
        self.aging_boost = aging_boost
        self.aging_interval = aging_interval

        self.observation_space = spaces.Box(
            low=-1, high=np.inf,
            shape=(encoder_context + 1, 7),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(max_priority)

        self.reset()

    def _get_info(self):
        return {}

    def _get_obs(self):
        obs = np.full((self.encoder_context + 1, 7), -1, dtype=np.int32)

        # Next arriving process
        if self.data_pointer < len(self.processes):
            pid, arrival, instr, remaining = self.processes[self.data_pointer]
            obs[0, :4] = [pid, arrival, instr, remaining]
            obs[0, 4] = -1
            obs[0, 5] = self.time_quantum
            obs[0, 6] = 0

        # Current queue
        for i, (priority, pid, arrival, instr, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            wait_time = self.total_wait.get(pid, 0) + (self.current_time - self.wait_since.get(pid, self.current_time))
            obs[i + 1] = [pid, arrival, instr, remaining, priority, quantum_rem, wait_time]

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
        self.execution_queue = []
        self.completed_processes = []

        self.total_completed = 0
        self.total_turnaround = 0

        # Response time tracking
        self.first_run_time = {}
        self.avg_instructions = np.mean([p[2] for p in self.processes])

        # Wait time tracking
        self.wait_since = {}
        self.total_wait = {}

        # Priority aging tracking
        self.assigned_priority = {}  # pid -> original assigned priority
        self.last_aging_time = 0     # FIX: always reset so new episodes age correctly

        return self._get_obs(), self._get_info()

    def _apply_aging(self):
        """
        Apply priority aging to processes that have waited too long.
        Returns True if any priorities were changed.
        """
        # Only apply aging periodically
        if self.current_time - self.last_aging_time < self.aging_interval:
            return False

        self.last_aging_time = self.current_time

        aged = False
        new_queue = []

        for priority, pid, arrival, instr, remaining, quantum_rem in self.execution_queue:
            current_wait = self.total_wait.get(pid, 0) + (
                self.current_time - self.wait_since.get(pid, self.current_time)
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

        return aged

    def step(self, action):
        new_completions = 0
        new_turnarounds = 0
        new_response_times = 0

        if self.data_pointer < len(self.processes):
            proc = self.processes[self.data_pointer]

            delta_time = proc[1] - self.current_time
            if delta_time > 0:
                completed, turnarounds, response = self._execute_for_time(delta_time)
                new_completions += completed
                new_turnarounds += turnarounds
                new_response_times += response

            if isinstance(action, np.ndarray):
                priority = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                priority = int(action)

            pid = proc[0]
            self.wait_since[pid] = self.current_time
            self.total_wait[pid] = 0
            self.assigned_priority[pid] = priority

            heapq.heappush(
                self.execution_queue,
                (priority, pid, proc[1], proc[2], proc[3], self.time_quantum)
            )

            self.data_pointer += 1

        else:
            remaining_time = sum(p[4] for p in self.execution_queue)
            if remaining_time > 0:
                completed, turnarounds, response = self._execute_for_time(remaining_time)
                new_completions += completed
                new_turnarounds += turnarounds
                new_response_times += response

        self.total_completed += new_completions
        self.total_turnaround += new_turnarounds

        queue_pressure = len(self.execution_queue)

        reward = (
            - new_turnarounds / self.avg_instructions
            - 0.5 * new_response_times / self.avg_instructions
            - 0.01 * queue_pressure
        )

        terminated = (
            self.data_pointer == len(self.processes)
            and len(self.execution_queue) == 0
        )

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _execute_for_time(self, time_available):
        """
        Execute processes for up to time_available ticks with preemption.
        Returns (completed_count, sum_of_turnarounds, sum_of_response_times).
        """
        completed = 0
        total_turnaround = 0
        new_response = 0
        remaining_time = time_available

        # Apply aging once before starting execution
        self._apply_aging()

        while remaining_time > 0 and self.execution_queue:
            priority, pid, arrival, instr, remaining, quantum_rem = heapq.heappop(self.execution_queue)

            # Accumulate wait time before this process runs
            wait_accumulated = self.current_time - self.wait_since.get(pid, self.current_time)
            self.total_wait[pid] = self.total_wait.get(pid, 0) + wait_accumulated

            # Track first run time for response time
            if pid not in self.first_run_time:
                self.first_run_time[pid] = self.current_time
                new_response += self.current_time - arrival

            run_time = min(remaining_time, remaining, quantum_rem)

            self.current_time += run_time
            remaining_time -= run_time
            remaining -= run_time
            quantum_rem -= run_time

            if remaining == 0:
                turnaround = self.current_time - arrival
                self.completed_processes.append((pid, turnaround))
                completed += 1
                total_turnaround += turnaround
                self.wait_since.pop(pid, None)
                self.total_wait.pop(pid, None)
                self.assigned_priority.pop(pid, None)

            elif quantum_rem == 0:
                # Preempt - restore original priority, aging will reapply separately
                original_priority = self.assigned_priority.get(pid, priority)
                self.wait_since[pid] = self.current_time
                heapq.heappush(
                    self.execution_queue,
                    (original_priority, pid, arrival, instr, remaining, self.time_quantum)
                )
                self._apply_aging()

            else:
                # Time ran out mid-quantum - push back with remaining quantum
                original_priority = self.assigned_priority.get(pid, priority)
                heapq.heappush(
                    self.execution_queue,
                    (original_priority, pid, arrival, instr, remaining, quantum_rem)
                )

        return completed, total_turnaround, new_response

    def render(self):
        print(f"\nTime: {self.current_time}")
        print("Queue:")
        for p in self.execution_queue:
            pid = p[1]
            wait = self.total_wait.get(pid, 0) + (self.current_time - self.wait_since.get(pid, self.current_time))
            orig = self.assigned_priority.get(pid, p[0])
            print(f"  Priority {p[0]} (orig {orig}): PID {pid}, "
                  f"Remaining: {p[4]}/{p[3]}, Quantum left: {p[5]}, Waiting: {wait}")
        print(f"Completed: {len(self.completed_processes)} processes")
        if self.completed_processes:
            avg_turnaround = sum(t for _, t in self.completed_processes) / len(self.completed_processes)
            print(f"Average Turnaround: {avg_turnaround:.2f}")