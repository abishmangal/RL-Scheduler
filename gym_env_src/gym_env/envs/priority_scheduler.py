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
        self.time_quantum = time_quantum

        # Updated weights - response now has higher priority
        self.weights = {
            'turnaround': 1.0,
            'waiting': 0.5,          # Reduced - already good
            'response': 2.5,         # INCREASED - primary focus!
            'queue': 0.01,
            'starvation': 1.0,
            'progress': 0.5          # Increased
        }
        
        self.starvation_threshold = 100
        
        # 9 features: +queue_position, +wait_time, +time_since_last_run
        self.observation_space = spaces.Box(
            low=-1, high=np.inf,
            shape=(encoder_context + 1, 9),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(max_priority)

        self.reset()

    def _get_info(self):
        return {}

    def _get_obs(self):
        obs = np.full((self.encoder_context + 1, 9), -1, dtype=np.int32)

        # Row 0: Next arriving process
        if self.data_pointer < len(self.processes):
            pid, arrival, instr, remaining = self.processes[self.data_pointer]
            obs[0, 0] = pid
            obs[0, 1] = arrival
            obs[0, 2] = instr
            obs[0, 3] = remaining
            obs[0, 4] = -1
            obs[0, 5] = self.time_quantum
            obs[0, 6] = len(self.execution_queue)
            obs[0, 7] = 0
            obs[0, 8] = 0

        # Rows 1+: Processes in queue
        for i, (priority, pid, arrival, instr, remaining, quantum_rem) in enumerate(self.execution_queue):
            if i >= self.encoder_context:
                break
            
            last_ran = self.last_run_time.get(pid, 0)
            time_since_last_run = self.current_time - last_ran if last_ran > 0 else 0
            
            total_wait_time = self.total_wait.get(pid, 0) + (
                self.current_time - self.wait_since.get(pid, self.current_time)
            )
            
            obs[i + 1, 0] = pid
            obs[i + 1, 1] = arrival
            obs[i + 1, 2] = instr
            obs[i + 1, 3] = remaining
            obs[i + 1, 4] = priority
            obs[i + 1, 5] = quantum_rem
            obs[i + 1, 6] = i                      # Queue position
            obs[i + 1, 7] = total_wait_time
            obs[i + 1, 8] = time_since_last_run

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
        
        self.first_run_time = {}
        self.avg_instructions = np.mean([p[2] for p in self.processes]) if self.processes else 1
        
        self.wait_since = {}
        self.total_wait = {}
        self.last_run_time = {}
        self.assigned_priority = {}
        
        # ============================================================
        # DENSE RESPONSE TRACKING - Track cumulative response penalty
        # ============================================================
        self.last_response_penalty_time = {}  # When we last applied response penalty
        self.cumulative_response_penalty = 0   # Running total for current step
        
        self.prev_queue_pressure = 0
        self.reward_history = []
        self.last_aging_time = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        new_completions = 0
        sum_turnarounds = 0      # Renamed for clarity (was new_turnarounds)
        sum_response_times = 0   # Renamed for clarity (was new_response_times)
        sum_waiting_times = 0    # Renamed for clarity (was new_waiting_times)
        starvation_penalty = 0
        progress_bonus = 0
        
        # ============================================================
        # DENSE RESPONSE PENALTY - Applied every tick waiting for first run
        # ============================================================
        dense_response_penalty = 0

        if self.data_pointer < len(self.processes):
            proc = self.processes[self.data_pointer]

            delta_time = proc[1] - self.current_time
            if delta_time > 0:
                completed, turnarounds, response, waiting = self._execute_for_time(delta_time)
                new_completions += completed
                sum_turnarounds += turnarounds
                sum_response_times += response
                sum_waiting_times += waiting

            if isinstance(action, np.ndarray):
                priority = int(action.item()) if action.ndim == 0 else int(action[0])
            else:
                priority = int(action)

            pid = proc[0]
            self.wait_since[pid] = self.current_time
            self.total_wait[pid] = 0
            self.assigned_priority[pid] = priority
            self.last_run_time[pid] = 0
            self.last_response_penalty_time[pid] = self.current_time  # Initialize

            heapq.heappush(
                self.execution_queue,
                (priority, pid, proc[1], proc[2], proc[3], self.time_quantum)
            )

            self.data_pointer += 1

        else:
            remaining_time = sum(p[4] for p in self.execution_queue)
            if remaining_time > 0:
                completed, turnarounds, response, waiting = self._execute_for_time(remaining_time)
                new_completions += completed
                sum_turnarounds += turnarounds
                sum_response_times += response
                sum_waiting_times += waiting

        self.total_completed += new_completions
        self.total_turnaround += sum_turnarounds

        queue_pressure = len(self.execution_queue)

        # ============================================================
        # DENSE RESPONSE PENALTY - Apply penalty for every tick processes wait
        # for their FIRST run (not just once at first run)
        # ============================================================
        for pid in self.wait_since:
            # Only apply to processes that haven't had their first run yet
            if pid not in self.first_run_time:
                wait_time = self.current_time - self.wait_since[pid]
                if wait_time > 0:
                    # Apply penalty every response_interval ticks
                    response_interval = 10  # Apply penalty every 10 ticks
                    last_penalty = self.last_response_penalty_time.get(pid, self.wait_since[pid])
                    
                    if self.current_time - last_penalty >= response_interval:
                        # Update last penalty time
                        self.last_response_penalty_time[pid] = self.current_time
                        # Add dense penalty proportional to wait time
                        dense_response_penalty += (wait_time / self.avg_instructions) * 0.5

        # Normalize dense response penalty
        dense_response_penalty = -self.weights['response'] * dense_response_penalty

        # Reward components
        turnaround_penalty = -self.weights['turnaround'] * sum_turnarounds / self.avg_instructions
        waiting_penalty = -self.weights['waiting'] * sum_waiting_times / self.avg_instructions
        # Original sparse response penalty
        response_penalty = -self.weights['response'] * sum_response_times / self.avg_instructions
        queue_penalty = -self.weights['queue'] * queue_pressure

        # Starvation penalty using TOTAL wait time
        starvation_penalty = 0
        for pid in self.wait_since:
            total_wait = self.total_wait.get(pid, 0) + (
                self.current_time - self.wait_since.get(pid, self.current_time)
            )
            if total_wait > self.starvation_threshold:
                excess = total_wait - self.starvation_threshold
                starvation_penalty += self.weights['starvation'] * (excess / 50)
        starvation_penalty = -starvation_penalty

        # Progress bonus
        if new_completions > 0:
            progress_bonus += self.weights['progress'] * new_completions
            avg_completion_time = self.total_turnaround / max(1, self.total_completed)
            if avg_completion_time < self.avg_instructions:
                progress_bonus += 0.1 * new_completions

        if queue_pressure < self.prev_queue_pressure:
            reduction = self.prev_queue_pressure - queue_pressure
            progress_bonus += 0.05 * reduction

        # ============================================================
        # COMBINE REWARDS - Now includes dense response penalty
        # ============================================================
        reward = (
            turnaround_penalty +
            waiting_penalty +
            response_penalty +      # Sparse (once per process)
            dense_response_penalty + # Dense (every 10 ticks while waiting)
            queue_penalty +
            starvation_penalty +
            progress_bonus
        )

        reward = np.clip(reward, -15.0, 5.0)  # Slightly wider range for dense penalty

        self.prev_queue_pressure = queue_pressure
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)

        terminated = (
            self.data_pointer == len(self.processes)
            and len(self.execution_queue) == 0
        )

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _execute_for_time(self, time_available):
        completed = 0
        total_turnaround = 0
        new_response = 0
        total_waiting = 0
        remaining_time = time_available

        while remaining_time > 0 and self.execution_queue:
            priority, pid, arrival, instr, remaining, quantum_rem = heapq.heappop(self.execution_queue)

            # Wait time for this run
            wait_time = self.current_time - self.wait_since.get(pid, self.current_time)
            self.total_wait[pid] = self.total_wait.get(pid, 0) + wait_time
            total_waiting += wait_time

            # FIRST RUN DETECTION (sparse response)
            if pid not in self.first_run_time:
                self.first_run_time[pid] = self.current_time
                response_time = self.current_time - arrival
                new_response += response_time
                # Clean up dense response tracking for this process
                self.last_response_penalty_time.pop(pid, None)

            # Update last run time
            self.last_run_time[pid] = self.current_time

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
                
                # Clean up
                self.wait_since.pop(pid, None)
                self.total_wait.pop(pid, None)
                self.assigned_priority.pop(pid, None)
                self.last_run_time.pop(pid, None)
                self.last_response_penalty_time.pop(pid, None)

            elif quantum_rem == 0:
                original_priority = self.assigned_priority.get(pid, priority)
                self.wait_since[pid] = self.current_time
                heapq.heappush(
                    self.execution_queue,
                    (original_priority, pid, arrival, instr, remaining, self.time_quantum)
                )

            else:
                original_priority = self.assigned_priority.get(pid, priority)
                heapq.heappush(
                    self.execution_queue,
                    (original_priority, pid, arrival, instr, remaining, quantum_rem)
                )

        return completed, total_turnaround, new_response, total_waiting

    def render(self):
        print(f"\nTime: {self.current_time}")
        print("Queue:")
        for i, p in enumerate(self.execution_queue):
            pid = p[1]
            total_wait = self.total_wait.get(pid, 0) + (self.current_time - self.wait_since.get(pid, self.current_time))
            orig = self.assigned_priority.get(pid, p[0])
            last_run = self.last_run_time.get(pid, 0)
            time_since = self.current_time - last_run if last_run > 0 else 0
            first_run_status = "✓" if pid in self.first_run_time else "⏳"
            print(f"  Pos {i}: Priority {p[0]} (orig {orig}): PID {pid}, "
                  f"Remaining: {p[4]}/{p[3]}, Quantum: {p[5]}, "
                  f"Wait: {total_wait}, LastRun: {time_since}s ago, FirstRun: {first_run_status}")
        print(f"Completed: {len(self.completed_processes)} processes")
        if self.completed_processes:
            avg_turnaround = sum(t for _, t in self.completed_processes) / len(self.completed_processes)
            avg_waiting = sum(self.total_wait.values()) / max(1, len(self.total_wait))
            print(f"Average Turnaround: {avg_turnaround:.2f}")
            print(f"Average Waiting: {avg_waiting:.2f}")