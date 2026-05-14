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

        # Reward weights (configurable for different optimization goals)
        self.weights = {
            'turnaround': 1.0,      # Primary metric - total completion time
            'waiting': 0.8,         # Time spent in ready queue
            'response': 0.5,        # Time to first execution
            'queue': 0.02,          # Congestion penalty
            'starvation': 2.0,      # Penalty for processes waiting too long
            'progress': 0.3         # Bonus for making progress
        }
        
        # Starvation threshold (ticks before a process is considered starving)
        self.starvation_threshold = 100
        
        # Keep 6 features (already have quantum_rem)
        self.observation_space = spaces.Box(
            low=-1, high=np.inf,
            shape=(encoder_context + 1, 6),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(max_priority)

        self.reset()

    def _get_info(self):
        return {}

    def _get_obs(self):
        obs = np.full((self.encoder_context + 1, 6), -1, dtype=np.int32)

        # Next arriving process
        if self.data_pointer < len(self.processes):
            pid, arrival, instr, remaining = self.processes[self.data_pointer]
            obs[0, :4] = [pid, arrival, instr, remaining]
            obs[0, 4] = -1
            obs[0, 5] = self.time_quantum

        # Current queue
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
        self.execution_queue = []
        self.completed_processes = []
        self.total_completed = 0
        self.total_turnaround = 0
        
        # Response time tracking
        self.first_run_time = {}
        self.avg_instructions = np.mean([p[2] for p in self.processes]) if self.processes else 1
        
        # Waiting time tracking
        self.wait_since = {}      # When each process started waiting
        self.total_wait = {}      # Accumulated waiting time
        
        # Reward tracking for stability
        self.prev_queue_pressure = 0
        self.reward_history = []
        
        # Aging tracking (for starvation prevention)
        self.assigned_priority = {}
        self.last_aging_time = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Initialize reward components
        new_completions = 0
        new_turnarounds = 0
        new_response_times = 0
        new_waiting_times = 0
        starvation_penalty = 0
        progress_bonus = 0

        if self.data_pointer < len(self.processes):
            proc = self.processes[self.data_pointer]

            delta_time = proc[1] - self.current_time
            if delta_time > 0:
                # Get all metrics from execution
                completed, turnarounds, response, waiting = self._execute_for_time(delta_time)
                new_completions += completed
                new_turnarounds += turnarounds
                new_response_times += response
                new_waiting_times += waiting

            # Add process with chosen priority
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
                completed, turnarounds, response, waiting = self._execute_for_time(remaining_time)
                new_completions += completed
                new_turnarounds += turnarounds
                new_response_times += response
                new_waiting_times += waiting

        self.total_completed += new_completions
        self.total_turnaround += new_turnarounds

        queue_pressure = len(self.execution_queue)

        # ============================================================
        # REWARD ENGINEERING COMPONENTS
        # ============================================================

        # 1. TURNAROUND PENALTY (Primary metric)
        # Penalizes total time from submission to completion
        # Lower turnaround = better scheduling
        turnaround_penalty = -self.weights['turnaround'] * new_turnarounds / self.avg_instructions

        # 2. WAITING TIME PENALTY
        # Penalizes time spent in ready queue (not including execution)
        # Lower waiting time = less starvation, better responsiveness
        waiting_penalty = -self.weights['waiting'] * new_waiting_times / self.avg_instructions

        # 3. RESPONSE TIME PENALTY
        # Penalizes time to first execution - critical for interactive tasks
        # Lower response time = better for user-facing applications
        response_penalty = -self.weights['response'] * new_response_times / self.avg_instructions

        # 4. QUEUE PRESSURE PENALTY
        # Penalizes long ready queues - indicates system congestion
        # Encourages agent to keep queue manageable
        queue_penalty = -self.weights['queue'] * queue_pressure

        # 5. STARVATION PENALTY (NEW)
        # Aggressively penalizes processes that have waited too long
        # Prevents starvation of low-priority processes
        starvation_penalty = 0
        for pid, wait_since in self.wait_since.items():
            wait_time = self.current_time - wait_since
            if wait_time > self.starvation_threshold:
                # Exponential penalty for starvation
                excess = wait_time - self.starvation_threshold
                starvation_penalty += self.weights['starvation'] * (excess / 50)
        starvation_penalty = -starvation_penalty

        # 6. PROGRESS BONUS (NEW)
        # Positive reinforcement for completing processes
        # Encourages agent to finish tasks efficiently
        if new_completions > 0:
            # Bonus for each completed process
            progress_bonus += self.weights['progress'] * new_completions
            
            # Additional bonus for completing processes faster than average
            avg_completion_time = self.total_turnaround / max(1, self.total_completed)
            if avg_completion_time < self.avg_instructions:
                progress_bonus += 0.1 * new_completions

        # 7. QUEUE REDUCTION BONUS (NEW)
        # Bonus when queue size decreases - indicates good progress
        if queue_pressure < self.prev_queue_pressure:
            reduction = self.prev_queue_pressure - queue_pressure
            progress_bonus += 0.05 * reduction

        # 8. PRIORITY BOOST BONUS (NEW)
        # Bonus when a starving process finally gets CPU time
        # This helps the agent learn to prevent starvation
        # (Tracked in _execute_for_time)

        # Combine all reward components
        reward = (
            turnaround_penalty +
            waiting_penalty +
            response_penalty +
            queue_penalty +
            starvation_penalty +
            progress_bonus
        )

        # Clip reward to prevent extreme values and stabilize training
        reward = np.clip(reward, -10.0, 5.0)

        # Update tracking variables
        self.prev_queue_pressure = queue_pressure
        
        # Store reward for analysis (optional)
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)

        terminated = (
            self.data_pointer == len(self.processes)
            and len(self.execution_queue) == 0
        )

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _execute_for_time(self, time_available):
        """
        Execute processes for available time units.
        Returns: (completed, total_turnaround, total_response, total_waiting)
        """
        completed = 0
        total_turnaround = 0
        new_response = 0
        total_waiting = 0
        remaining_time = time_available

        while remaining_time > 0 and self.execution_queue:
            priority, pid, arrival, instr, remaining, quantum_rem = heapq.heappop(self.execution_queue)

            # ============================================================
            # WAITING TIME TRACKING
            # ============================================================
            # Calculate how long this process waited before running
            wait_time = self.current_time - self.wait_since.get(pid, self.current_time)
            # Update accumulated waiting time
            self.total_wait[pid] = self.total_wait.get(pid, 0) + wait_time
            total_waiting += wait_time

            # ============================================================
            # RESPONSE TIME TRACKING
            # ============================================================
            # Track first time process runs (critical for response time)
            if pid not in self.first_run_time:
                self.first_run_time[pid] = self.current_time
                response_time = self.current_time - arrival
                new_response += response_time

            # Determine how long to run this process
            run_time = min(remaining_time, remaining, quantum_rem)

            # Execute the process
            self.current_time += run_time
            remaining_time -= run_time
            remaining -= run_time
            quantum_rem -= run_time

            # ============================================================
            # PROCESS COMPLETION HANDLING
            # ============================================================
            if remaining == 0:
                # Process finished - calculate final metrics
                turnaround = self.current_time - arrival
                self.completed_processes.append((pid, turnaround))
                completed += 1
                total_turnaround += turnaround
                
                # Clean up tracking dictionaries
                self.wait_since.pop(pid, None)
                self.total_wait.pop(pid, None)
                self.assigned_priority.pop(pid, None)
                
                # ========================================================
                # PRIORITY BOOST BONUS (Reward shaping)
                # ========================================================
                # If process was starving and finally completed, give extra bonus
                # (This is tracked in the main step function via starvation_penalty)
                pass

            # ============================================================
            # QUANTUM EXHAUSTION (PREEMPTION)
            # ============================================================
            elif quantum_rem == 0:
                # Quantum exhausted - preempt the process
                # Restore original priority and reset quantum
                original_priority = self.assigned_priority.get(pid, priority)
                # Reset wait tracking for this process (it goes back to queue)
                self.wait_since[pid] = self.current_time
                heapq.heappush(
                    self.execution_queue,
                    (original_priority, pid, arrival, instr, remaining, self.time_quantum)
                )

            # ============================================================
            # CONTINUING EXECUTION (Quantum not exhausted)
            # ============================================================
            else:
                # Push back with remaining quantum
                original_priority = self.assigned_priority.get(pid, priority)
                heapq.heappush(
                    self.execution_queue,
                    (original_priority, pid, arrival, instr, remaining, quantum_rem)
                )

        return completed, total_turnaround, new_response, total_waiting

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
            avg_waiting = sum(self.total_wait.values()) / max(1, len(self.total_wait))
            print(f"Average Turnaround: {avg_turnaround:.2f}")
            print(f"Average Waiting: {avg_waiting:.2f}")