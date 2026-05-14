from abc import ABC, abstractmethod
import numpy as np
import time

class Scheduler(ABC):
    def __init__(self, data):
        self.gantt = []
        self.data = data
        if data is not None:
            self.pids = data[:,0].astype(int)
            self.arrivals = data[:,1]
            self.instr_count = data[:,2]
            # Build PID→data lookup (avoids fragile PID-as-index assumption)
            self.pid_info = {}
            for i, pid in enumerate(self.pids):
                self.pid_info[int(pid)] = (float(self.arrivals[i]), int(self.instr_count[i]))

    def cpu_util(self):
        "Ranges from 0-1, can be converted to percentage"
        cpu_utilization = (len(self.gantt) - self.gantt.count(-1)) / float(len(self.gantt))
        return cpu_utilization

    def throughput(self):
        """Number of processes completed per time unit"""
        return len(self.pids) / len(self.gantt)

    def turnaround_time(self):
        """Average turnaround time - time from arrival to completion"""
        turnaround_times = []
        for pid in self.pids:
            arrival, _ = self.pid_info[int(pid)]
            last_index = len(self.gantt) - 1 - self.gantt[::-1].index(pid)
            turnaround_times.append((last_index + 1) - arrival)
        return sum(turnaround_times) / len(turnaround_times)

    def waiting_time(self):
        """Average waiting time - turnaround time minus execution time"""
        waiting_times = []
        for pid in self.pids:
            arrival, instr = self.pid_info[int(pid)]
            last_index = len(self.gantt) - 1 - self.gantt[::-1].index(pid)
            completion_time = last_index + 1
            waiting_times.append((completion_time - arrival) - instr)
        return sum(waiting_times) / len(waiting_times)

    def response_time(self):
        "Average response time - amount of time it takes from when a request was submitted until the first response is produced"
        response_times = []
        for pid in self.pids:
            arrival, _ = self.pid_info[int(pid)]
            first_index = self.gantt.index(pid)
            response_times.append(first_index - arrival)
        return sum(response_times) / len(response_times)
    
    def time_run(self):
        "Overhead - amount of time (real) to complete running a particular dataset"
        start_time = time.time()
        self.run()
        stop_time = time.time()
        self.stat_runtime = stop_time - start_time

    def calc_stats(self):
        self.stat_cpu_util = self.cpu_util()
        self.stat_throughput = self.throughput()
        self.stat_turnaround_time = self.turnaround_time()
        self.stat_response_time = self.response_time()
        self.stat_waiting_time = self.waiting_time()
        self.stat_mean_runtime = self.stat_runtime / len(self.pids)

    def print_stats(self):
        print("CPU Utilization :",self.stat_cpu_util)
        print("Throughput      :",self.stat_throughput)
        print("Turnaround Time :",self.stat_turnaround_time)
        print("Waiting Time    :",self.stat_waiting_time)
        print("Response Time   :",self.stat_response_time)
        print("Runtime         :",self.stat_runtime)
        print("Mean Runtime    :",self.stat_mean_runtime)

    @abstractmethod
    def run():
        pass

    