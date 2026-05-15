"""
Live demo comparing schedulers side-by-side with Gantt charts
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def plot_gantt(gantt_data, scheduler_name, num_processes=50, save_path=None):
    """Plot Gantt chart for a scheduler"""
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Get unique process IDs
    unique_pids = sorted(set([p for p in gantt_data if p != -1]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pids)))
    color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_pids)}
    
    time = 0
    current_start = 0
    current_pid = gantt_data[0] if gantt_data else -1
    
    for i, pid in enumerate(gantt_data):
        if pid != current_pid:
            # Draw segment for previous process
            if current_pid != -1:
                ax.barh(current_pid, i - current_start, left=current_start, 
                       height=0.8, color=color_map[current_pid], edgecolor='black', linewidth=0.5)
            current_start = i
            current_pid = pid
    
    # Draw last segment
    if current_pid != -1:
        ax.barh(current_pid, len(gantt_data) - current_start, left=current_start,
               height=0.8, color=color_map[current_pid], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Process ID', fontsize=12)
    ax.set_title(f'{scheduler_name} Scheduler - Gantt Chart (First {len(gantt_data)} steps)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Gantt chart saved to: {save_path}")
    
    plt.show()
    return fig

def live_demo_with_gantt(dataset, num_processes=50, show_gantt=False):
    """Run live demo with optional Gantt chart visualization"""
    
    from schedulers.fifo import FIFO
    from schedulers.round_robin import RoundRobin
    from schedulers.cfs import CFS
    from schedulers.mlq import MLQ
    from schedulers.ml_prio import MLPriority
    from schedulers.dpo_prio import DPOPriority
    from schedulers.dqn_prio import DQNPriority
    
    # Take subset for faster demo
    subset = dataset[:num_processes]
    
    schedulers = {
        'FIFO': (FIFO, {}),
        'RoundRobin': (RoundRobin, {'time_quantum': 4}),
        'CFS': (CFS, {}),
        'PPO': (MLPriority, {
            'encoder_context': 30,
            'max_priority': 10,
            'model_path': 'model_weights/ml_priority_scheduler_5mil_30context.pt'
        }),
        'DPO': (DPOPriority, {
            'encoder_context': 30,
            'max_priority': 10,
            'model_path': 'model_weights/dpo_scheduler.pt'
        }),
        'DQN': (DQNPriority, {
            'encoder_context': 30,
            'max_priority': 10,
            'model_path': 'model_weights/dqn_scheduler.pt'
        })
    }
    
    results = []
    gantt_data = {}
    
    print("\n" + "="*80)
    print("LIVE SCHEDULER DEMO (with Gantt)")
    print(f"Dataset: {num_processes} processes")
    print("="*80)
    
    for name, (scheduler_class, kwargs) in schedulers.items():
        print(f"\n🔄 Running {name}...")
        start = time.time()
        
        try:
            sched = scheduler_class(subset.copy(), **kwargs)  # Use copy to preserve data
            sched.time_run()
            sched.calc_stats()
            
            elapsed = time.time() - start
            
            # Store gantt chart data (first 200 steps for visualization)
            gantt_data[name] = sched.gantt[:200] if hasattr(sched, 'gantt') else []
            
            results.append({
                'Scheduler': name,
                'Turnaround': sched.stat_turnaround_time,
                'Waiting': sched.stat_waiting_time,
                'Response': sched.stat_response_time,
                'CPU Util': sched.stat_cpu_util * 100,
                'Runtime': elapsed,
                'Gantt Length': len(sched.gantt)
            })
            
            print(f"  ✅ Completed in {elapsed:.2f}s")
            print(f"     Turnaround: {sched.stat_turnaround_time:.2f}")
            print(f"     Waiting: {sched.stat_waiting_time:.2f}")
            print(f"     Response: {sched.stat_response_time:.2f}")
            print(f"     Gantt steps: {len(sched.gantt)}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'Scheduler': name,
                'Turnaround': float('inf'),
                'Waiting': float('inf'),
                'Response': float('inf'),
                'CPU Util': 0,
                'Runtime': 0,
                'Gantt Length': 0
            })
            gantt_data[name] = []
    
    # Display results table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    table_data = []
    for r in results:
        table_data.append([
            r['Scheduler'],
            f"{r['Turnaround']:.2f}",
            f"{r['Waiting']:.2f}",
            f"{r['Response']:.2f}",
            f"{r['CPU Util']:.1f}%",
            f"{r['Runtime']:.2f}s",
            r['Gantt Length']
        ])
    
    headers = ['Scheduler', 'Turnaround', 'Waiting', 'Response', 'CPU Util', 'Runtime', 'Gantt Steps']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Find best performers
    valid_results = [r for r in results if r['Turnaround'] != float('inf')]
    if valid_results:
        best_turnaround = min(valid_results, key=lambda x: x['Turnaround'])
        best_waiting = min(valid_results, key=lambda x: x['Waiting'])
        best_response = min(valid_results, key=lambda x: x['Response'])
        
        print(f"\n🏆 Best Turnaround: {best_turnaround['Scheduler']} ({best_turnaround['Turnaround']:.2f})")
        print(f"🏆 Best Waiting: {best_waiting['Scheduler']} ({best_waiting['Waiting']:.2f})")
        print(f"🏆 Best Response: {best_response['Scheduler']} ({best_response['Response']:.2f})")
    
    # Show Gantt charts if requested
    if show_gantt:
        print("\n" + "="*80)
        print("GANTT CHARTS")
        print("="*80)
        
        # Show best scheduler's Gantt chart
        if best_turnaround and gantt_data.get(best_turnaround['Scheduler']):
            print(f"\n📊 Gantt Chart for Best Scheduler: {best_turnaround['Scheduler']}")
            plot_gantt(gantt_data[best_turnaround['Scheduler']], 
                      best_turnaround['Scheduler'], 
                      num_processes,
                      save_path=f"gantt_{best_turnaround['Scheduler']}.png")
        
        # Optionally show all Gantt charts
        show_all = input("\nShow Gantt charts for all schedulers? (y/n): ")
        if show_all.lower() == 'y':
            for name in schedulers.keys():
                if gantt_data.get(name):
                    print(f"\n📊 Gantt Chart for {name}")
                    plot_gantt(gantt_data[name], name, num_processes,
                              save_path=f"gantt_{name}.png")
    
    return results, gantt_data


# Run demo with Gantt
if __name__ == "__main__":
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    dataset_path = "./dataset/dataset1.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    dataset = np.genfromtxt(dataset_path, delimiter=',', skip_header=1)
    print(f"Loaded {len(dataset)} processes")
    
    # Run demo with Gantt charts
    results, gantt_data = live_demo_with_gantt(dataset, num_processes=50, show_gantt=True)