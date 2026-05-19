import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def plot_gantt(gantt_data, scheduler_name, num_processes=50, save_path=None, gantt_priority=None):
    """Plot Gantt chart for a scheduler, with optional priority annotations"""
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Get unique process IDs
    unique_pids = sorted(set([p for p in gantt_data if p != -1]))
    if not unique_pids:
        print(f"  No valid process data for {scheduler_name}")
        return None
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pids)))
    color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_pids)}
    
    # Build a priority-per-pid lookup for coloring
    has_priority = gantt_priority is not None and len(gantt_priority) == len(gantt_data)
    if has_priority:
        pid_priority = {}
        for pid, pri in zip(gantt_data, gantt_priority):
            if pid != -1 and pri != -1:
                pid_priority[pid] = pri
    
    current_start = 0
    current_pid = gantt_data[0] if gantt_data else -1
    current_pri = gantt_priority[0] if has_priority else None
    
    for i, pid in enumerate(gantt_data):
        pri_at_i = gantt_priority[i] if has_priority else None
        if pid != current_pid or (has_priority and pri_at_i != current_pri):
            if current_pid != -1 and current_pid in color_map:
                bar = ax.barh(current_pid, i - current_start, left=current_start,
                             height=0.8, color=color_map[current_pid],
                             edgecolor='black', linewidth=0.5)
                if has_priority and current_pri is not None and current_pri != -1:
                    mid = current_start + (i - current_start) / 2
                    ax.text(mid, current_pid, f'P{int(current_pri)}',
                           ha='center', va='center', fontsize=7, fontweight='bold',
                           color='white' if current_pri > 5 else 'black')
            current_start = i
            current_pid = pid
            if has_priority:
                current_pri = pri_at_i
    
    if current_pid != -1 and current_pid in color_map:
        bar = ax.barh(current_pid, len(gantt_data) - current_start, left=current_start,
                     height=0.8, color=color_map[current_pid],
                     edgecolor='black', linewidth=0.5)
        if has_priority and current_pri is not None and current_pri != -1:
            mid = current_start + (len(gantt_data) - current_start) / 2
            ax.text(mid, current_pid, f'P{int(current_pri)}',
                   ha='center', va='center', fontsize=7, fontweight='bold',
                   color='white' if current_pri > 5 else 'black')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Process ID', fontsize=12)
    title = f'{scheduler_name} Scheduler - Gantt Chart'
    if has_priority:
        title += ' (with Priority Annotations)'
    title += f' (First {len(gantt_data)} steps)'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Gantt chart saved to: {save_path}")
    
    plt.show()
    return fig

def live_demo_with_gantt(dataset, num_processes=50, show_gantt=False):
    """Run live demo with optional Gantt chart visualization"""
    
    from schedulers.fifo import FIFO
    from schedulers.round_robin import RoundRobin
    from schedulers.cfs import CFS
    from schedulers.mlq import MLQ
    from schedulers.mfq import MFQ
    from schedulers.ml_prio import MLPriority
    from schedulers.dpo_prio import DPOPriority
    from schedulers.dqn_prio import DQNPriority
    
    # Take subset for faster demo - create a fresh copy
    subset = np.copy(dataset[:num_processes])  # Use np.copy instead of .copy()
    
    # Define schedulers with correct parameters for updated environment
    schedulers = {
        'FIFO': (FIFO, {}),
        'RoundRobin': (RoundRobin, {'time_quantum': 4}),
        'CFS': (CFS, {}),
        'MLQ': (MLQ, {}),  # Added MLQ
        'MFQ': (MFQ, {}),  # Added MFQ
        'PPO': (MLPriority, {
            'encoder_context': 30,
            'max_priority': 10,
            'time_quantum': 4,  # Add for updated env
            'model_path': 'model_weights/ppo_trained_model.pt'
        }),
        'DPO': (DPOPriority, {
            'encoder_context': 30,
            'max_priority': 10,
            'time_quantum': 4,  # Add for updated env
            'model_path': 'model_weights/dpo_trained_model.pt'
        }),
        'DQN': (DQNPriority, {
            'encoder_context': 30,
            'max_priority': 10,
            'time_quantum': 4,  # Add for updated env
            'model_path': 'model_weights/dqn_trained_model.pt'
        })
    }
    
    results = []
    gantt_data = {}
    gantt_priority_data = {}
    
    print("\n" + "="*80)
    print("LIVE SCHEDULER DEMO (with Gantt)")
    print(f"Dataset: {num_processes} processes")
    print("="*80)
    
    for name, (scheduler_class, kwargs) in schedulers.items():
        print(f"\n🔄 Running {name}...")
        start = time.time()
        
        try:
            # Create a fresh copy for each scheduler
            data_copy = np.copy(subset)
            sched = scheduler_class(data_copy, **kwargs)
            sched.time_run()
            sched.calc_stats()
            
            elapsed = time.time() - start
            
            # Store gantt chart data (first 200 steps)
            gantt_data[name] = sched.gantt[:200] if hasattr(sched, 'gantt') else []
            gantt_priority_data[name] = (sched.gantt_priority[:200] 
                                         if hasattr(sched, 'gantt_priority') else None)
            
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
            
        except FileNotFoundError as e:
            print(f"  ❌ Model file not found: {e}")
            print(f"     Please train {name} model first or check model path")
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
            gantt_priority_data[name] = None
            
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
            gantt_priority_data[name] = None
    
    # Display results table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    table_data = []
    for r in results:
        if r['Turnaround'] != float('inf'):
            table_data.append([
                r['Scheduler'],
                f"{r['Turnaround']:.2f}",
                f"{r['Waiting']:.2f}",
                f"{r['Response']:.2f}",
                f"{r['CPU Util']:.1f}%",
                f"{r['Runtime']:.2f}s",
                r['Gantt Length']
            ])
        else:
            table_data.append([
                r['Scheduler'],
                "ERROR",
                "ERROR",
                "ERROR",
                "N/A",
                "ERROR",
                0
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
    if show_gantt and valid_results:
        print("\n" + "="*80)
        print("GANTT CHARTS")
        print("="*80)
        
        # Show best scheduler's Gantt chart
        if best_turnaround and gantt_data.get(best_turnaround['Scheduler']):
            print(f"\n📊 Gantt Chart for Best Scheduler: {best_turnaround['Scheduler']}")
            plot_gantt(gantt_data[best_turnaround['Scheduler']], 
                      best_turnaround['Scheduler'], 
                      num_processes,
                      save_path=f"gantt_{best_turnaround['Scheduler']}.png",
                      gantt_priority=gantt_priority_data.get(best_turnaround['Scheduler']))
        
        # Optionally show all Gantt charts
        try:
            show_all = input("\nShow Gantt charts for all schedulers? (y/n): ")
            if show_all.lower() == 'y':
                for name in schedulers.keys():
                    if gantt_data.get(name) and name != best_turnaround['Scheduler']:
                        print(f"\n📊 Gantt Chart for {name}")
                        plot_gantt(gantt_data[name], name, num_processes,
                                  save_path=f"gantt_{name}.png",
                                  gantt_priority=gantt_priority_data.get(name))
        except (EOFError, KeyboardInterrupt):
            print("\n  Skipping additional Gantt charts...")
    
    return results, gantt_data


# Run demo with Gantt
if __name__ == "__main__":
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    dataset_path = "./dataset/test/dataset_challenging_500.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    dataset = np.genfromtxt(dataset_path, delimiter=',', skip_header=1)
    print(f"Loaded {len(dataset)} processes")
    
    # Check if model files exist before running
    model_paths = [
        'model_weights/ppo_trained_model.pt',
        'model_weights/dpo_trained_model.pt',
        'model_weights/dqn_trained_model.pt'
    ]
    
    print("\nChecking model files:")
    for path in model_paths:
        if os.path.exists(path):
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} NOT FOUND - train this model first")
    
    # Run demo with Gantt charts
    results, gantt_data = live_demo_with_gantt(dataset, num_processes=50, show_gantt=True)