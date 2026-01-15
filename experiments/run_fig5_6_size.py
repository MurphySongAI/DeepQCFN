
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepqcfn.environment.cfn_env import CFNEnv
from deepqcfn.agent.pstdqn import PSTDQNAgent
from deepqcfn.agent.d3qn import D3QNAgent
from deepqcfn.agent.ddpg import DDPGAgent
from deepqcfn.utils.simulation import evaluate_plan

# Utility to get global state
def get_global_state(env):
    task_feats = []
    for task in env.current_tasks_buffer:
        task_feats.extend([task.data_volume, task.priority, float(task.source_node)])
    server_feats = []
    for s in env.servers:
        server_feats.append(s.get_queue_delay())
    return np.array(task_feats + server_feats, dtype=np.float32)

def run_experiment(agent_type, size_range, num_episodes=20):
    cost_results = []
    delay_results = []
    
    for size in size_range:
        print(f"Running {agent_type} for Task Size {size} MB...")
        env = CFNEnv(num_tes=10, num_ess=3, time_slots=20)
        # Force task size
        env.task_gen.data_volume_range = (size, size)
        
        # Initialize Agent
        if agent_type == 'PSTDQN':
            state_size = 3 * 10 + 3
            agent = PSTDQNAgent(state_size, 10, 3)
            # Brief Train
            for _ in range(10):
                env.reset()
                term = False
                while not term:
                    state = get_global_state(env)
                    plan = agent.generate_plan(state)
                    candidates = agent.evolve_candidates(plan)
                    best_plan = min(candidates, key=lambda p: evaluate_plan(env, p))
                    agent.remember(state, best_plan, evaluate_plan(env, best_plan))
                    agent.train_step()
                    # Execute
                    for action in best_plan:
                        env.step(action)
                        if env.current_task_idx >= len(env.current_tasks_buffer): break
                    if env.current_time_slot >= env.time_slots: term = True
                    else: env._new_time_slot()
                    
        elif agent_type == 'D3QN':
            state_size = 9
            agent = D3QNAgent(state_size, 3)
            for _ in range(10):
                obs, _ = env.reset()
                term = False
                while not term:
                    action = agent.act(obs)
                    n_obs, r, term, _, _ = env.step(action)
                    agent.remember(obs, action, r, n_obs, term)
                    agent.train_step()
                    obs = n_obs
                    
        elif agent_type == 'DDPG':
            state_size = 9
            agent = DDPGAgent(state_size, 3)
            for _ in range(10):
                obs, _ = env.reset()
                term = False
                while not term:
                    ac = agent.act(obs)
                    a = agent.select_discrete_action(ac)
                    n_obs, r, term, _, _ = env.step(a)
                    agent.remember(obs, ac, r, n_obs, term)
                    agent.train_step()
                    obs = n_obs

        # Evaluate
        total_delay_sum = 0 # For Cost
        count_tasks = 0
        
        for _ in range(num_episodes):
            env.reset()
            obs = env._get_obs()
            term = False
            
            while not term:
                if agent_type == 'round-robin':
                    # We need to maintain idx? 
                    # Assuming random or simple modulo? 
                    # Let's use simple random for RR equivalent or strict RR
                    # Strict RR across tasks
                    action = count_tasks % 3
                elif agent_type == 'Near':
                    task = env.current_task
                    # Nearest Server logic
                    best_server_idx = 0
                    min_dist = float('inf')
                    for i in range(3):
                        server_node = env.es_node_indices[i]
                        try:
                            dist = env.network.graph[task.source_node][server_node]['distance']
                        except: dist = 100.0
                        if dist < min_dist:
                            min_dist = dist
                            best_server_idx = i
                    action = best_server_idx
                elif agent_type == 'PSTDQN':
                    # Need to handle slot-based vs task-based
                    # Here we are inside step loop
                    # But PSTDQN generates plan at start of slot
                    if env.current_task_idx == 0:
                        s = get_global_state(env)
                        plan = agent.generate_plan(s)
                        candidates = agent.evolve_candidates(plan)
                        current_plan = min(candidates, key=lambda p: evaluate_plan(env, p))
                    action = current_plan[env.current_task_idx]
                elif agent_type == 'D3QN':
                    action = agent.act(obs)
                elif agent_type == 'DDPG':
                    ac = agent.act(obs, add_noise=False)
                    action = agent.select_discrete_action(ac)
                
                _, _, term, _, info = env.step(action)
                obs = env._get_obs()
                
                total_delay_sum += info['delay']
                count_tasks += 1
                
        # Cost = Average Total Delay per Episode? Or Sum?
        # If Cost is ~4000 and we have 10 tasks * 20 slots * 20 episodes... way too high.
        # Cost ~4000 likely means sum of delays for ONE SET of tasks (e.g. 10 tasks) * X?
        # Or Delay ~400 (per task). 10 tasks = 4000.
        # So Cost = Sum of delays per Slot (10 tasks).
        
        avg_delay_per_task = total_delay_sum / count_tasks if count_tasks > 0 else 0
        avg_cost_per_slot = avg_delay_per_task * 10
        
        cost_results.append(avg_cost_per_slot)
        delay_results.append(avg_delay_per_task)
        
    return cost_results, delay_results

if __name__ == "__main__":
    sizes = np.arange(1, 11, 1) # 1 to 10
    
    agents = ['round-robin', 'Near', 'DDPG', 'D3QN', 'PSTDQN']
    results = {}
    
    for ag in agents:
        c, d = run_experiment(ag, sizes, num_episodes=5) # Reduced episodes for speed
        results[ag] = {'cost': c, 'delay': d}
        
    # Fig 5: Cost
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, results['round-robin']['cost'], 'r-o', label='round-robin')
    plt.plot(sizes, results['Near']['cost'], 'b-s', label='Near')
    plt.plot(sizes, results['DDPG']['cost'], 'g-x', label='DDPG')
    plt.plot(sizes, results['D3QN']['cost'], 'm-^', label='D3QN')
    plt.plot(sizes, results['PSTDQN']['cost'], 'y-v', label='PSTDQN')
    
    plt.xlabel('Task size/MB')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/fig5_cost_size.png')
    print("Saved experiments/fig5_cost_size.png")
    
    # Fig 6: Delay
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, results['round-robin']['delay'], 'r-o', label='round-robin')
    plt.plot(sizes, results['Near']['delay'], 'b-s', label='Near')
    plt.plot(sizes, results['DDPG']['delay'], 'g-x', label='DDPG')
    plt.plot(sizes, results['D3QN']['delay'], 'm-^', label='D3QN')
    plt.plot(sizes, results['PSTDQN']['delay'], 'y-v', label='PSTDQN')
    
    plt.xlabel('Task Size/MB')
    plt.ylabel('Delay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/fig6_delay_size.png')
    print("Saved experiments/fig6_delay_size.png")
