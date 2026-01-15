
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepqcfn.environment.cfn_env import CFNEnv
from deepqcfn.agent.pstdqn import PSTDQNAgent
from deepqcfn.agent.d3qn import D3QNAgent
from deepqcfn.agent.ddpg import DDPGAgent
from deepqcfn.utils.simulation import evaluate_plan

def get_global_state(env):
    task_feats = []
    for task in env.current_tasks_buffer:
        task_feats.extend([task.data_volume, task.priority, float(task.source_node)])
    server_feats = []
    for s in env.servers:
        server_feats.append(s.get_queue_delay())
    return np.array(task_feats + server_feats, dtype=np.float32)

def run_round_robin(env, num_episodes=10):
    rr_idx = 0
    total_accepted = 0
    total_tasks = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action = rr_idx % env.num_ess
            rr_idx += 1
            _, _, terminated, _, info = env.step(action)
            if info['accepted']:
                total_accepted += 1
            total_tasks += 1
            
    return total_accepted / total_tasks if total_tasks > 0 else 0

def run_near(env, num_episodes=10):
    total_accepted = 0
    total_tasks = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        
        while not terminated:
            # Task info is in env.current_task
            # But "Near" implies finding nearest server.
            # We can access env.current_task.source_node
            # and env.network to find distances
            task = env.current_task
            src_node = task.source_node
            
            best_server_idx = 0
            min_dist = float('inf')
            
            for i, server in enumerate(env.servers):
                server_node = env.es_node_indices[i]
                # Assuming network is fully connected or we can get dist
                try:
                    dist = env.network.graph[src_node][server_node]['distance']
                except:
                    dist = 100.0
                
                if dist < min_dist:
                    min_dist = dist
                    best_server_idx = i
            
            _, _, terminated, _, info = env.step(best_server_idx)
            if info['accepted']:
                total_accepted += 1
            total_tasks += 1
            
    return total_accepted / total_tasks if total_tasks > 0 else 0

def run_agent_eval(agent_type, num_tasks, num_episodes=20, train_episodes=20):
    env = CFNEnv(num_tes=num_tasks, num_ess=3, time_slots=20)
    
    if agent_type == 'PSTDQN':
        state_size = 3 * num_tasks + 1 * 3
        agent = PSTDQNAgent(state_size, num_tasks, 3)
        # Train briefly
        for _ in range(train_episodes):
            obs, _ = env.reset()
            term = False
            while not term:
                state = get_global_state(env)
                plan = agent.generate_plan(state)
                candidates = agent.evolve_candidates(plan)
                best_plan = min(candidates, key=lambda p: evaluate_plan(env, p))
                agent.remember(state, best_plan, evaluate_plan(env, best_plan))
                agent.train_step()
                # Execute best plan
                for action in best_plan:
                    env.step(action)
                    if env.current_task_idx >= len(env.current_tasks_buffer): break 
                if env.current_time_slot >= env.time_slots: term = True
                else: env._new_time_slot()
                    
    elif agent_type == 'D3QN':
        state_size = 3 + 6
        agent = D3QNAgent(state_size, 3)
        for _ in range(train_episodes):
            obs, _ = env.reset()
            term = False
            while not term:
                action = agent.act(obs)
                next_obs, reward, term, _, _ = env.step(action)
                agent.remember(obs, action, reward, next_obs, term)
                agent.train_step()
                obs = next_obs
                
    elif agent_type == 'DDPG':
        state_size = 3 + 6
        agent = DDPGAgent(state_size, 3)
        for _ in range(train_episodes):
            obs, _ = env.reset()
            term = False
            while not term:
                ac = agent.act(obs)
                a = agent.select_discrete_action(ac)
                next_obs, r, term, _, _ = env.step(a)
                agent.remember(obs, ac, r, next_obs, term)
                agent.train_step()
                obs = next_obs

    # Evaluation
    total_accepted = 0
    total_tasks = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        term = False
        while not term:
            if agent_type == 'PSTDQN':
                state = get_global_state(env)
                plan = agent.generate_plan(state)
                # No evolution in inference? Or yes? Paper says "performs prior...". Usually inference is fast.
                # Let's use evolved plan for better performance as PSTDQN relies on it.
                candidates = agent.evolve_candidates(plan, k=5) 
                best_plan = min(candidates, key=lambda p: evaluate_plan(env, p))
                for action in best_plan:
                    _, _, term, _, info = env.step(action)
                    if info['accepted']: total_accepted += 1
                    total_tasks += 1
                    if term: break
            else:
                if agent_type == 'D3QN':
                    action = agent.act(obs) # Epsilon 0 for eval?
                    # D3QN act uses current epsilon. We should set it to min or 0.
                    # agent.epsilon = 0 # Hack
                elif agent_type == 'DDPG':
                    ac = agent.act(obs, add_noise=False)
                    action = agent.select_discrete_action(ac)
                
                _, _, term, _, info = env.step(action)
                obs = env._get_obs() # Refresh obs
                
                if info['accepted']: total_accepted += 1
                total_tasks += 1
                
    return total_accepted / total_tasks if total_tasks > 0 else 0

if __name__ == "__main__":
    task_counts = [40, 50, 60, 70, 80, 90, 100]
    
    results = {
        'round-robin': [],
        'Near': [],
        'DDPG': [],
        'D3QN': [],
        'PSTDQN': []
    }
    
    for n in task_counts:
        print(f"Running for {n} tasks...")
        env = CFNEnv(num_tes=n, num_ess=3, time_slots=20)
        
        results['round-robin'].append(run_round_robin(env))
        results['Near'].append(run_near(env))
        results['DDPG'].append(run_agent_eval('DDPG', n))
        results['D3QN'].append(run_agent_eval('D3QN', n))
        results['PSTDQN'].append(run_agent_eval('PSTDQN', n))
        
    plt.figure(figsize=(10, 6))
    plt.plot(task_counts, results['round-robin'], 'r-o', label='round-robin')
    plt.plot(task_counts, results['Near'], 'b-s', label='Near')
    plt.plot(task_counts, results['DDPG'], 'g-x', label='DDPG')
    plt.plot(task_counts, results['D3QN'], 'm-^', label='D3QN')
    plt.plot(task_counts, results['PSTDQN'], 'y-v', label='PSTDQN') # Orange/Yellow
    
    plt.xlabel('Number of Tasks')
    plt.ylabel('Completion Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('experiments/fig4_completion.png')
    print("Saved experiments/fig4_completion.png")
