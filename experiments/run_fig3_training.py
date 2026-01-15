
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent directory to path to allow imports
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

def train_pstdqn(num_episodes=200):
    print("Training PSTDQN...")
    env = CFNEnv(num_tes=10, num_ess=3, time_slots=50)
    state_size = 3 * 10 + 1 * 3
    agent = PSTDQNAgent(state_size=state_size, num_tasks=10, num_servers=3)
    
    costs = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        total_cost = 0
        steps = 0
        
        while not terminated:
            state = get_global_state(env)
            
            # PSTDQN Logic
            initial_plan = agent.generate_plan(state)
            candidates = agent.evolve_candidates(initial_plan, k=10)
            
            best_plan = None
            best_cost_val = float('inf')
            
            for plan in candidates:
                c = evaluate_plan(env, plan)
                if c < best_cost_val:
                    best_cost_val = c
                    best_plan = plan
            
            # Execute
            slot_cost = 0
            for action in best_plan:
                _, r, term, _, info = env.step(action)
                # Cost is Delay. Reward = -Delay * Priority. 
                # Paper Fig 3 Y-axis is "Cost". We can assume Cost = Total Delay (or Weighted Delay).
                # Let's use Total Delay for now.
                slot_cost += info['delay']
                if term: terminated = True
            
            total_cost += slot_cost
            steps += 1
            
            agent.remember(state, best_plan, best_cost_val)
            agent.train_step()
            
        avg_cost = total_cost / steps
        costs.append(avg_cost)
        if (episode+1) % 10 == 0:
            print(f"PSTDQN Episode {episode+1}: Cost {avg_cost:.2f}")
            
    return costs

def train_d3qn(num_episodes=200):
    print("Training D3QN...")
    env = CFNEnv(num_tes=10, num_ess=3, time_slots=50)
    # State for D3QN is per-task observation: [Data, Prio, Src, ServerFeats...]
    # Size = 3 + 2 * num_ess = 3 + 6 = 9
    state_size = 3 + 2 * 3
    action_size = 3
    agent = D3QNAgent(state_size, action_size, lr=1e-3)
    
    costs = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        total_cost = 0
        steps = 0 # Count slots? Or tasks?
        # We need Cost per "Round" (Time Slot usually, or Episode?)
        # Fig 3 X-axis is "Training Rounds". Usually 1 Round = 1 Episode if not specified.
        
        slot_costs = [] # To average per slot if needed, or just sum for episode
        
        # We want metric comparable to PSTDQN. 
        # PSTDQN metric calculated above was Avg Cost PER SLOT (total_cost / steps).
        
        current_slot_cost = 0
        tasks_in_slot = 0
        
        completed_slots = 0
        
        while not terminated:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.remember(obs, action, reward, next_obs, terminated)
            agent.train_step()
            
            obs = next_obs
            current_slot_cost += info['delay']
            tasks_in_slot += 1
            
            # Check if slot changed (trickier with step-based env)
            # Actually env.current_task_idx resets to 0 when slot changes
            # But we don't easily know when slot changes from outside unless we check env state
            # A simple heuristic: if env.current_task_idx == 0 and we just stepped, we might have started new slot?
            # Better: accumulate for the whole episode and divide by number of slots (50)
            
        # Total Cost of Episode
        # Average Cost per Slot = Total Episode Cost / 50
        avg_cost = current_slot_cost / 50.0 
        costs.append(avg_cost)
        
        agent.update_target_network()
        
        if (episode+1) % 10 == 0:
            print(f"D3QN Episode {episode+1}: Cost {avg_cost:.2f}")
            
    return costs

def train_ddpg(num_episodes=200):
    print("Training DDPG...")
    env = CFNEnv(num_tes=10, num_ess=3, time_slots=50)
    state_size = 3 + 2 * 3 # 9
    action_size = 3
    agent = DDPGAgent(state_size, action_size)
    
    costs = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        current_slot_cost = 0
        
        while not terminated:
            # Action is continuous vector
            action_cont = agent.act(obs)
            # Discrete limit
            action = agent.select_discrete_action(action_cont)
            
            next_obs, reward, terminated, _, info = env.step(action)
            
            # Store continuous action in memory
            # Note: next_obs is valid for next step
            agent.remember(obs, action_cont, reward, next_obs, terminated)
            agent.train_step()
            
            obs = next_obs
            current_slot_cost += info['delay']
            
        avg_cost = current_slot_cost / 50.0
        costs.append(avg_cost)
        
        if (episode+1) % 10 == 0:
            print(f"DDPG Episode {episode+1}: Cost {avg_cost:.2f}")
            
    return costs

if __name__ == "__main__":
    # Run experiments
    # PSTDQN
    pstdqn_costs = train_pstdqn(num_episodes=200)
    
    # D3QN
    d3qn_costs = train_d3qn(num_episodes=200)
    
    # DDPG
    ddpg_costs = train_ddpg(num_episodes=200)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(pstdqn_costs, label='PSTDQN', color='orange')
    plt.plot(d3qn_costs, label='D3QN', color='purple')
    plt.plot(ddpg_costs, label='DDPG', color='green')
    
    plt.xlabel('Number of Training Rounds')
    plt.ylabel('Cost')
    plt.title('Algorithm Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'experiments/fig3_training.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
