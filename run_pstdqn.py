
import numpy as np
import matplotlib.pyplot as plt
import torch
from deepqcfn.environment.cfn_env import CFNEnv
from deepqcfn.agent.pstdqn import PSTDQNAgent
from deepqcfn.utils.simulation import evaluate_plan

def get_global_state(env):
    """
    Constructs a global state representation for the Generator inputs.
    Concatenates Task Features (for all tasks in buffer) and Server Features.
    """
    # Tasks are in env.current_tasks_buffer
    # Each task: [Data, Priority, SourceNode]
    task_feats = []
    for task in env.current_tasks_buffer:
        task_feats.extend([task.data_volume, task.priority, float(task.source_node)])
        
    # Server Features: [Load, Dist_to_Base?]
    # Distances are relative to tasks, which is complex for global state.
    # Let's use Server Queue Load.
    server_feats = []
    for s in env.servers:
        server_feats.append(s.get_queue_delay())
        
    # Pad if necessary (if N varies, but here we assume fixed N for MLP)
    return np.array(task_feats + server_feats, dtype=np.float32)

def run_experiment(num_episodes=50):
    # Parameters from Table I (Approximation)
    num_tes = 10 
    num_ess = 3
    time_slots = 50 # Episodes length
    
    env = CFNEnv(num_tes=num_tes, num_ess=num_ess, time_slots=time_slots)
    
    # State Size Calculation
    # Task Feat (3) * num_tes + Server Feat (1) * num_ess
    state_size = 3 * num_tes + 1 * num_ess
    
    agent = PSTDQNAgent(state_size=state_size, num_tasks=num_tes, num_servers=num_ess)
    
    rewards_history = []
    loss_history = []
    
    print(f"Starting PSTDQN Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        while not terminated:
            # 1. Construct Global State
            # Note: env.current_tasks_buffer contains the tasks for THIS slot
            state = get_global_state(env)
            
            # 2. Generator -> Initial Plan X (Action Vector)
            initial_plan = agent.generate_plan(state)
            
            # 3. Evolutionary Candidates (Mutation/Crossover)
            # PSTDQN loop
            candidates = agent.evolve_candidates(initial_plan, k=10)
            
            # 4. Evaluate Candidates
            best_plan = None
            best_cost = float('inf')
            
            for plan in candidates:
                cost = evaluate_plan(env, plan)
                # Cost is Delay (lower is better)
                # Paper metric: Delay.
                if cost < best_cost:
                    best_cost = cost
                    best_plan = plan
            
            # 5. Execute Best Plan on Real Env
            # Loop through tasks and apply actions
            slot_reward = 0
            for action in best_plan:
                # env.step handles one task from buffer and moves to next
                _, r, term, _, info = env.step(action)
                slot_reward += r
                if term:
                    terminated = True
                    
            episode_reward += slot_reward
            steps += 1
            
            # 6. Store Experience and Train
            # Store (State, Best_Plan)
            # Note: Storing "Best Plan" effectively makes this Imitation Learning (Supervised)
            # The "Ground Truth" is the evolved solution.
            agent.remember(state, best_plan, best_cost)
            
            # Train
            loss = agent.train_step()
            episode_loss += loss
            
        avg_reward = episode_reward / steps
        avg_loss = episode_loss / steps
        rewards_history.append(avg_reward)
        loss_history.append(avg_loss)
        
        print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Best Slot Cost: {best_cost:.2f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Average Reward per Slot")
    plt.xlabel("Episode")
    plt.ylabel("Reward (-Delay)")
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, color='orange')
    plt.title("Generator Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.savefig('pstdqn_training_results.png')
    print("Results saved to pstdqn_training_results.png")

if __name__ == "__main__":
    run_experiment()
