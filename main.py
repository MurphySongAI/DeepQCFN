
import os
import numpy as np
import matplotlib.pyplot as plt
from deepqcfn.environment.cfn_env import CFNEnv
from deepqcfn.agent.agent import PSTDQNAgent

def main():
    # Hyperparameters
    EPISODES = 50 # Keep small for demo/testing
    TIME_SLOTS = 20
    NUM_TES = 10
    NUM_ESS = 3
    BATCH_SIZE = 32
    
    # Initialize Environment
    env = CFNEnv(num_tes=NUM_TES, num_ess=NUM_ESS, time_slots=TIME_SLOTS)
    
    # Initialize Agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PSTDQNAgent(state_size, action_size, batch_size=BATCH_SIZE)
    
    # Logging
    scores = []
    avg_delays = []
    losses = []
    
    print("Starting Training...")
    
    for e in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        total_delay = 0
        total_tasks = 0
        
        done = False
        while not done:
            # Action
            action = agent.act(state)
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Remember
            agent.remember(state, action, reward, next_state, done)
            
            # Learn
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
                
            state = next_state
            total_reward += reward
            total_delay += info['delay']
            total_tasks += 1
            
        # End of Episode
        agent.update_target_network()
        
        avg_delay = total_delay / total_tasks if total_tasks > 0 else 0
        scores.append(total_reward)
        avg_delays.append(avg_delay)
        
        print(f"Episode {e+1}/{EPISODES} | Reward: {total_reward:.2f} | Avg Delay: {avg_delay:.4f} | Epsilon: {agent.epsilon:.2f}")

    # Plotting
    plot_results(scores, avg_delays, losses)

def plot_results(scores, avg_delays, losses):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(avg_delays)
    plt.title('Average Task Delay per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Delay (s)')
    
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    output_path = 'training_results.png'
    plt.savefig(output_path)
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    main()
