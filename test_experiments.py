
import sys
import os

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_fig3_training import train_pstdqn, train_d3qn, train_ddpg
from experiments.run_fig4_tasks import run_agent_eval, run_round_robin, run_near, CFNEnv
from experiments.run_fig5_6_size import run_experiment

def test_fig3():
    print("Testing Fig 3 logic...")
    # Run 2 episodes
    train_pstdqn(num_episodes=2)
    train_d3qn(num_episodes=2)
    train_ddpg(num_episodes=2)
    print("Fig 3 test passed.")

def test_fig4():
    print("Testing Fig 4 logic...")
    env = CFNEnv(num_tes=10, num_ess=3, time_slots=5)
    run_round_robin(env, num_episodes=2)
    run_near(env, num_episodes=2)
    run_agent_eval('PSTDQN', num_tasks=10, num_episodes=2, train_episodes=2)
    # run_agent_eval('D3QN', num_tasks=10, num_episodes=2, train_episodes=2) # Should work
    print("Fig 4 test passed.")

def test_fig5_6():
    print("Testing Fig 5/6 logic...")
    run_experiment('round-robin', [1], num_episodes=2)
    run_experiment('PSTDQN', [1], num_episodes=2)
    print("Fig 5/6 test passed.")

if __name__ == "__main__":
    test_fig3()
    test_fig4()
    test_fig5_6()
    print("All tests passed!")
