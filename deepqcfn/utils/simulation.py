
import copy
import numpy as np
from deepqcfn.environment.cfn_env import CFNEnv

def evaluate_plan(env: CFNEnv, plan: list) -> float:
    """
    Evaluates a candidate plan (list of actions, one per task) on the current environment state.
    
    Args:
        env: The current environment instance.
        plan: List of actions (server indices), length must match env.current_tasks_buffer.
        
    Returns:
        Total Delay (Cost) for this plan.
    """
    # Clone environment to avoid side effects
    # Deepcopy is necessary because we modify server queues
    sim_env = copy.deepcopy(env)
    
    total_delay = 0.0
    
    # Iterate through plan and tasks
    # Ensure sim_env is in correct state
    # We assume 'plan' corresponds to 'sim_env.current_tasks_buffer' order (Priority order)
    
    if len(plan) != len(sim_env.current_tasks_buffer):
        # Mismatch
        return float('inf')
        
    for action in plan:
        # Step environment
        obs, reward, term, trunc, info = sim_env.step(action)
        
        # Collect delay
        # Info contains 'delay' which is the cost for that task
        total_delay += info.get('delay', 0.0)
        
    return total_delay
