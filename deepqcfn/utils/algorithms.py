
from typing import List, Tuple
import numpy as np
from deepqcfn.models.task import Task

def priority_sort(tasks: List[Task]) -> List[Task]:
    """
    Algorithm I: Task Priority Algorithm.
    
    The paper describes a randomized sorting algorithm (like QuickSort partition) using h_i.
    However, the goal is explicitly stated as:
    "ensuring that the order in which tasks are processed is consistent with the criticality and urgency of the task."
    
    The most robust way to achieve this consistency is a stable sort descending by priority.
    We will use Python's Timsort (sorted) which is stable and efficient.
    
    Args:
        tasks: List of Task objects.
        
    Returns:
        Sorted list of tasks (Highest Priority First).
    """
    # Eq (12) h_i = A / D (Priority)
    # We assume 'priority' attribute is already calculated on Task.
    return sorted(tasks, key=lambda t: t.priority, reverse=True)

def resource_allocation(tasks: List[Task], edge_server_capacity: float) -> Tuple[List[float], List[float]]:
    """
    Algorithm II: Resource Allocation Algorithm.
    
    Allocates computational resources (C_e) to tasks based on priority.
    
    Args:
        tasks: List of Task objects (Assumed sorted by priority).
        edge_server_capacity: Total compute power (f_e) available at the server.
                              NOTE: Paper uses C_e as "maximum computational resources".
                              Paper Eq (15a) sum(C_{i,e}) <= C_e.
                              Here C_{i,e} is the generic resource allocated.
        
    Returns:
        allocated_resources: List of allocated resources for each task (C_{i,e}).
        delays: List of computing delays (t^c) for each task.
    """
    
    # 1. Output Q (Priority Queue) - Assumed input 'tasks' is already Q.
    # 2. q = number of tasks
    
    allocated_resources = []
    computing_delays = []
    
    # Remaining capacity
    current_capacity = edge_server_capacity # C_e
    
    # 3. for i=1:q
    for task in tasks:
        # Task requirement: C_{i,e}.
        # Wait, the paper says:
        # 4: if (C_e > C_{i,e})
        # 5:   update C_e for C_e - C_{i,e}
        # 6:   Z[i] = 1
        # 
        # But what is C_{i,e}? Is it the *required* resource or the *allocated*?
        # Typically, a task *needs* C_req cycles to complete. 
        # If we allocate f_{i,e} (cycles/sec), then Time = C_req / f_{i,e}.
        #
        # Eq (4) t^c = A_{i,e} / f_e. 
        # This implies the COMPLETE server power f_e is used for the task if processed sequentially?
        #
        # Re-reading Algorithm II: 
        # It seems to be checking if there is *enough* resource to accept the task?
        # Or is it allocating a chunk of capacity?
        #
        # Let's look at Eq (15a): sum(C_{i,e}) <= C_e.
        # This usually implies purely resource partitioning (like RAM or specialized cores), OR
        # bandwidth partitioning.
        #
        # If the server processes tasks *sequentially* (FIFO/Priority Queue), 
        # then "Resource Allocation" essentially means "Admission Control" or "Time Slotting".
        #
        # However, line 6 "Z[i]=1" implies a binary decision (Accept/Reject?).
        #
        # Let's interpret Algorithm II as:
        # Allocating a specific resource "C" (e.g. CPU cycles) to ensure the task finishes within some deadline?
        # 
        # Alternative interpretation:
        # The paper might be doing "Multi-tasking" where C_e is total CPU freq, and we split it among tasks.
        # But Eq (4) uses t^c = A / f_e (Total power?). This suggests Sequential processing.
        # 
        # IF sequential:
        # Then resource constraint is Time? sum(t^c) <= TimeSlot?
        #
        # Let's assume Algorithm II is "Admission Control based on Capacity".
        # Check if Task Requirement <= Remaining Capacity.
        #
        # Task.compute_required (Cycles).
        # We need to allocate enough cycles.
        #
        # Let's implement specific logic:
        # Iterate tasks. If server has capacity to handle it, accept it (Z=1).
        # Otherwise, reject or handle differently (Z=0).
        
        # NOTE: Paper variable names:
        # C_e: Max computational resources.
        # C_{i,e}: Resources for task i (Usually requirement).
        
        req_res = task.compute_required
        
        if current_capacity >= req_res:
            current_capacity -= req_res
            allocated_resources.append(req_res)
            # Completed fully
            # Delay depends on how fast it runs. 
            # If we carve out static resource: Time = Req / Allocated. 
            # If Allocated == Req (Capacity chunks), this implies "Space" allocation models.
            # But let's assume standard delay model t = data / rate.
            # We'll return the fraction allocated.
        else:
            # Cannot allocate fully.
            # Rejection? Or partial? Algo says "else... end". 
            # Implies Z[i]=0 (not in list).
            allocated_resources.append(0.0)
            
    return allocated_resources

