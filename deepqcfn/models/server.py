
import collections
from deepqcfn.models.task import Task

class EdgeServer:
    def __init__(self, id: int, compute_power: float):
        """
        id: Server ID
        compute_power: f_e (e.g., Gigacycles per second)
        """
        self.id = id
        self.compute_power = compute_power
        self.queue = collections.deque() # Stores Tasks
        self.current_load = 0.0 # Sum of A_{n,e} (Queue size in data bits? Or compute cycles?) 
                                # Eq (5) uses A_{n,e} / f_e. 
                                # A_{n,e} is "original set of tasks on ES e". 
                                # Usually queue delay depends on remaining compute workload.
                                # Let's assume A_{n,e} represents accumulated compute load or data load.
                                # Eq (5) implies A is size. Let's track expected compute time.

    def add_task(self, task: Task):
        self.queue.append(task)
    
    def process(self, time_step: float):
        """Simulates processing of tasks in queue for a duration of time_step."""
        # Simple processing logic: First In First Out
        capacity = self.compute_power * time_step
        
        while self.queue and capacity > 0:
            task = self.queue[0]
            # How much compute this task needs:
            # We assume C_k used for processing.
            # If the paper implies A_i is processed directly, then use A_i.
            # Eq (4) t^c = A_{i,e} / f_e. It uses A (data) not C (compute).
            # This implies processing time is proportional to data volume directly.
            # Let's use data_volume as the workload metric to align with Eq (4).
            
            # Reduce task volume
            processed = min(task.data_volume, capacity) # Treating f_e as bits/sec or similar
            task.data_volume -= processed
            capacity -= processed
            
            if task.data_volume <= 0:
                self.queue.popleft() # Task Completed

    def get_computing_delay(self, task: Task) -> float:
        """
        Eq (4): t^c_{i,e} = A_{i,e} / f_e
        Time to process THIS specific task.
        """
        return task.data_volume / self.compute_power

    def get_queue_delay(self) -> float:
        """
        Eq (5): t^q_{i,e} = A_{n,e} / f_e
        Time to process all tasks currently in queue.
        A_{n,e} is sum of data volumes of tasks in queue.
        """
        total_queue_volume = sum([t.data_volume for t in self.queue])
        return total_queue_volume / self.compute_power

    def clear(self):
        self.queue.clear()
