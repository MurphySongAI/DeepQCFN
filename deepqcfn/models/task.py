
import numpy as np
from dataclasses import dataclass

@dataclass
class Task:
    id: int
    data_volume: float        # A_k (Megabits or similar unit)
    compute_required: float   # C_k (Megacycles or similar unit)
    max_delay: float          # D_k (Seconds)
    
    # Priority h_i calculated later
    priority: float = 0.0

    def calculate_priority(self):
        """
        Calculates priority h_i based on Eq (12): h_i = A_i / D_i
        Note: The paper allows defining h_i differently, but Eq(12) is explicit.
        """
        if self.max_delay > 0:
            self.priority = self.data_volume / self.max_delay
        else:
            self.priority = float('inf') # Urgent if delay is 0

class TaskGenerator:
    def __init__(self, 
                 data_volume_range=(1, 10), 
                 compute_factor_range=(0.5, 2.0),
                 max_delay_range=(0.1, 1.0)):
        """
        Generates tasks with random properties.
        data_volume: Uniformly distributed in range.
        compute_required: data_volume * Factor (cycles per bit).
        max_delay: Uniformly distributed.
        """
        self.data_volume_range = data_volume_range
        self.compute_factor_range = compute_factor_range
        self.max_delay_range = max_delay_range
        self.task_counter = 0

    def generate_task(self) -> Task:
        """Generates a single task."""
        data_vol = np.random.uniform(*self.data_volume_range)
        # Assuming compute required is proportional to data volume with some variance
        # C_k typically dependent on task type, here modeled as a factor of data size.
        compute_factor = np.random.uniform(*self.compute_factor_range)
        compute_req = data_vol * compute_factor
        
        max_delay = np.random.uniform(*self.max_delay_range)
        
        task = Task(
            id=self.task_counter,
            data_volume=data_vol,
            compute_required=compute_req,
            max_delay=max_delay
        )
        task.calculate_priority()
        self.task_counter += 1
        return task

    def generate_tasks(self, num_tasks: int):
        return [self.generate_task() for _ in range(num_tasks)]
