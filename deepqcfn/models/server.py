
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
        # Algorithm II: Update C_e for C_e - C_{i,e}
        # We track usage.
        self.current_cycle_load += task.compute_required
    
    def can_accommodate(self, task: Task, slot_duration: float) -> bool:
        """
        Checks if the server has enough resources (cycles) in the current slot 
        to handle the task, based on Eq (15a) and Algorithm II.
        
        Max Resources C_e = compute_power * slot_duration.
        Required C_{i,e} = task.compute_required.
        """
        max_cycles = self.compute_power * slot_duration
        if self.current_cycle_load + task.compute_required <= max_cycles:
            return True
        return False
        
    def process(self, time_step: float):
        """Simulates processing of tasks in queue for a duration of time_step."""
        # Simple processing logic: First In First Out
        capacity = self.compute_power * time_step
        
        while self.queue and capacity > 0:
            task = self.queue[0]
            # Reduce task volume
            # Convert cycles to data for processing tracking? 
            # Or just track cycles.
            # Existing code used data_volume.
            # let's assume processing rate consumed data.
            
            # Using data volume based processing to match legacy logic
            # but ensuring we respect consistency.
            processed_data = min(task.data_volume, capacity * (task.data_volume / task.compute_required))
            # Wait, easier:
            # cycles_available = capacity (since compute_power is cycles/sec)
            # cycles_needed = task.compute_required
            
            # Let's stick to the existing data-centric logic for 'process' 
            # but assume compute_power is strictly consistent with cycles.
            # For simulation step, we just clear the queue if we assume slot-based allocation?
            # If we are using Algorithm II (Resource Allocation), we determine WHO is in.
            # Then they are processed.
            
            # If we assume the slot is long enough for the admitted tasks (which is the constraint),
            # then all admitted tasks should finish?
            # "C <= C_max". C_max = Power * Time. 
            # So yes, if admitted, they finish.
            
            self.queue.popleft() 
            # In a strict Discrete Event Sim we might carry over, 
            # but Eq 15a implies Slot-based framing.
            
            # Legacy code maintained.
            
    def get_computing_delay(self, task: Task) -> float:
        """
        Eq (4): t^c_{i,e} = A_{i,e} / f_e  (Using Data Vol)
        OR t = C_{i,e} / f_e (Using Cycles)
        Paper defines f_e as GHz.
        """
        if self.compute_power <= 0: return float('inf')
        return task.compute_required / self.compute_power    

    def get_queue_delay(self) -> float:
        """
        Eq (5): t^q_{i,e} = A_{n,e} / f_e
        Time to process all tasks currently in queue.
        """
        total_cycles = sum([t.compute_required for t in self.queue])
        if self.compute_power <= 0: return 0.0
        return total_cycles / self.compute_power

    def clear(self):
        self.queue.clear()
        self.current_cycle_load = 0.0

    current_cycle_load: float = 0.0

