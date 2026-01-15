
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from deepqcfn.models.task import TaskGenerator
from deepqcfn.models.network import Network, Channel
from deepqcfn.models.server import EdgeServer

class CFNEnv(gym.Env):
    def __init__(self, 
                 num_tes=10, 
                 num_ess=3, 
                 time_slots=100,
                 slot_duration=1.0):
        super().__init__()
        
        self.num_tes = num_tes
        self.num_ess = num_ess
        self.time_slots = time_slots
        self.slot_duration = slot_duration
        
        # Components
        self.task_gen = TaskGenerator()
        self.network = Network(num_nodes=num_tes + num_ess)
        self.channel = Channel(bandwidth=10e6, noise_power=1e-9) # 10MHz, Low noise
        self.servers = [EdgeServer(id=i, compute_power=2e9) for i in range(num_ess)] # 2GHz
        
        # Mapping TEs and ESs to Network Nodes
        # TEs: 0 to num_tes-1
        # ESs: num_tes to num_tes+num_ess-1
        self.es_node_indices = list(range(num_tes, num_tes + num_ess))
        
        # Action Space: Choose one of the ESs
        self.action_space = spaces.Discrete(num_ess)
        
        # Observation Space:
        # [Task_Data, Task_Priority, Task_Source_Node, 
        #  Server1_Load, Server1_Dist, ..., ServerN_Load, ServerN_Dist]
        # Size = 3 + 2 * num_ess
        low = np.array([0, 0, 0] + [0, 0] * num_ess)
        high = np.array([np.inf, np.inf, num_tes] + [np.inf, np.inf] * num_ess)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.current_time_slot = 0
        self.current_tasks_buffer = [] # Sorted tasks for current slot
        self.current_task_idx = 0
        self.current_task = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time_slot = 0
        self.task_gen.task_counter = 0
        for server in self.servers:
            server.clear()
            
        self._new_time_slot()
        return self._get_obs(), {}

    def _new_time_slot(self):
        # Generate tasks, one per TE
        raw_tasks = self.task_gen.generate_tasks(self.num_tes)
        
        # Assign Source Nodes to tasks (simple 1-to-1 mapping TE_i -> Task_i)
        for i, task in enumerate(raw_tasks):
            task.source_node = i 
            
        # PSTDQN: Sort by Priority (h_i) descending
        # Eq (12) h_i = A / D. Priority is high if h_i is high?
        # Paper says "tasks with high priority are given priority".
        # Assume Descending sort.
        self.current_tasks_buffer = sorted(raw_tasks, key=lambda t: t.priority, reverse=True)
        self.current_task_idx = 0
        self.current_task = self.current_tasks_buffer[0]

    def _get_obs(self):
        # Task Features
        task_data = self.current_task.data_volume
        task_prio = self.current_task.priority
        source_node = self.current_task.source_node
        
        obs = [task_data, task_prio, float(source_node)]
        
        # Server Features
        for i, server in enumerate(self.servers):
            server_node = self.es_node_indices[i]
            # Load = Queue Delay estimate
            load = server.get_queue_delay()
            # Distance from Task Source to this Server
            try:
                dist = self.network.graph[source_node][server_node]['distance']
            except:
                dist = 100.0
            
            obs.extend([load, dist])
            
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        target_server_idx = action
        target_server = self.servers[target_server_idx]
        target_server_node = self.es_node_indices[target_server_idx]
        task = self.current_task
        
        # Calculate Delays
        # 1. Transmission Delay (Source -> ES)
        # Assuming direct transmission or via network?
        # Paper Eq (2) t^t = A / R. (Access Delay)
        # Assuming Source is connected to SOME Access Point.
        # Simplification: Source transmits DIRECTLY to Target ES via Wireless Channel Eq(1).
        # OR: Source -> Local BS -> Backhaul -> Target ES.
        
        # Using Network Model:
        # We calculate rate between Source and Target.
        tx_power = 0.5 # Watts (27dBm)
        rate = self.network.get_transmission_rate(task.source_node, target_server_node, self.channel, tx_power)
        t_tx = self.network.get_transmission_delay(task.data_volume, rate)
        
        # 2. Queue Delay
        t_q = target_server.get_queue_delay()
        
        # 3. Computing Delay
        t_c = target_server.get_computing_delay(task)
        
        # 4. Migration Delay
        # If the task was "originally" at a "directly connected ES" and moved?
        # Paper "Transmission model": "data transmission rate of device n to a directly connected node".
        # This implies tasks go Device -> Local ES first?
        # "Task i migrates from node e_begin to e_end".
        # Let's simplify:
        # The Action DECIDES where the task is processed.
        # If Target != Local_Connected_ES, add Migration Delay.
        # For simulation, let's assume TE i is connected to ES (i % num_ess).
        local_es_idx = task.source_node % self.num_ess
        local_es_node = self.es_node_indices[local_es_idx]
        
        if local_es_idx != target_server_idx:
            t_mig = self.network.get_migration_delay(local_es_node, target_server_node, task.data_volume)
        else:
            t_mig = 0.0
            
        total_delay = t_tx + t_q + t_c + t_mig
        
        # Update Server State
        target_server.add_task(task)
        
        # Calculate Reward (Eq 11 Cost)
        # Cost = Sum(Total Delay * Priority) / Sum(Priority)
        # Since we punish step-by-step, Reward = -1 * (Delay * Priority)
        # Normalizing might be good for DQN stability.
        reward = -1.0 * total_delay * task.priority
        
        # Move to next task
        self.current_task_idx += 1
        terminated = False
        truncated = False
        
        if self.current_task_idx >= len(self.current_tasks_buffer):
            # End of Time Slot
            # Process Server Queues
            for s in self.servers:
                s.process(self.slot_duration)
                
            self.current_time_slot += 1
            if self.current_time_slot >= self.time_slots:
                terminated = True
            else:
                self._new_time_slot()
        else:
            self.current_task = self.current_tasks_buffer[self.current_task_idx]
            
        return self._get_obs(), reward, terminated, truncated, {"delay": total_delay}

