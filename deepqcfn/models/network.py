
import numpy as np
import math
import networkx as nx

class Channel:
    def __init__(self, bandwidth: float, noise_power: float):
        """
        bandwidth: B_n (Hz)
        noise_power: sigma^2 (Watts)
        """
        self.bandwidth = bandwidth
        self.noise_power = noise_power

    def calculate_rate(self, distance: float, tx_power: float) -> float:
        """
        Eq (1): R_n = B_n * log2(1 + (G_n * P_n) / sigma^2)
        Path Loss G_n = 10^(-u_n / 10)
        u_n = 127 + 30 * log10(d_n)
        d_n: distance in km? usually meters or km.
             127 + 30log(d) looks like Close-in free space reference distance path loss.
             Typically frequency dependent. 
             If d_n is in km: 128.1 + 37.6 log10(R) is typical for LTE.
             Let's assume d_n is in km for reasonable values if 30log(d).
        """
        # Avoid log(0)
        dist = max(distance, 0.001)
        
        # Path Loss Model
        u_n = 127 + 30 * np.log10(dist)
        gain_db = -u_n
        gain_linear = 10 ** (gain_db / 10.0)
        
        snr = (gain_linear * tx_power) / self.noise_power
        rate = self.bandwidth * np.log2(1 + snr)
        return rate

class Network:
    def __init__(self, num_nodes: int, migration_factor: float = 0.01):
        """
        Manages connectivity between nodes (TEs and ESs).
        num_nodes: Total nodes
        migration_factor: epsilon in Eq (8).
        """
        self.graph = nx.Graph()
        self.migration_factor = migration_factor # epsilon
        self.num_nodes = num_nodes
        
        # Setup fully connected or random topology
        # For simplicity, fully connected with weights w_{i,j}
        # Eq (6) & (7) define link loads w.
        # We assume physical links exist.
        
        # Initialize link loads (latency/cost)
        # Random initial distances/loads
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # distance in km
                dist = np.random.uniform(0.1, 5.0) 
                self.graph.add_edge(i, j, distance=dist, load=0)

    def get_transmission_rate(self, node_i, node_j, channel: Channel, tx_power: float) -> float:
        if node_i == node_j:
            return float('inf')
        
        dist = self.graph[node_i][node_j]['distance']
        return channel.calculate_rate(dist, tx_power)

    def get_transmission_delay(self, task_data: float, rate: float) -> float:
        """Eq (2): t^t = A / R"""
        if rate <= 0: return float('inf')
        return task_data / rate

    def get_migration_delay(self, source_node: int, target_node: int, task_data: float) -> float:
        """
        Eq (8): t^l = sum(epsilon * path_i * A / R)
        Here we assume migration hops.
        path_i is just 1 if direct link?
        The Eq (8) says sum over path.
        Lets calculate shortest path delay.
        """
        if source_node == target_node:
            return 0.0
            
        try:
            path = nx.shortest_path(self.graph, source_node, target_node, weight='distance')
        except nx.NetworkXNoPath:
            return float('inf')
            
        delay = 0.0
        # Iterate path segments
        # Assuming fixed migration bandwidth or using link capacities
        # Paper Eq(8) uses A_i / R_{x,x'} where R is link rate?
        # Let's assume a standard inter-node link rate for migration
        # For simplicity, use a constant large rate for backhaul or similar to channel model
        
        # NOTE: Paper mentions "link migration delay factor epsilon".
        # It simplifies to proportional to distance/hops?
        # Let's use proportional to distance * factor * data
        
        distance = nx.shortest_path_length(self.graph, source_node, target_node, weight='distance')
        # Simple interpretation: Migration delay = epsilon * distance * data
        delay = self.migration_factor * distance * task_data
        return delay
