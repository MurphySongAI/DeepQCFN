
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from collections import deque

class Generator(nn.Module):
    def __init__(self, input_size, num_tasks, num_servers, hidden_size=128):
        super(Generator, self).__init__()
        self.num_tasks = num_tasks
        self.num_servers = num_servers
        
        # Simple MLP Generator
        # Input: State representation
        # Output: Probabilities for each task assignment (num_tasks * num_servers)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_tasks * num_servers)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape to (Batch, Tasks, Servers)
        x = x.view(-1, self.num_tasks, self.num_servers)
        return x

class PSTDQNAgent:
    def __init__(self, 
                 state_size, 
                 num_tasks, 
                 num_servers,
                 lr=1e-3, 
                 gamma=0.99,
                 buffer_size=128): # Unusually small buffer size mentioned in paper (128)
        
        self.state_size = state_size
        self.num_tasks = num_tasks
        self.num_servers = num_servers
        self.lr = lr
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = Generator(state_size, num_tasks, num_servers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Experience Buffer
        # Paper says: "Maximum capacity... 128" (Very small!)
        # "When experience exceeds 64, data samples randomly selected to train"
        self.memory = deque(maxlen=buffer_size)
        
    def generate_plan(self, state):
        """
        Uses Neural Network to generate an initial plan X.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state_t)
        self.model.train()
        
        # Greedy selection from Probabilities
        probs = torch.softmax(logits, dim=2)
        # Shape (1, Tasks, Servers)
        actions = torch.argmax(probs, dim=2).cpu().numpy()[0]
        return actions

    def evolve_candidates(self, base_plan, k=10, mutation_prob=0.2):
        """
        Generates k candidate solutions using Crossover and Mutation.
        """
        candidates = []
        candidates.append(base_plan.copy()) # Keep original
        
        for _ in range(k-1):
            candidate = base_plan.copy()
            
            # Mutation: Randomly change assignment for some tasks
            if np.random.rand() < mutation_prob:
                # Select random task to mutate
                idx = np.random.randint(0, self.num_tasks)
                candidate[idx] = np.random.randint(0, self.num_servers)
                
            # Crossover: (Self-crossover with random plan? or between candidates?)
            # Paper says "interchanging some bits...".
            # Let's interact with a Random Plan
            if np.random.rand() < 0.5:
                random_plan = np.random.randint(0, self.num_servers, size=self.num_tasks)
                mask = np.random.rand(self.num_tasks) < 0.5
                candidate[mask] = random_plan[mask]
            
            candidates.append(candidate)
            
        return candidates

    def remember(self, state, best_plan, delay):
        self.memory.append((state, best_plan, delay))

    def train_step(self):
        if len(self.memory) < 64:
            return 0.0
        
        # Random sample
        batch_size = 32
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        target_plans = torch.LongTensor(np.array([t[1] for t in batch])).to(self.device)
        
        # Forward
        logits = self.model(states)
        # Logits: (Batch, Tasks, Servers)
        # Targets: (Batch, Tasks)
        
        # Flatten for CrossEntropy
        loss = self.criterion(logits.view(-1, self.num_servers), target_plans.view(-1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

