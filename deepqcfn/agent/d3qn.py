
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream
        self.value_fc = nn.Linear(hidden_size, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Dueling Q: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

class D3QNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.online_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_t)
        self.online_net.train()
        
        return torch.argmax(q_values).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0
            
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).unsqueeze(1).to(self.device)
        
        # Double DQN Logic
        # 1. Select action using Online Net
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            
            # 2. Evaluate action using Target Net
            next_q_target = self.target_net(next_states)
            max_next_q = next_q_target.gather(1, next_actions)
            
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
            
        curr_q = self.online_net(states).gather(1, actions)
        
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
