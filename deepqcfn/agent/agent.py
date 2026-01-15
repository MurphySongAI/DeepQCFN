
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from deepqcfn.agent.dqn import QNetwork

class PSTDQNAgent:
    def __init__(self, state_size, action_size, 
                 lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        self.policy_net.train()
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a)
        # Gather q values at indices action
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute V(s_{t+1}) for all next states.
        # Bellman Optimality Equation: Q_target = r + gamma * max(Q_target(s', a'))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
