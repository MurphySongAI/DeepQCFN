
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size) # Continuous output for each server "score"
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) # Bound between -1 and 1
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Critic, self).__init__()
        # Q(s, a)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_size, action_size, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.memory = deque(maxlen=buffer_size)
        
        # Noise
        self.noise_std = 0.2
        
    def act(self, state, add_noise=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_continuous = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            action_continuous = np.clip(action_continuous + noise, -1.0, 1.0)
            
        return action_continuous
        
    def select_discrete_action(self, action_continuous):
        return np.argmax(action_continuous)
        
    def remember(self, state, action, reward, next_state, done):
        # Store continuous action
        self.memory.append((state, action, reward, next_state, done))
        
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0
            
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).unsqueeze(1).to(self.device)
        
        # Critic Update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q * (1 - dones))
            
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor Update
        # Maximize Q(s, actor(s)) -> Minimize -Q
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft Update Target Networks
        tau = 0.005
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
            
        return critic_loss.item()
