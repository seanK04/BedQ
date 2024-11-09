import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Tuple
from patient import Patient

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """Neural network for approximating Q-values"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(QNetwork, self).__init__()
        
        # Neural network architecture:
        # Layer 1: State size -> Hidden size with ReLU activation
        # Layer 2: Hidden size -> Hidden size with ReLU activation
        # Layer 3: Hidden size -> Action size (Q-values for each action)
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Random sample of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class BedAllocationDQL:
    """Deep Q-Learning agent for hospital bed allocation"""
    def __init__(self,
                 num_beds: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 target_update: int = 10):
        
        # Environment parameters
        self.num_beds = num_beds
        self.state_size = self._calculate_state_size()
        self.action_size = num_beds + 1  # +1 for "wait" action
        
        # DQL parameters
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        
        # Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer()
        
    def _calculate_state_size(self) -> int:
        """Calculate the size of the state vector"""
        # State includes:
        # - Binary occupancy for each bed
        # - Severity score for each occupied bed
        # - Remaining stay duration for each occupied bed
        # - Waiting patient's severity (if any)
        # - Number of waiting patients
        return self.num_beds * 3 + 2
        
    def _get_state(self, beds: List[Patient], waiting: List[Patient]) -> np.ndarray:
        """Convert current hospital state to network input vector"""
        state = np.zeros(self.state_size)
        
        # Fill bed information
        for i, patient in enumerate(beds):
            if patient:
                state[i] = 1  # bed occupied
                state[i + self.num_beds] = patient.severity / 10  # normalized severity
                state[i + 2 * self.num_beds] = patient.stay_duration  # remaining stay
        
        # Add waiting patient information
        if waiting:
            state[-2] = waiting[0].severity / 10  # first waiting patient's severity
            state[-1] = len(waiting)  # number of waiting patients
            
        return state
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def update(self, experience: Experience) -> float:
        """Update the Q-network using a single experience"""
        self.memory.push(experience)
        
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample experiences
        experiences = self.memory.sample(self.batch_size)
        
        # Prepare batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Get current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
    def calculate_reward(self, 
                        old_state: np.ndarray,
                        action: int,
                        new_state: np.ndarray,
                        waiting_time: float,
                        severity: float) -> float:
        """Calculate reward for the taken action"""
        # Base reward is negative waiting time * severity
        reward = -waiting_time * severity
        
        # Additional rewards/penalties based on bed utilization
        old_utilization = np.sum(old_state[:self.num_beds])
        new_utilization = np.sum(new_state[:self.num_beds])
        
        # Reward efficient bed utilization
        reward += (new_utilization - old_utilization) * 10
        
        # Penalize unnecessary waiting when beds are available
        if action == self.num_beds and new_utilization < self.num_beds:
            reward -= 50
            
        return reward
        
    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        
    def load(self, path: str) -> None:
        """Load model weights"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

def main():
    """Test model initialization and basic operations"""
    model = BedAllocationDQL(num_beds=100)
    state = np.random.rand(model.state_size)
    action = model.select_action(state)
    print(f"Selected action: {action}")
    
if __name__ == "__main__":
    main()