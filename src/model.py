import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class BedAllocationNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 82)  # One output per bed
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQLAgent:
    def __init__(self, state_size, 
                 memory_size=10000,
                 batch_size=64,
                 gamma=0.99,
                 epsilon=1,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=1e-3):
        
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BedAllocationNetwork(state_size).to(self.device)
        self.target_model = BedAllocationNetwork(state_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def get_state(self, beds, waiting_patients, current_time):
        # Construct state vector: [bed_occupancy, patient_severity, wait_times, time_features]
        bed_state = np.zeros(82)  # 0 for empty, 1 for occupied
        severity_state = np.zeros(82)  # severity of patient in each bed (0 if empty)
        
        for i, patient in enumerate(beds):
            if patient is not None:
                bed_state[i] = 1
                severity_state[i] = patient.severity
        
        # Add waiting patient info and time features
        waiting_severities = [p.severity for p in waiting_patients][:5]  # Consider up to 5 waiting patients
        waiting_severities.extend([0] * (5 - len(waiting_severities)))  # Pad to fixed length
        
        time_features = [
            np.sin(2 * np.pi * current_time / 24),  # Time of day
            np.cos(2 * np.pi * current_time / 24)
        ]
        
        state = np.concatenate([
            bed_state,
            severity_state,
            waiting_severities,
            time_features
        ])
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            q_values = self.model(state).cpu().numpy()[0]
            # Mask invalid actions with large negative values
            masked_q_values = np.full(82, float('-inf'))
            masked_q_values[valid_actions] = q_values[valid_actions]
            return np.argmax(masked_q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch]).to(self.device)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(self.device)
        next_states = torch.cat([x[3] for x in batch])
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())