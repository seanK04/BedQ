import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

from patient import Patient, PatientGenerator
from model import BedAllocationDQL, Experience

@dataclass
class SimulationMetrics:
    """Tracks metrics during simulation"""
    episode_rewards: List[float] = None
    waiting_times: List[float] = None
    bed_utilization: List[float] = None
    loss_values: List[float] = None
    
    def __post_init__(self):
        self.episode_rewards = []
        self.waiting_times = []
        self.bed_utilization = []
        self.loss_values = []
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode rewards
        axes[0,0].plot(self.episode_rewards)
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        
        # Plot average waiting times
        axes[0,1].plot(self.waiting_times)
        axes[0,1].set_title('Average Waiting Times')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Hours')
        
        # Plot bed utilization
        axes[1,0].plot(self.bed_utilization)
        axes[1,0].set_title('Bed Utilization')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Percentage')
        
        # Plot loss values
        axes[1,1].plot(self.loss_values)
        axes[1,1].set_title('Training Loss')
        axes[1,1].set_xlabel('Update Step')
        axes[1,1].set_ylabel('Loss')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class HospitalEnvironment:
    """Simulation environment for hospital bed allocation"""
    def __init__(self, 
                 num_beds: int = 82,
                 simulation_duration: float = 24*7,  # one week
                 patient_generator: Optional[PatientGenerator] = None):
        
        self.num_beds = num_beds
        self.duration = simulation_duration
        self.generator = patient_generator or PatientGenerator()
        
        # Initialize environment
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_time = 0
        self.beds = [None] * self.num_beds
        self.waiting_patients = deque()
        self.discharged_patients = []
        
        # Generate initial patients
        self.future_patients = deque(
            self.generator.generate_batch(self.duration)
        )
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Convert current hospital state to numpy array"""
        state = np.zeros(self.num_beds * 3 + 2)
        
        # Fill bed information
        for i, patient in enumerate(self.beds):
            if patient:
                state[i] = 1  # occupancy
                state[i + self.num_beds] = patient.severity / 10  # normalized severity
                remaining_stay = max(0, patient.stay_duration - 
                                  (self.current_time - patient.arrival_time))
                state[i + 2 * self.num_beds] = remaining_stay / (24 * 7)  # normalized remaining stay
        
        # Fill waiting information
        if self.waiting_patients:
            state[-2] = self.waiting_patients[0].severity / 10
            state[-1] = len(self.waiting_patients) / 10  # normalized queue length
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one simulation step"""
        # Process action
        reward = 0
        old_state = self._get_state()
        
        if action < self.num_beds and self.waiting_patients:
            # Assign patient to bed
            if self.beds[action] is None:
                patient = self.waiting_patients.popleft()
                self.beds[action] = patient
                patient.bed_id = action
                waiting_time = self.current_time - patient.arrival_time
                reward = -waiting_time * patient.severity
            else:
                reward = -100  # Penalty for trying to assign to occupied bed
        
        # Advance time to next event
        next_event_time = float('inf')
        
        # Check next patient arrival
        if self.future_patients:
            next_event_time = min(next_event_time, 
                                self.future_patients[0].arrival_time)
        
        # Check next discharge
        for patient in self.beds:
            if patient:
                discharge_time = patient.arrival_time + patient.stay_duration
                if discharge_time > self.current_time:
                    next_event_time = min(next_event_time, discharge_time)
        
        if next_event_time == float('inf'):
            return self._get_state(), reward, True
        
        # Advance to next event
        self.current_time = next_event_time
        
        # Process discharges
        for i, patient in enumerate(self.beds):
            if patient and (patient.arrival_time + patient.stay_duration <= self.current_time):
                self.discharged_patients.append(patient)
                self.beds[i] = None
        
        # Process arrivals
        while (self.future_patients and 
               self.future_patients[0].arrival_time <= self.current_time):
            patient = self.future_patients.popleft()
            self.waiting_patients.append(patient)
            patient.waiting_start = self.current_time
        
        done = self.current_time >= self.duration
        return self._get_state(), reward, done

def train_model(env: HospitalEnvironment,
                model: BedAllocationDQL,
                num_episodes: int = 1000,
                max_steps_per_episode: int = 1000) -> SimulationMetrics:
    """Train the DQL model"""
    metrics = SimulationMetrics()
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        waiting_times = []
        bed_utilization = []
        
        for step in range(max_steps_per_episode):
            # Select action
            action = model.select_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store experience
            experience = Experience(state, action, reward, next_state, done)
            loss = model.update(experience)
            
            # Update metrics
            episode_reward += reward
            waiting_times.append(len(env.waiting_patients))
            bed_utilization.append(
                sum(1 for bed in env.beds if bed is not None) / env.num_beds
            )
            if loss is not None:
                metrics.loss_values.append(loss)
            
            state = next_state
            
            if done:
                break
        
        # Record episode metrics
        metrics.episode_rewards.append(episode_reward)
        metrics.waiting_times.append(np.mean(waiting_times))
        metrics.bed_utilization.append(np.mean(bed_utilization))
        
        # Log progress
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Average Reward: {episode_reward:.2f}")
            print(f"Average Waiting Time: {np.mean(waiting_times):.2f}")
            print(f"Average Bed Utilization: {np.mean(bed_utilization)*100:.1f}%")
    
    return metrics

def evaluate_model(env: HospitalEnvironment,
                  model: BedAllocationDQL,
                  num_episodes: int = 100) -> SimulationMetrics:
    """Evaluate the trained model"""
    metrics = SimulationMetrics()
    model.epsilon = 0  # No exploration during evaluation
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0
        waiting_times = []
        bed_utilization = []
        
        while True:
            action = model.select_action(state)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            waiting_times.append(len(env.waiting_patients))
            bed_utilization.append(
                sum(1 for bed in env.beds if bed is not None) / env.num_beds
            )
            
            if done:
                break
                
            state = next_state
        
        metrics.episode_rewards.append(episode_reward)
        metrics.waiting_times.append(np.mean(waiting_times))
        metrics.bed_utilization.append(np.mean(bed_utilization))
    
    return metrics

def main():
    """Train and evaluate the model"""
    # Initialize environment and model
    env = HospitalEnvironment(num_beds=82)
    model = BedAllocationDQL(num_beds=82)
    
    # Train model
    print("Starting training...")
    training_metrics = train_model(env, model)
    training_metrics.plot_metrics("training_metrics.png")
    
    # Save trained model
    model.save("trained_model.pth")
    
    # Evaluate model
    print("\nStarting evaluation...")
    eval_metrics = evaluate_model(env, model)
    eval_metrics.plot_metrics("evaluation_metrics.png")
    
    # Print final metrics
    print("\nFinal Evaluation Metrics:")
    print(f"Average Episode Reward: {np.mean(eval_metrics.episode_rewards):.2f}")
    print(f"Average Waiting Time: {np.mean(eval_metrics.waiting_times):.2f}")
    print(f"Average Bed Utilization: {np.mean(eval_metrics.bed_utilization)*100:.1f}%")

if __name__ == "__main__":
    main()