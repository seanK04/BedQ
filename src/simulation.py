import numpy as np
import torch
from patient import PatientGenerator
from model import DQLAgent
from logger import HospitalLogger

class HospitalEnvironment:
    def __init__(self, n_beds=82):
        self.n_beds = n_beds
        self.beds = [None] * n_beds
        self.waiting_patients = []
        self.current_time = 0
        self.generator = PatientGenerator()
        
    def reset(self):
        self.beds = [None] * self.n_beds
        self.waiting_patients = []
        self.current_time = 0
        return self.get_valid_actions()
        
    def step(self, time_delta=1.0):
        self.current_time += time_delta
        new_patients = self.generator.generate_batch(time_delta)
        for patient in new_patients:
            self.waiting_patients.append(patient)
        
        for i, patient in enumerate(self.beds):
            if patient is not None:
                if self.current_time - patient.arrival_time >= patient.stay_duration:
                    self.beds[i] = None
        return self.get_valid_actions()
    
    def get_valid_actions(self):
        return [i for i, bed in enumerate(self.beds) if bed is None]
    
    def assign_bed(self, bed_idx, patient_idx):
        if bed_idx >= self.n_beds or patient_idx >= len(self.waiting_patients):
            return False
        if self.beds[bed_idx] is not None:
            return False
        patient = self.waiting_patients.pop(patient_idx)
        self.beds[bed_idx] = patient
        return True
        
    def calculate_reward(self):
        utilization = sum(1 for bed in self.beds if bed is not None) / self.n_beds
        waiting_penalty = sum(
            patient.severity * (self.current_time - patient.arrival_time)
            for patient in self.waiting_patients
        )
        return utilization - 0.1 * waiting_penalty

def train(episodes=1000, steps_per_episode=100):
    env = HospitalEnvironment()
    state_size = 82 + 82 + 5 + 2  # bed_state + severity_state + waiting_patients + time_features
    agent = DQLAgent(state_size)
    logger = HospitalLogger()
    
    for episode in range(episodes):
        env.reset()
        total_reward = 0
        losses = []
        
        for step in range(steps_per_episode):
            state = agent.get_state(env.beds, env.waiting_patients, env.current_time)
            valid_actions = env.get_valid_actions()
            
            action_taken = False
            if valid_actions and env.waiting_patients:
                bed_idx = agent.act(state, valid_actions)
                action_successful = env.assign_bed(bed_idx, 0)
                
                if action_successful:
                    action_taken = True
                
                env.step()
                next_state = agent.get_state(env.beds, env.waiting_patients, env.current_time)
                reward = env.calculate_reward()
                
                agent.remember(state, bed_idx, reward, next_state, False)
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)
                
                total_reward += reward
            else:
                env.step()
            
            logger.log_step(env.beds, env.waiting_patients, action_taken)
        
        agent.update_target_model()
        avg_loss = np.mean(losses) if losses else 0
        
        logger.log_episode(
            episode=episode,
            reward=total_reward,
            loss=avg_loss,
            beds=env.beds,
            waiting_patients=env.waiting_patients,
            elapsed_time=env.current_time
        )
            
    return agent

if __name__ == "__main__":
    trained_agent = train()
    torch.save(trained_agent.model.state_dict(), "trained_model.pth")