import numpy as np
from pathlib import Path

class HospitalLogger:
    def __init__(self):
        self.reset_episode_stats()
        
    def reset_episode_stats(self):
        self.current_episode = {
            'assigned_patients': 0,
            'total_patients': 0,
            'wait_times': []
        }
    
    def log_step(self, beds, waiting_patients, action_taken=None):
        if action_taken:
            self.current_episode['assigned_patients'] += 1
        self.current_episode['total_patients'] += len(waiting_patients)
    
    def log_episode(self, episode, reward, loss, beds, waiting_patients, elapsed_time):
        # Calculate basic metrics
        bed_utilization = sum(1 for bed in beds if bed is not None) / len(beds)
        avg_wait = np.mean(self.current_episode['wait_times']) if self.current_episode['wait_times'] else 0
        
        # Print simple summary
        print(
            f"Episode {episode:3d} | "
            f"Reward: {reward:6.1f} | "
            f"Loss: {loss:6.3f} | "
            f"Beds Used: {bed_utilization:5.1%} | "
            f"Waiting: {len(waiting_patients):3d}"
        )
        
        self.reset_episode_stats()