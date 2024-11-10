import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Patient:
    """Patient class representing a single hospital patient"""
    arrival_time: float  # hours since simulation start
    stay_duration: float  # expected stay in hours
    severity: float  # severity score 1-10
    bed_id: Optional[int] = None  # bed assigned to patient, if any
    waiting_start: Optional[float] = None  # when patient started waiting
        
    @property
    def wait_time(self) -> float:
        """Calculate current wait time in hours"""
        if self.waiting_start is None:
            return 0
        return datetime.now().timestamp() - self.waiting_start

    @property
    def loss(self) -> float:
        """Calculate current loss based on wait time and severity"""
        return self.wait_time * self.severity

class PatientGenerator:
    """Generates patient vectors using Monte Carlo simulation"""
    def __init__(self, 
                 arrival_rate: float = 2.5,  # patients per hour
                 mean_stay: float = 84,      # mean stay in hours (3.5 days)
                 std_stay: float = 36,       # std of stay in hours (1.5 days)
                 severity_alpha: float = 2,   # beta distribution α
                 severity_beta: float = 5):   # beta distribution β
        self.arrival_rate = arrival_rate
        self.mean_stay = mean_stay
        self.std_stay = std_stay
        self.severity_alpha = severity_alpha
        self.severity_beta = severity_beta
        
    def generate_arrival_times(self, duration: float) -> np.ndarray:
        """Generate Poisson arrival times for simulation duration"""
        n_arrivals = np.random.poisson(self.arrival_rate * duration)
        return np.sort(np.random.uniform(0, duration, n_arrivals))
    
    def generate_patient(self, arrival_time: float) -> Patient:
        """Generate a single patient vector"""
        # Correct calculation of log-normal parameters
        variance = self.std_stay ** 2
        mu = np.log(self.mean_stay**2 / np.sqrt(variance + self.mean_stay**2))
        sigma = np.sqrt(np.log(1 + (variance / self.mean_stay**2)))
        
        # Generate log-normal stay duration
        stay = np.random.lognormal(mean=mu, sigma=sigma)
        
        # Generate beta-distributed severity score
        severity = np.random.beta(self.severity_alpha, self.severity_beta)
        severity = 1 + (severity * 9)  # Scale to [1,10]
        
        return Patient(
            arrival_time=arrival_time,
            stay_duration=stay,
            severity=severity
        )
        
    def generate_batch(self, duration: float) -> list:
        """Generate a batch of patients for given duration"""
        arrival_times = self.generate_arrival_times(duration)
        return [self.generate_patient(t) for t in arrival_times]

def main():
    """Test patient generation"""
    generator = PatientGenerator()
    patients = generator.generate_batch(duration=24)  # 24 hours
    
    print(f"Generated {len(patients)} patients")
    for i, patient in enumerate(patients[:5]):
        print(f"\nPatient {i+1}:")
        print(f"Arrival: {patient.arrival_time:.1f} hours")
        print(f"Stay duration: {patient.stay_duration/24:.1f} days")
        print(f"Severity: {patient.severity:.1f}")

if __name__ == "__main__":
    main()
