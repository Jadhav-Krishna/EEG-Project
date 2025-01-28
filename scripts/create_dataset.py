import numpy as np
import pandas as pd

# Define columns
columns = ['patient_id'] + [f'eeg_{i}' for i in range(1, 15)] + ['health_status', 'disease', 'mood']

# Generate realistic data
np.random.seed(42)
data = []
for i in range(1000):
    patient_id = i + 1
    health_status = np.random.choice(['healthy', 'unhealthy'], p=[0.3, 0.7])  # 30% healthy, 70% unhealthy
    disease = 'none' if health_status == 'healthy' else np.random.choice(['parkinson', 'brain_tumor', 'brain_stroke', 'alzheimer', 'epilepsy', 'migraine'])
    
    # Generate EEG signals based on health and disease
    if health_status == 'healthy':
        eeg_signals = np.random.normal(50, 10, 14)  # Normal EEG for healthy patients
    else:
        eeg_signals = np.random.normal(70, 20, 14)  # Abnormal EEG for unhealthy patients
    
    # Assign mood based on EEG signals
    if np.mean(eeg_signals) < 55:
        mood = 'happy'
    elif np.mean(eeg_signals) > 65:
        mood = 'sad'
    else:
        mood = 'neutral'
    
    data.append([patient_id] + list(eeg_signals) + [health_status, disease, mood])

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('data/raw_data.csv', index=False)
print("Dataset created and saved to data/raw_data.csv")