import pandas as pd

df = pd.read_csv('data/raw_data.csv')

df['eeg_mean'] = df[[f'eeg_{i}' for i in range(1, 15)]].mean(axis=1)

df.to_csv('data/processed_data.csv', index=False)
print("Mean calculated and saved to data/processed_data.csv")