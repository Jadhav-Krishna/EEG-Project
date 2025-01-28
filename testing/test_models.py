import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

mood_classifier = load_model('models/mood_classifier.h5')
health_predictor = load_model('models/health_predictor.h5')
disease_predictor = load_model('models/disease_predictor.h5')

# For mood
mood_encoder = LabelEncoder()
mood_encoder.classes_ = np.array(['happy', 'sad', 'neutral'])

# For disease
disease_encoder = LabelEncoder()
disease_encoder.classes_ = np.array(['parkinson', 'brain_tumor', 'brain_stroke', 'alzheimer', 'epilepsy', 'migraine'])  # Replace with actual classes if saved

# Sample data for testing
sample_data = {
    'eeg_1': [50.2, 55.1, 60.3],
    'eeg_2': [35.7, 40.3, 45.6],
    'eeg_3': [65.2, 70.1, 75.6],
    'eeg_4': [30.6, 35.9, 40.2],
    'eeg_5': [60.3, 65.8, 70.1],
    'eeg_6': [45.6, 50.1, 55.2],
    'eeg_7': [75.6, 80.2, 85.3],
    'eeg_8': [25.4, 30.7, 35.8],
    'eeg_9': [55.2, 60.8, 65.3],
    'eeg_10': [40.3, 45.9, 50.2],
    'eeg_11': [70.1, 75.6, 80.3],
    'eeg_12': [35.7, 40.2, 45.6],
    'eeg_13': [60.3, 65.7, 70.2],
    'eeg_14': [50.2, 55.1, 60.3],
    'mood_encoded': [0, 1, 2]  # 0: happy, 1: sad, 2: neutral
}

# Convert sample data to DataFrame
sample_df = pd.DataFrame(sample_data)

# Prepare input data for each model
X_mood = sample_df[[f'eeg_{i}' for i in range(1, 15)]].values
X_mood = X_mood.reshape((X_mood.shape[0], X_mood.shape[1], 1))  # Reshape for LSTM

X_health = sample_df[[f'eeg_{i}' for i in range(1, 15)] + ['mood_encoded']].values

X_disease = sample_df[[f'eeg_{i}' for i in range(1, 15)] + ['mood_encoded']].values
X_disease = X_disease.reshape((X_disease.shape[0], X_disease.shape[1], 1))  # Reshape for LSTM

# Predict mood
mood_predictions = mood_classifier.predict(X_mood).argmax(axis=1)
sample_df['predicted_mood'] = mood_encoder.inverse_transform(mood_predictions)

# Predict health status
health_predictions = (health_predictor.predict(X_health) > 0.5).astype(int)
sample_df['predicted_health'] = ['healthy' if pred == 0 else 'unhealthy' for pred in health_predictions]

# Predict disease (only for unhealthy patients)
sample_df['predicted_disease'] = 'none'  # Default value for healthy patients
unhealthy_mask = sample_df['predicted_health'] == 'unhealthy'  # Mask for unhealthy patients
if unhealthy_mask.any():
    X_disease = sample_df[unhealthy_mask][[f'eeg_{i}' for i in range(1, 15)] + ['mood_encoded']].values
    X_disease = X_disease.reshape((X_disease.shape[0], X_disease.shape[1], 1))  # Reshape for LSTM
    disease_predictions = disease_predictor.predict(X_disease).argmax(axis=1)
    sample_df.loc[unhealthy_mask, 'predicted_disease'] = disease_encoder.inverse_transform(disease_predictions)

# Display results
print("Sample Data with Predictions:")
print(sample_df[['eeg_1', 'eeg_2', 'mood_encoded', 'predicted_mood', 'predicted_health', 'predicted_disease']])

# True labels for accuracy calculation
true_mood = ['happy', 'sad', 'neutral']
true_health = ['healthy', 'unhealthy', 'unhealthy']
true_disease = ['none', 'parkinson', 'brain_tumor']

# Calculate accuracy
mood_accuracy = accuracy_score(true_mood, sample_df['predicted_mood'])
health_accuracy = accuracy_score(true_health, sample_df['predicted_health'])
disease_accuracy = accuracy_score(true_disease, sample_df['predicted_disease'])

print(f"\nMood Prediction Accuracy: {mood_accuracy * 100:.2f}%")
print(f"Health Prediction Accuracy: {health_accuracy * 100:.2f}%")
print(f"Disease Prediction Accuracy: {disease_accuracy * 100:.2f}%")

# Generate Graphs
sns.set(style="whitegrid")

# 1. Mood Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='predicted_mood', data=sample_df, palette='viridis')
plt.title('Predicted Mood Distribution')
plt.xlabel('Mood')
plt.ylabel('Count')
plt.savefig('graphs/predicted_mood_distribution.png')
plt.show()

# 2. Health Status Distribution
plt.figure(figsize=(8, 6))
sample_df['predicted_health'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Predicted Health Status Distribution')
plt.ylabel('')
plt.savefig('graphs/predicted_health_distribution.png')
plt.show()

# 3. Disease Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='predicted_disease', data=sample_df, palette='magma')
plt.title('Predicted Disease Distribution (Unhealthy Patients)')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('graphs/predicted_disease_distribution.png')
plt.show()

# 4. Confusion Matrix for Mood Classification
cm_mood = confusion_matrix(true_mood, sample_df['predicted_mood'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mood, annot=True, fmt='d', cmap='Blues', xticklabels=mood_encoder.classes_, yticklabels=mood_encoder.classes_)
plt.title('Confusion Matrix for Mood Classification')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/mood_confusion_matrix.png')
plt.show()

# 5. Confusion Matrix for Health Prediction
cm_health = confusion_matrix(true_health, sample_df['predicted_health'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_health, annot=True, fmt='d', cmap='Greens', xticklabels=['healthy', 'unhealthy'], yticklabels=['healthy', 'unhealthy'])
plt.title('Confusion Matrix for Health Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/health_confusion_matrix.png')
plt.show()

# 6. Confusion Matrix for Disease Prediction
cm_disease = confusion_matrix(true_disease, sample_df['predicted_disease'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm_disease, annot=True, fmt='d', cmap='Reds', xticklabels=disease_encoder.classes_, yticklabels=disease_encoder.classes_)
plt.title('Confusion Matrix for Disease Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/disease_confusion_matrix.png')
plt.show()