import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load all datasets
raw_data = pd.read_csv('data/raw_data.csv')
processed_data = pd.read_csv('data/processed_data.csv')
mood_data = pd.read_csv('data/mood_data.csv')
health_data = pd.read_csv('data/health_data.csv')
disease_data = pd.read_csv('data/disease_data.csv')

# Set style for plots
sns.set(style="whitegrid")

# 1. EEG Signal Distribution
plt.figure(figsize=(14, 8))
for i in range(1, 15):
    sns.kdeplot(raw_data[f'eeg_{i}'], label=f'EEG {i}')
plt.title('Distribution of EEG Signals')
plt.xlabel('EEG Signal Value')
plt.ylabel('Density')
plt.legend()
plt.savefig('graphs/eeg_signal_distribution.png')
plt.show()

# 2. Mood Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='mood', data=mood_data, palette='viridis')
plt.title('Mood Distribution')
plt.xlabel('Mood')
plt.ylabel('Count')
plt.savefig('graphs/mood_distribution.png')
plt.show()

# 3. Health Status Distribution
plt.figure(figsize=(8, 6))
health_data['health_status'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Health Status Distribution')
plt.ylabel('')
plt.savefig('graphs/health_status_distribution.png')
plt.show()

# 4. Disease Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='disease', data=disease_data, palette='magma')
plt.title('Disease Distribution (Unhealthy Patients)')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('graphs/disease_distribution.png')
plt.show()

# 5. Model Performance (Example for Mood Classification)
# Load mood classification history (if available)
# history = pd.read_csv('logs/mood_classification_history.csv')
# plt.figure(figsize=(12, 6))
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title('Mood Classification Model Performance')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('graphs/mood_classification_performance.png')
# plt.show()

# 6. Confusion Matrix for Mood Classification
y_true = mood_data['mood_encoded']
y_pred = mood_data['predicted_mood'].map({'happy': 0, 'sad': 1, 'neutral': 2})
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['happy', 'sad', 'neutral'], yticklabels=['happy', 'sad', 'neutral'])
plt.title('Confusion Matrix for Mood Classification')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/mood_confusion_matrix.png')
plt.show()

# 7. Feature Correlation
plt.figure(figsize=(12, 8))
corr = processed_data[[f'eeg_{i}' for i in range(1, 15)] + ['mood_encoded', 'health_encoded']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('graphs/feature_correlation.png')
plt.show()

# 8. Mean EEG Signal by Disease
plt.figure(figsize=(12, 8))
mean_eeg_by_disease = disease_data.groupby('disease')[[f'eeg_{i}' for i in range(1, 15)]].mean()
sns.heatmap(mean_eeg_by_disease, annot=True, cmap='viridis', fmt='.2f')
plt.title('Mean EEG Signal by Disease')
plt.xlabel('EEG Signal')
plt.ylabel('Disease')
plt.savefig('graphs/mean_eeg_by_disease.png')
plt.show()

# 9. Confusion Matrix for Disease Prediction
y_true = disease_data['disease_encoded']
y_pred = disease_data['predicted_disease'].map({disease: i for i, disease in enumerate(label_encoder.classes_)})
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Disease Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/disease_confusion_matrix.png')
plt.show()

# 10. Training and Validation Curves (Example for Disease Prediction)
# Load disease prediction history (if available)
# history = pd.read_csv('logs/disease_prediction_history.csv')
# plt.figure(figsize=(12, 6))
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title('Disease Prediction Model Performance')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('graphs/disease_prediction_performance.png')
# plt.show()