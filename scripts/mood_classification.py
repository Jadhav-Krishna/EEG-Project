import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Load processed data
df = pd.read_csv('data/processed_data.csv')

# Encode mood labels
label_encoder = LabelEncoder()
df['mood_encoded'] = label_encoder.fit_transform(df['mood'])

# Prepare data for LSTM
X = df[[f'eeg_{i}' for i in range(1, 15)]].values
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM
y = df['mood_encoded'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 output classes: happy, sad, neutral
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('models/mood_classifier.h5')
print("Mood classification model saved to models/mood_classifier.h5")

# Predict mood
y_pred = model.predict(X_test).argmax(axis=1)

# Calculate metrics
print("\nMood Classification Metrics:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# New: Calculate and print specific precision, recall, and F1 score for each class
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average=None, zero_division=0)
recall = recall_score(y_test, y_pred, average=None, zero_division=0)
f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

for idx, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} - Precision: {precision[idx]:.2f}, Recall: {recall[idx]:.2f}, F1 Score: {f1[idx]:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Mood Classification')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('graphs/mood_confusion_matrix.png')
plt.show()

# ROC Curve and AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Binarize labels for OvR
y_pred_prob = model.predict(X_test)  # Predicted probabilities

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(len(label_encoder.classes_)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Mood Classification')
plt.legend(loc="lower right")
plt.savefig('graphs/mood_roc_curve.png')
plt.show()

# Save mood data
df['predicted_mood'] = label_encoder.inverse_transform(model.predict(X).argmax(axis=1))
df.to_csv('data/mood_data.csv', index=False)
print("Mood data saved to data/mood_data.csv")
