import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/processed_data.csv')

# Encode mood labels
label_encoder = LabelEncoder()
df['mood_encoded'] = label_encoder.fit_transform(df['mood'])

# Prepare data for CNN
X = df[[f'eeg_{i}' for i in range(1, 15)]].values
X = X.reshape((X.shape[0], X.shape[1], 1))
y = df['mood_encoded'].values

# Build CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model on the entire dataset
model.fit(X, y, epochs=100, batch_size=32)

# Save model
model.save('models/mood_classifier.h5')
print("Mood classification model saved to models/mood_classifier.h5")

# Predict mood
y_pred = model.predict(X).argmax(axis=1)
accuracy = accuracy_score(y, y_pred)
print(f"Mood Classification Accuracy: {accuracy * 100:.2f}%")

# Save mood data
df['predicted_mood'] = label_encoder.inverse_transform(y_pred)
df.to_csv('data/mood_data.csv', index=False)
print("Mood data saved to data/mood_data.csv")