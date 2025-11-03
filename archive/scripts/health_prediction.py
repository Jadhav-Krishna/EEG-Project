import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/mood_data.csv')

# Encode health status
label_encoder = LabelEncoder()
df['health_encoded'] = label_encoder.fit_transform(df['health_status'])

X = df[[f'eeg_{i}' for i in range(1, 15)] + ['mood_encoded']].values
y = df['health_encoded'].values

# Build Dense Neural Network(DNN)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model on the entire dataset
model.fit(X, y, epochs=100, batch_size=32)

# Save model
model.save('models/health_predictor.h5')
print("Health prediction model saved to models/health_predictor.h5")

# Predict health status
y_pred = (model.predict(X) > 0.5).astype(int)
accuracy = accuracy_score(y, y_pred)
print(f"Health Prediction Accuracy: {accuracy * 100:.2f}%")

# Save health data
df['predicted_health'] = label_encoder.inverse_transform(y_pred)
df.to_csv('data/health_data.csv', index=False)
print("Health data saved to data/health_data.csv")