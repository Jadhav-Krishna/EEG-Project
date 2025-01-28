import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv('data/health_data.csv')

# Filter unhealthy patients
unhealthy_df = df[df['health_status'] == 'unhealthy']

# Encode disease labels
label_encoder = LabelEncoder()
unhealthy_df['disease_encoded'] = label_encoder.fit_transform(unhealthy_df['disease'])

# Prepare data
X = unhealthy_df[[f'eeg_{i}' for i in range(1, 15)] + ['mood_encoded']].values
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM: (samples, timesteps, features)
y = unhealthy_df['disease_encoded'].values

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),  # First LSTM layer
    Dropout(0.3),
    BatchNormalization(),
    LSTM(64, return_sequences=False),  # Second LSTM layer
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=500, batch_size=32, class_weight=class_weights)

# Save model
model.save('models/disease_predictor_lstm.h5')
print("LSTM-based disease prediction model saved to models/disease_predictor_lstm.h5")

# Predict disease
y_pred = model.predict(X).argmax(axis=1)
accuracy = accuracy_score(y, y_pred)
print(f"Disease Prediction Accuracy: {accuracy * 100:.2f}%")

# Save disease data
unhealthy_df['predicted_disease'] = label_encoder.inverse_transform(y_pred)
unhealthy_df.to_csv('data/disease_data_lstm.csv', index=False)
print("Disease data saved to data/disease_data_lstm.csv")