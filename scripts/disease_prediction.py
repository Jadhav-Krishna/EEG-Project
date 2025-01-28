import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
y = unhealthy_df['disease_encoded'].values

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

# Build Dense Neural Network(DNN)
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model on the entire dataset with class weights
model.fit(X, y, epochs=500, batch_size=32, class_weight=class_weights)

# Save model
model.save('models/disease_predictor.h5')
print("Disease prediction model saved to models/disease_predictor.h5")

# Predict disease
y_pred = model.predict(X).argmax(axis=1)
accuracy = accuracy_score(y, y_pred)
print(f"Disease Prediction Accuracy: {accuracy * 100:.2f}%")

# Save disease data
unhealthy_df['predicted_disease'] = label_encoder.inverse_transform(y_pred)
unhealthy_df.to_csv('data/disease_data.csv', index=False)
print("Disease data saved to data/disease_data.csv")