import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

# Load data
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

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

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
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=500, batch_size=32, class_weight=class_weights)

# Save model
model.save('models/disease_predictor_lstm.h5')
print("LSTM-based disease prediction model saved to models/disease_predictor_lstm.h5")

# Predict disease
y_pred_prob = model.predict(X)
y_pred = y_pred_prob.argmax(axis=1)

# Metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')

print(f"Disease Prediction Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("Classification Report:\n", classification_report(y, y_pred))

# ROC & AUC Curve
if len(np.unique(y)) > 2:  # Multi-class ROC-AUC
    y_bin = label_binarize(y, classes=np.unique(y))
    auc_score = roc_auc_score(y_bin, y_pred_prob, average='weighted', multi_class='ovr')
    print(f"ROC-AUC Score: {auc_score:.2f}")
    
    plt.figure()
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f'Class {i}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
else:
    fpr, tpr, _ = roc_curve(y, y_pred_prob[:, 1])
    auc_score = roc_auc_score(y, y_pred_prob[:, 1])
    print(f"ROC-AUC Score: {auc_score:.2f}")
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Save predictions
unhealthy_df['predicted_disease'] = label_encoder.inverse_transform(y_pred)
unhealthy_df.to_csv('data/disease_data_lstm.csv', index=False)
print("Disease data saved to data/disease_data_lstm.csv")
