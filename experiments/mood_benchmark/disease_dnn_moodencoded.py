import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .utils import compute_metrics, save_metrics, get_fit_kwargs


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def build_model(input_dim: int, num_classes: int):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
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
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def load_disease_with_mood_encoded(test_size: float = 0.25, random_state: int = 42):
    df = pd.read_csv(DATA_PATH)
    # features: eeg_1..eeg_14 + mood_encoded (single integer label)
    eeg_cols = [f'eeg_{i}' for i in range(1, 15)]
    X_eeg = df[eeg_cols].astype('float32').values

    # mood encoded as integers with LabelEncoder (deterministic alphabetical ordering)
    le_mood = LabelEncoder()
    mood_ids = le_mood.fit_transform(df['mood'].astype(str).values).astype('float32')
    X = np.hstack([X_eeg, mood_ids.reshape(-1, 1)])

    # target: disease encoded
    le_dis = LabelEncoder()
    y = le_dis.fit_transform(df['disease'].astype(str).values)
    class_names = list(le_dis.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, class_names


def train_and_evaluate():
    X_train, X_test, y_train, y_test, class_names = load_disease_with_mood_encoded()

    model = build_model(X_train.shape[1], len(class_names))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    weights_path = os.path.join(RESULTS_DIR, 'disease_dnn_moodencoded.weights.h5')
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    ]

    start = time.time()
    fit_kwargs = get_fit_kwargs('disease', y_train)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
        **fit_kwargs
    )
    train_seconds = time.time() - start

    # Ensure best weights loaded
    try:
        model.load_weights(weights_path)
    except Exception:
        pass

    y_prob = model.predict(X_test, verbose=0)
    metrics = compute_metrics(y_test, y_prob, class_names)

    out_path = save_metrics(
        model_name='DNN (mood_encoded)',
        target='disease',
        class_names=class_names,
        params=model.count_params(),
        train_seconds=train_seconds,
        metrics=metrics,
        extra={'source': 'processed_data: eeg_1..14 + mood_encoded', 'val_best_loss': float(np.min(history.history.get('val_loss', [np.nan])))} ,
        filename='disease_dnn_moodencoded_metrics.json'
    )

    # Save model (optional, for reproducibility)
    model.save(os.path.join(RESULTS_DIR, 'disease_dnn_moodencoded.h5'))
    print(f"Saved metrics to {out_path}")


if __name__ == '__main__':
    train_and_evaluate()
