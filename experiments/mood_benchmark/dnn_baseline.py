import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .utils import load_data, compute_metrics, save_metrics, get_fit_kwargs


def build_model(input_dim: int, num_classes: int):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
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


def train_and_evaluate(target: str = 'mood'):
    X_train, X_test, y_train, y_test, encoder, class_names = load_data(target=target, include_mean=True)
    model = build_model(X_train.shape[1], len(class_names))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    start = time.time()
    fit_kwargs = get_fit_kwargs(target, y_train)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
        **fit_kwargs
    )
    train_seconds = time.time() - start

    y_prob = model.predict(X_test, verbose=0)
    metrics = compute_metrics(y_test, y_prob, class_names)

    out_path = save_metrics(
        model_name='DNN Baseline',
        target=target,
        class_names=class_names,
        params=model.count_params(),
        train_seconds=train_seconds,
        metrics=metrics,
        extra={'val_best_loss': float(np.min(history.history['val_loss']))}
    )
    base_dir = os.path.dirname(out_path)
    model_path = os.path.join(base_dir, f"dnn_baseline_{target}.h5")
    model.save(model_path)
    model.save_weights(os.path.join(base_dir, f"dnn_baseline_{target}.weights.h5"))
    return metrics

if __name__ == '__main__':
    train_and_evaluate()
