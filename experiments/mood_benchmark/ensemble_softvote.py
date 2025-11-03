import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from .utils import load_data, shape_for_sequence, compute_metrics, save_metrics, RESULTS_DIR


def filenames_for_target(target: str):
    return [
        (f'dnn_baseline_{target}.h5', False),
        (f'cnn_1d_{target}.h5', True),
        (f'lstm_{target}.h5', True),
        (f'rnn_{target}.h5', True),
    ]


def train_and_evaluate(target: str = 'mood'):
    # Load a consistent split
    X_train, X_test, y_train, y_test, encoder, class_names = load_data(target=target, include_mean=True)
    X_test_seq = shape_for_sequence(X_test)

    probs = []
    for fname, is_seq in filenames_for_target(target):
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        model = load_model(path, compile=False)
        if is_seq:
            y_prob = model.predict(X_test_seq, verbose=0)
        else:
            y_prob = model.predict(X_test, verbose=0)
        probs.append(y_prob)

    if not probs:
        raise RuntimeError("No base model predictions found. Run base models first.")

    avg_prob = np.mean(probs, axis=0)
    metrics = compute_metrics(y_test, avg_prob, class_names)

    out_path = save_metrics(
        model_name='Ensemble SoftVote (DNN+CNN+LSTM+RNN)',
        target=target,
        class_names=class_names,
        params=0,
        train_seconds=0.0,
        metrics=metrics,
        extra={'base_models': [m for m, _ in filenames_for_target(target)]}
    )
    return metrics

if __name__ == '__main__':
    train_and_evaluate()
