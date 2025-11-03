import json
import os
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import random
import tensorflow as tf

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed_data.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_data(target: str = 'mood', include_mean: bool = True, test_size: float = 0.25,
              include_aux_mood: bool = False, standardize: bool = True):
    """Load processed_data.csv and return X_train, X_test, y_train, y_test, encoder, class_names.
    - target: one of 'mood', 'health_status', or 'disease'.
    - include_mean: include 'eeg_mean' as a feature along with eeg_1..eeg_14.
    - include_aux_mood: when True and 'mood' column exists, append one-hot mood features to X (useful for disease target).
    """
    df = pd.read_csv(DATA_PATH)
    # Features
    eeg_cols = [f'eeg_{i}' for i in range(1, 15)]
    if include_mean and 'eeg_mean' in df.columns:
        eeg_cols = eeg_cols + ['eeg_mean']
    X = df[eeg_cols].values.astype('float32')

    # Optional auxiliary mood feature
    if include_aux_mood and 'mood' in df.columns:
        le_mood = LabelEncoder()
        mood_ids = le_mood.fit_transform(df['mood'].astype(str).values)
        mood_onehot = np.eye(len(le_mood.classes_), dtype='float32')[mood_ids]
        X = np.hstack([X, mood_onehot])

    # Target
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in data. Available: {list(df.columns)}")
    y_text = df[target].astype(str).values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    class_names = list(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    # Standardize features (fit on train, apply to test) for better optimization
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, encoder, class_names


def shape_for_sequence(X: np.ndarray) -> np.ndarray:
    """Reshape (n_samples, n_features) -> (n_samples, timesteps, channels) for sequence models.
    We treat features as timesteps with 1 channel.
    """
    return X.reshape((X.shape[0], X.shape[1], 1))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred).tolist()

    # ROC-AUC OvR for multiclass
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
        auc_ovr = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    except Exception:
        auc_ovr = None

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            'precision': precision_score(y_true, y_pred, labels=[idx], average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, labels=[idx], average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, labels=[idx], average='macro', zero_division=0)
        }

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'roc_auc_ovr': auc_ovr,
        'confusion_matrix': cm,
        'per_class': per_class,
    }


def save_metrics(model_name: str, target: str, class_names: List[str], params: int, train_seconds: float,
                 metrics: Dict[str, Any], extra: Dict[str, Any] = None, filename: str = None):
    payload = {
        'model': model_name,
        'target': target,
        'classes': class_names,
        'params': int(params),
        'train_seconds': float(train_seconds),
    }
    payload.update(metrics)
    if extra:
        payload.update(extra)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if filename is None:
        filename = f"{target}_{model_name.lower().replace(' ', '_')}_metrics.json"
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path


def get_fit_kwargs(target: str, y_train: np.ndarray) -> Dict[str, Any]:
    """Return additional kwargs for model.fit based on target/task specifics.
    - For 'disease', apply class weights to mitigate imbalance.
    """
    if target == 'disease':
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        return {'class_weight': class_weights}
    return {}


def set_global_seed(seed: int):
    """Set Python, NumPy, and TensorFlow seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)
