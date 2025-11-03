import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from .plot_results import plot_confusion, plot_roc, plot_pr


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_disease_with_mood_encoded(test_size: float = 0.25, random_state: int = 42):
    df = pd.read_csv(DATA_PATH)
    eeg_cols = [f'eeg_{i}' for i in range(1, 15)]
    X_eeg = df[eeg_cols].astype('float32').values
    le_mood = LabelEncoder()
    mood_ids = le_mood.fit_transform(df['mood'].astype(str).values).astype('float32')
    X = np.hstack([X_eeg, mood_ids.reshape(-1, 1)])

    le_dis = LabelEncoder()
    y = le_dis.fit_transform(df['disease'].astype(str).values)
    class_names = list(le_dis.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_test, y_test, class_names


def main():
    graphs_dir = os.path.join(BASE_DIR, 'graphs', 'disease_benchmark')
    os.makedirs(graphs_dir, exist_ok=True)

    X_test, y_test, class_names = load_disease_with_mood_encoded()
    model_path = os.path.join(RESULTS_DIR, 'disease_dnn_moodencoded.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run disease_dnn_moodencoded.py first.")

    model = load_model(model_path)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Save plots with a clear prefix
    plot_confusion(y_test, y_pred, class_names,
                   title='Confusion Matrix — DNN (mood_encoded) (disease)',
                   out_path=os.path.join(graphs_dir, 'confusion_disease_dnn_moodencoded.png'))
    plot_roc(y_test, y_prob, class_names,
             title='ROC (OvR) — DNN (mood_encoded) (disease)',
             out_path=os.path.join(graphs_dir, 'roc_disease_dnn_moodencoded.png'))
    plot_pr(y_test, y_prob, class_names,
            title='Precision–Recall — DNN (mood_encoded) (disease)',
            out_path=os.path.join(graphs_dir, 'pr_disease_dnn_moodencoded.png'))
    print(f"External-like DNN graphs saved to {graphs_dir}")


if __name__ == '__main__':
    main()
