import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from .utils import load_data, shape_for_sequence
from . import dnn_baseline, cnn_1d, lstm_model, rnn_model, cnn_lstm, transformer_encoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


MODEL_REGISTRY = {
    'dnn_baseline': {'module': dnn_baseline, 'is_seq': False},
    'cnn_1d': {'module': cnn_1d, 'is_seq': True},
    'lstm': {'module': lstm_model, 'is_seq': True},
    'rnn': {'module': rnn_model, 'is_seq': True},
    'cnn_lstm': {'module': cnn_lstm, 'is_seq': True},
    'transformer_encoder': {'module': transformer_encoder, 'is_seq': True},
}


def plot_confusion(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true, y_prob, class_names, title, out_path):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(6, 5))
    # Per-class ROC
    for i in range(n_classes):
        col = y_true_bin[:, i]
        if col.max() == col.min():
            continue
        fpr, tpr, _ = roc_curve(col, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f"micro-average (AUC={auc_micro:.3f})", linestyle='--', color='black')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true, y_prob, class_names, title, out_path):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(6, 5))
    # Per-class PR
    for i in range(n_classes):
        col = y_true_bin[:, i]
        if col.max() == col.min():
            continue
        precision, recall, _ = precision_recall_curve(col, y_prob[:, i])
        ap = average_precision_score(col, y_prob[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.3f})")
    # Micro-average PR
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
    ap_micro = average_precision_score(y_true_bin, y_prob, average='micro')
    plt.plot(recall, precision, label=f"micro-average (AP={ap_micro:.3f})", linestyle='--', color='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _weights_path(model_key: str, target: str) -> str:
    return os.path.join(RESULTS_DIR, f"{model_key}_{target}.weights.h5")


def predict_with_weights(model_key, target, X, X_seq, num_classes):
    entry = MODEL_REGISTRY[model_key]
    module = entry['module']
    is_seq = entry['is_seq']
    weights_path = _weights_path(model_key, target)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    if is_seq:
        model = module.build_model(X_seq.shape[1], num_classes)
        model.load_weights(weights_path)
        prob = model.predict(X_seq, verbose=0)
    else:
        model = module.build_model(X.shape[1], num_classes)
        model.load_weights(weights_path)
        prob = model.predict(X, verbose=0)
    pred = np.argmax(prob, axis=1)
    return prob, pred


def predict_ensemble(base_keys, target, X, X_seq, num_classes):
    probs = []
    for key in base_keys:
        p, _ = predict_with_weights(key, target, X, X_seq, num_classes)
        probs.append(p)
    avg_prob = np.mean(probs, axis=0)
    return avg_prob, np.argmax(avg_prob, axis=1)


def generate_for_target(target: str, graphs_dir: str):
    os.makedirs(graphs_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, enc, class_names = load_data(target=target, include_mean=True)
    X_test_seq = shape_for_sequence(X_test)

    keys = list(MODEL_REGISTRY.keys())
    existing = [k for k in keys if os.path.exists(_weights_path(k, target))]

    for k in existing:
        prob, pred = predict_with_weights(k, target, X_test, X_test_seq, len(class_names))
        plot_confusion(y_test, pred, class_names, f"Confusion Matrix — {k} ({target})",
                       os.path.join(graphs_dir, f"confusion_{k}.png"))
        plot_roc(y_test, prob, class_names, f"ROC (OvR) — {k} ({target})",
                 os.path.join(graphs_dir, f"roc_{k}.png"))
        plot_pr(y_test, prob, class_names, f"Precision–Recall — {k} ({target})",
                os.path.join(graphs_dir, f"pr_{k}.png"))

    ensemble_sets = [
        ("ensemble_dnn_cnn_lstm", ['dnn_baseline', 'cnn_1d', 'lstm']),
        ("ensemble_dnn_cnn_lstm_rnn", ['dnn_baseline', 'cnn_1d', 'lstm', 'rnn'])
    ]
    for tag, base_files in ensemble_sets:
        if all(os.path.exists(_weights_path(b, target)) for b in base_files):
            prob, pred = predict_ensemble(base_files, target, X_test, X_test_seq, len(class_names))
            plot_confusion(y_test, pred, class_names, f"Confusion Matrix — {tag} ({target})",
                           os.path.join(graphs_dir, f"confusion_{tag}.png"))
            plot_roc(y_test, prob, class_names, f"ROC (OvR) — {tag} ({target})",
                     os.path.join(graphs_dir, f"roc_{tag}.png"))
            plot_pr(y_test, prob, class_names, f"Precision–Recall — {tag} ({target})",
                    os.path.join(graphs_dir, f"pr_{tag}.png"))

    print(f"Graphs saved to {graphs_dir}")


def main():
    generate_for_target('mood', os.path.join(BASE_DIR, 'graphs', 'mood_benchmark'))
    generate_for_target('disease', os.path.join(BASE_DIR, 'graphs', 'disease_benchmark'))


if __name__ == '__main__':
    main()
