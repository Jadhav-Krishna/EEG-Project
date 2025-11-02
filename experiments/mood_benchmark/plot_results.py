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
GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs', 'mood_benchmark')
os.makedirs(GRAPHS_DIR, exist_ok=True)


MODEL_REGISTRY = {
    'dnn_baseline': {'module': dnn_baseline, 'is_seq': False, 'weight_file': 'dnn_baseline.weights.h5'},
    'cnn_1d': {'module': cnn_1d, 'is_seq': True, 'weight_file': 'cnn_1d.weights.h5'},
    'lstm': {'module': lstm_model, 'is_seq': True, 'weight_file': 'lstm.weights.h5'},
    'rnn': {'module': rnn_model, 'is_seq': True, 'weight_file': 'rnn.weights.h5'},
    'cnn_lstm': {'module': cnn_lstm, 'is_seq': True, 'weight_file': 'cnn_lstm.weights.h5'},
    'transformer_encoder': {'module': transformer_encoder, 'is_seq': True, 'weight_file': 'transformer_encoder.weights.h5'},
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
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    plt.figure(figsize=(6, 5))
    for i in range(n_classes):
        col = y_true_bin[:, i]
        if col.max() == col.min():
            continue
        fpr, tpr, _ = roc_curve(col, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"micro-average (AUC={roc_auc:.3f})", linestyle='--', color='black')
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true, y_prob, class_names, title, out_path):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    plt.figure(figsize=(6, 5))
    for i in range(n_classes):
        col = y_true_bin[:, i]
        if col.max() == col.min():
            continue
        precision, recall, _ = precision_recall_curve(col, y_prob[:, i])
        ap = average_precision_score(col, y_prob[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.3f})")
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


def predict_with_weights(model_key, X, X_seq, num_classes):
    entry = MODEL_REGISTRY[model_key]
    module = entry['module']
    is_seq = entry['is_seq']
    weights_path = os.path.join(RESULTS_DIR, entry['weight_file'])
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


def predict_ensemble(base_keys, X, X_seq, num_classes):
    probs = []
    for key in base_keys:
        p, _ = predict_with_weights(key, X, X_seq, num_classes)
        probs.append(p)
    avg_prob = np.mean(probs, axis=0)
    return avg_prob, np.argmax(avg_prob, axis=1)


def main():
    X_train, X_test, y_train, y_test, enc, class_names = load_data(target='mood', include_mean=True)
    X_test_seq = shape_for_sequence(X_test)

    # Discover models
    keys = list(MODEL_REGISTRY.keys())
    existing = [k for k in keys if os.path.exists(os.path.join(RESULTS_DIR, MODEL_REGISTRY[k]['weight_file']))]

    for k in existing:
        prob, pred = predict_with_weights(k, X_test, X_test_seq, len(class_names))
        plot_confusion(y_test, pred, class_names, f"Confusion Matrix — {k}",
                       os.path.join(GRAPHS_DIR, f"confusion_{k}.png"))
        plot_roc(y_test, prob, class_names, f"ROC (OvR) — {k}",
                 os.path.join(GRAPHS_DIR, f"roc_{k}.png"))
        plot_pr(y_test, prob, class_names, f"Precision–Recall — {k}",
                os.path.join(GRAPHS_DIR, f"pr_{k}.png"))

    ensemble_sets = [
        ("ensemble_dnn_cnn_lstm", ['dnn_baseline', 'cnn_1d', 'lstm']),
        ("ensemble_dnn_cnn_lstm_rnn", ['dnn_baseline', 'cnn_1d', 'lstm', 'rnn'])
    ]
    for tag, base_files in ensemble_sets:
        if all(os.path.exists(os.path.join(RESULTS_DIR, MODEL_REGISTRY[b]['weight_file'])) for b in base_files):
            prob, pred = predict_ensemble(base_files, X_test, X_test_seq, len(class_names))
            plot_confusion(y_test, pred, class_names, f"Confusion Matrix — {tag}",
                           os.path.join(GRAPHS_DIR, f"confusion_{tag}.png"))
            plot_roc(y_test, prob, class_names, f"ROC (OvR) — {tag}",
                     os.path.join(GRAPHS_DIR, f"roc_{tag}.png"))
            plot_pr(y_test, prob, class_names, f"Precision–Recall — {tag}",
                    os.path.join(GRAPHS_DIR, f"pr_{tag}.png"))

    print(f"Graphs saved to {GRAPHS_DIR}")


if __name__ == '__main__':
    main()
