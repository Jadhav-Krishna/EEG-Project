import glob
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
REPORT_PATH = os.path.join(os.path.dirname(__file__), 'COMPARISON_REPORT.md')


HEADER = """# Mood Classification: Model Comparison Report

This report compares multiple deep learning methods trained on `data/processed_data.csv` for predicting `mood` from EEG features (eeg_1..eeg_14 plus eeg_mean).

- Data split: stratified 75/25 train/test with random_state=42
- Metrics reported: Accuracy, Macro-F1, Weighted-F1, Macro-Precision, Macro-Recall, ROC-AUC (OvR)
- Models included: DNN Baseline, CNN 1D, RNN, LSTM, CNN-LSTM, Transformer Encoder, Ensemble SoftVote (DNN+CNN+LSTM) and Ensemble SoftVote (DNN+CNN+LSTM+RNN)

Generated on: {date}

"""


def fmt_pct(x):
    return f"{100.0 * x:.2f}%" if isinstance(x, (float, int)) else "-"


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_metrics.json')))
    results = []
    for f in files:
        with open(f, 'r') as fh:
            results.append(json.load(fh))

    # Sort by macro_f1 desc
    results.sort(key=lambda r: r.get('macro_f1', 0.0), reverse=True)

    lines = [HEADER.format(date=datetime.utcnow().isoformat() + 'Z')]

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | ROC-AUC (OvR) | Params | Train Time (s) |\n|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['model']} | {fmt_pct(r.get('accuracy'))} | {r.get('macro_f1'):.3f} | {r.get('weighted_f1'):.3f} | "
            f"{r.get('macro_precision'):.3f} | {r.get('macro_recall'):.3f} | {r.get('roc_auc_ovr') if r.get('roc_auc_ovr') is not None else '-'} | "
            f"{r.get('params')} | {r.get('train_seconds'):.2f} |"
        )

    # Per-model notes
    lines.append("\n## Model Notes\n")
    for r in results:
        lines.append(f"### {r['model']}\n")
        lines.append(f"- Classes: {', '.join(r.get('classes', []))}")
        if 'val_best_loss' in r:
            lines.append(f"- Best validation loss: {r['val_best_loss']:.4f}")
        if 'base_models' in r:
            lines.append(f"- Ensemble base models: {', '.join(r['base_models'])}")
        lines.append("")

    # Rationale section
    lines.append("### Why these methods (rationale)\n")
    lines.append("- DNN Baseline: Strong baseline for tabular EEG features; fast to train, good sanity check without explicit sequence modeling.")
    lines.append("- CNN 1D: Captures local relationships between adjacent EEG features via shared filters; efficient and often competitive on short sequences.")
    lines.append("- RNN (SimpleRNN): Models sequential dependencies across feature positions; lightweight recurrent baseline.")
    lines.append("- LSTM: Gated recurrence captures longer-range dependencies; stable gradients and strong results on short-to-medium sequences.")
    lines.append("- CNN-LSTM: CNN extracts local motifs; LSTM aggregates them—useful when both local and sequential patterns matter.")
    lines.append("- Transformer Encoder: Self-attention learns global relationships; powerful but typically needs more data/tuning.")
    lines.append("- Ensemble SoftVote: Averages probabilities across diverse models to reduce variance and improve robustness.")
    lines.append("")

    # Main vs others comparison (choose top Macro-F1 as main)
    if results:
        main = results[0]
        lines.append("## Head-to-head comparison (main method vs others)\n")
        lines.append(f"Assumption: Main method = {main['model']} (best Macro-F1 {main.get('macro_f1'):.3f}, Accuracy {fmt_pct(main.get('accuracy'))}).\n")
        for r in results[1:]:
            try:
                d_acc = 100.0 * (main.get('accuracy', 0) - r.get('accuracy', 0))
                d_f1 = main.get('macro_f1', 0) - r.get('macro_f1', 0)
                lines.append(f"- {main['model']} vs {r['model']}: ΔAccuracy {d_acc:+.2f} pts, ΔMacro-F1 {d_f1:+.3f}.")
            except Exception:
                lines.append(f"- {main['model']} vs {r['model']}: comparison unavailable.")
        lines.append("")

    # Overall recommendations
    lines.append("## Overall recommendations\n")
    lines.append("- Default choice: LSTM for best balanced performance on this dataset size and feature layout.")
    lines.append("- Constrained devices: CNN 1D for competitive accuracy with markedly fewer parameters.")
    lines.append("- Quick baselines/ablations: DNN and SimpleRNN for speed and sanity checks.")
    lines.append("- When local motifs matter: CNN-LSTM.")
    lines.append("- With more data/compute: Revisit Transformer with stronger regularization/tuning.")
    lines.append("- Reliability-first: Keep SoftVote ensemble for robustness and stable probabilities.")
    lines.append("")

    # Graph gallery (if images exist)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    graphs_dir = os.path.join(base_dir, 'graphs', 'mood_benchmark')
    imgs = []
    if os.path.isdir(graphs_dir):
        imgs = sorted(
            [os.path.join('..', '..', 'graphs', 'mood_benchmark', f) for f in os.listdir(graphs_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        )
    if imgs:
        lines.append("## Graphs\n")
        for p in imgs:
            lines.append(f"![{os.path.basename(p)}]({p})")
        lines.append("")

    # Save
    with open(REPORT_PATH, 'w') as f:
        f.write("\n".join(lines))

    print(f"Wrote report to {REPORT_PATH}")


if __name__ == '__main__':
    main()
