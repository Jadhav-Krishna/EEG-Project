import glob
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
REPORT_PATH = os.path.join(os.path.dirname(__file__), 'COMPARISON_REPORT.md')


HEADER = """# EEG Classification: Mood and Disease — Model Comparison Report

This report compares multiple deep learning methods trained on `data/processed_data.csv` for predicting targets from EEG features (eeg_1..eeg_14 plus eeg_mean).

- Targets covered: `mood` and `disease`
- Data split: stratified 75/25 train/test with random_state=42
- Metrics reported: Accuracy, F1 (macro), ROC-AUC (OvR); per-class precision/recall/f1 available inside each metrics JSON
- Models included: DNN Baseline, CNN 1D, RNN, LSTM, CNN-LSTM, Transformer Encoder, Ensemble SoftVote (DNN+CNN+LSTM) and Ensemble SoftVote (DNN+CNN+LSTM+RNN)

Generated on: {date}

"""


def fmt_pct(x):
    return f"{100.0 * x:.2f}%" if isinstance(x, (float, int)) else "-"


def _group_by_target(results):
    grouped = {}
    for r in results:
        tgt = r.get('target', 'unknown')
        grouped.setdefault(tgt, []).append(r)
    # within each target, keep only the best entry per model (highest F1)
    deduped = {}
    for tgt, rows in grouped.items():
        best_by_model = {}
        for r in rows:
            key = r.get('model', 'Unknown')
            cur = best_by_model.get(key)
            if cur is None or r.get('macro_f1', 0.0) > cur.get('macro_f1', 0.0):
                best_by_model[key] = r
        deduped[tgt] = sorted(best_by_model.values(), key=lambda x: x.get('macro_f1', 0.0), reverse=True)
    return deduped


def _render_table_for_target(lines, target, rows):
    lines.append(f"## {target.capitalize()} — Summary\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (OvR) | Params | Train Time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {fmt_pct(r.get('accuracy'))} | {r.get('macro_precision'):.3f} | {r.get('macro_recall'):.3f} | {r.get('macro_f1'):.3f} | "
            f"{r.get('roc_auc_ovr') if r.get('roc_auc_ovr') is not None else '-'} | "
            f"{r.get('params')} | {r.get('train_seconds'):.2f} |"
        )
    lines.append("")


def _render_head_to_head(lines, target, rows):
    if not rows:
        return
    top = rows[0]
    lines.append(f"### {target.capitalize()} — Head-to-head (top vs others)\n")
    lines.append(f"Main method = {top['model']} (best F1 {top.get('macro_f1'):.3f}, Accuracy {fmt_pct(top.get('accuracy'))}).\n")
    for r in rows[1:]:
        try:
            d_acc = 100.0 * (top.get('accuracy', 0) - r.get('accuracy', 0))
            d_f1 = top.get('macro_f1', 0) - r.get('macro_f1', 0)
            lines.append(f"- {top['model']} vs {r['model']}: ΔAccuracy {d_acc:+.2f} pts, ΔF1 {d_f1:+.3f}.")
        except Exception:
            lines.append(f"- {top['model']} vs {r['model']}: comparison unavailable.")
    lines.append("")


def _render_graphs(lines, target):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    subdir = 'mood_benchmark' if target == 'mood' else 'disease_benchmark'
    graphs_dir = os.path.join(base_dir, 'graphs', subdir)
    if not os.path.isdir(graphs_dir):
        return
    imgs = sorted(
        [os.path.join('..', '..', 'graphs', subdir, f) for f in os.listdir(graphs_dir)
         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
    )
    if imgs:
        lines.append(f"### {target.capitalize()} — Graphs\n")
        for p in imgs:
            lines.append(f"![{os.path.basename(p)}]({p})")
        lines.append("")


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_metrics.json')))
    results = []
    for f in files:
        with open(f, 'r') as fh:
            results.append(json.load(fh))

    grouped = _group_by_target(results)

    lines = [HEADER.format(date=datetime.utcnow().isoformat() + 'Z')]

    # Per-target summaries
    for target in ['mood', 'disease']:
        rows = grouped.get(target, [])
        _render_table_for_target(lines, target, rows)
        _render_head_to_head(lines, target, rows)
        _render_graphs(lines, target)

    # Per-model notes (across all)
    if results:
        lines.append("## Model Notes\n")
        for r in results:
            lines.append(f"### {r['model']} ({r.get('target', '-')})\n")
            lines.append(f"- Classes: {', '.join(r.get('classes', []))}")
            if 'val_best_loss' in r:
                lines.append(f"- Best validation loss: {r['val_best_loss']:.4f}")
            if 'base_models' in r:
                lines.append(f"- Ensemble base models: {', '.join(r['base_models'])}")
            lines.append("")

    # Rationale and recommendations
    lines.append("### Why these methods (rationale)\n")
    lines.append("- DNN Baseline: Strong baseline for tabular EEG features; fast to train, good sanity check without explicit sequence modeling.")
    lines.append("- CNN 1D: Captures local relationships between adjacent EEG features via shared filters; efficient and often competitive on short sequences.")
    lines.append("- RNN (SimpleRNN): Models sequential dependencies across feature positions; lightweight recurrent baseline.")
    lines.append("- LSTM: Gated recurrence captures longer-range dependencies; stable gradients and strong results on short-to-medium sequences.")
    lines.append("- CNN-LSTM: CNN extracts local motifs; LSTM aggregates them—useful when both local and sequential patterns matter.")
    lines.append("- Transformer Encoder: Self-attention learns global relationships; powerful but typically needs more data/tuning.")
    lines.append("- Ensemble SoftVote: Averages probabilities across diverse models to reduce variance and improve robustness.")
    lines.append("")

    lines.append("## Overall recommendations\n")
    lines.append("- Default choice: Favor the best-performing method per target above; CNN-LSTM and LSTM are often strong on these EEG features.")
    lines.append("- Constrained devices: CNN 1D for competitive accuracy with markedly fewer parameters.")
    lines.append("- Quick baselines/ablations: DNN and SimpleRNN for speed and sanity checks.")
    lines.append("- With more data/compute: Revisit Transformer with stronger regularization/tuning.")
    lines.append("- Reliability-first: Keep SoftVote ensemble for robustness and stable probabilities.")
    lines.append("")

    # Save
    with open(REPORT_PATH, 'w') as f:
        f.write("\n".join(lines))

    print(f"Wrote report to {REPORT_PATH}")


if __name__ == '__main__':
    main()
