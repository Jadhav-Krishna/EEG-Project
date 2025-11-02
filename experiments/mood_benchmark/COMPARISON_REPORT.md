# Mood Classification: Model Comparison Report

This report compares multiple deep learning methods trained on `data/processed_data.csv` for predicting `mood` from EEG features (eeg_1..eeg_14 plus eeg_mean).

- Data split: stratified 75/25 train/test with random_state=42
- Metrics reported: Accuracy, Macro-F1, Weighted-F1, Macro-Precision, Macro-Recall, ROC-AUC (OvR)
- Models included: DNN Baseline, CNN 1D, RNN, LSTM, CNN-LSTM, Transformer Encoder, Ensemble SoftVote (DNN+CNN+LSTM) and Ensemble SoftVote (DNN+CNN+LSTM+RNN)

Generated on: 2025-11-02T07:42:40.750257Z


## Summary Table

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | ROC-AUC (OvR) | Params | Train Time (s) |
|:------|---------:|---------:|------------:|----------------:|-------------:|--------------:|------:|---------------:|
| LSTM | 97.20% | 0.949 | 0.971 | 0.963 | 0.936 | 0.9981172752178636 | 121091 | 10.52 |
| CNN-LSTM | 96.40% | 0.930 | 0.962 | 0.965 | 0.906 | 0.993653053270194 | 50499 | 6.63 |
| CNN 1D | 96.00% | 0.918 | 0.956 | 0.976 | 0.885 | 0.9988319343190847 | 17475 | 5.40 |
| Ensemble SoftVote (DNN+CNN+LSTM) | 95.50% | 0.909 | 0.951 | 0.955 | 0.881 | 0.9987572295369667 | 0 | 0.00 |
| DNN Baseline | 93.60% | 0.865 | 0.929 | 0.931 | 0.832 | 0.9912554153562237 | 47235 | 6.47 |
| Ensemble SoftVote (DNN+CNN+LSTM+RNN) | 93.60% | 0.860 | 0.927 | 0.944 | 0.825 | 0.998054927043743 | 0 | 0.00 |
| RNN | 87.20% | 0.619 | 0.818 | 0.583 | 0.660 | 0.9384974035479742 | 34115 | 4.33 |
| Transformer Encoder | 68.00% | 0.470 | 0.637 | 0.449 | 0.496 | 0.7590349247523956 | 10915 | 5.35 |

## Model Notes

### LSTM

- Classes: happy, neutral, sad
- Best validation loss: 0.0572

### CNN-LSTM

- Classes: happy, neutral, sad
- Best validation loss: 0.0860

### CNN 1D

- Classes: happy, neutral, sad
- Best validation loss: 0.0686

### Ensemble SoftVote (DNN+CNN+LSTM)

- Classes: happy, neutral, sad
- Ensemble base models: dnn_baseline.h5, cnn_1d.h5, lstm.h5

### DNN Baseline

- Classes: happy, neutral, sad
- Best validation loss: 0.1427

### Ensemble SoftVote (DNN+CNN+LSTM+RNN)

- Classes: happy, neutral, sad
- Ensemble base models: dnn_baseline.h5, cnn_1d.h5, lstm.h5, rnn.h5

### RNN

- Classes: happy, neutral, sad
- Best validation loss: 0.2953

### Transformer Encoder

- Classes: happy, neutral, sad
- Best validation loss: 0.6548

### Why these methods (rationale)

- DNN Baseline: Strong baseline for tabular EEG features; fast to train, good sanity check without explicit sequence modeling.
- CNN 1D: Captures local relationships between adjacent EEG features via shared filters; efficient and often competitive on short sequences.
- RNN (SimpleRNN): Models sequential dependencies across feature positions; lightweight recurrent baseline.
- LSTM: Gated recurrence captures longer-range dependencies; stable gradients and strong results on short-to-medium sequences.
- CNN-LSTM: CNN extracts local motifs; LSTM aggregates them—useful when both local and sequential patterns matter.
- Transformer Encoder: Self-attention learns global relationships; powerful but typically needs more data/tuning.
- Ensemble SoftVote: Averages probabilities across diverse models to reduce variance and improve robustness.

## Head-to-head comparison (main method vs others)

Assumption: Main method = LSTM (best Macro-F1 0.949, Accuracy 97.20%).

- LSTM vs CNN-LSTM: ΔAccuracy +0.80 pts, ΔMacro-F1 +0.019.
- LSTM vs CNN 1D: ΔAccuracy +1.20 pts, ΔMacro-F1 +0.030.
- LSTM vs Ensemble SoftVote (DNN+CNN+LSTM): ΔAccuracy +1.70 pts, ΔMacro-F1 +0.039.
- LSTM vs DNN Baseline: ΔAccuracy +3.60 pts, ΔMacro-F1 +0.083.
- LSTM vs Ensemble SoftVote (DNN+CNN+LSTM+RNN): ΔAccuracy +3.60 pts, ΔMacro-F1 +0.089.
- LSTM vs RNN: ΔAccuracy +10.00 pts, ΔMacro-F1 +0.329.
- LSTM vs Transformer Encoder: ΔAccuracy +29.20 pts, ΔMacro-F1 +0.478.

## Overall recommendations

- Default choice: LSTM for best balanced performance on this dataset size and feature layout.
- Constrained devices: CNN 1D for competitive accuracy with markedly fewer parameters.
- Quick baselines/ablations: DNN and SimpleRNN for speed and sanity checks.
- When local motifs matter: CNN-LSTM.
- With more data/compute: Revisit Transformer with stronger regularization/tuning.
- Reliability-first: Keep SoftVote ensemble for robustness and stable probabilities.

## Graphs

![confusion_cnn_1d.png](../../graphs/mood_benchmark/confusion_cnn_1d.png)
![confusion_cnn_lstm.png](../../graphs/mood_benchmark/confusion_cnn_lstm.png)
![confusion_dnn_baseline.png](../../graphs/mood_benchmark/confusion_dnn_baseline.png)
![confusion_ensemble_dnn_cnn_lstm.png](../../graphs/mood_benchmark/confusion_ensemble_dnn_cnn_lstm.png)
![confusion_ensemble_dnn_cnn_lstm_rnn.png](../../graphs/mood_benchmark/confusion_ensemble_dnn_cnn_lstm_rnn.png)
![confusion_lstm.png](../../graphs/mood_benchmark/confusion_lstm.png)
![confusion_rnn.png](../../graphs/mood_benchmark/confusion_rnn.png)
![confusion_transformer_encoder.png](../../graphs/mood_benchmark/confusion_transformer_encoder.png)
![pr_cnn_1d.png](../../graphs/mood_benchmark/pr_cnn_1d.png)
![pr_cnn_lstm.png](../../graphs/mood_benchmark/pr_cnn_lstm.png)
![pr_dnn_baseline.png](../../graphs/mood_benchmark/pr_dnn_baseline.png)
![pr_ensemble_dnn_cnn_lstm.png](../../graphs/mood_benchmark/pr_ensemble_dnn_cnn_lstm.png)
![pr_ensemble_dnn_cnn_lstm_rnn.png](../../graphs/mood_benchmark/pr_ensemble_dnn_cnn_lstm_rnn.png)
![pr_lstm.png](../../graphs/mood_benchmark/pr_lstm.png)
![pr_rnn.png](../../graphs/mood_benchmark/pr_rnn.png)
![pr_transformer_encoder.png](../../graphs/mood_benchmark/pr_transformer_encoder.png)
![roc_cnn_1d.png](../../graphs/mood_benchmark/roc_cnn_1d.png)
![roc_cnn_lstm.png](../../graphs/mood_benchmark/roc_cnn_lstm.png)
![roc_dnn_baseline.png](../../graphs/mood_benchmark/roc_dnn_baseline.png)
![roc_ensemble_dnn_cnn_lstm.png](../../graphs/mood_benchmark/roc_ensemble_dnn_cnn_lstm.png)
![roc_ensemble_dnn_cnn_lstm_rnn.png](../../graphs/mood_benchmark/roc_ensemble_dnn_cnn_lstm_rnn.png)
![roc_lstm.png](../../graphs/mood_benchmark/roc_lstm.png)
![roc_rnn.png](../../graphs/mood_benchmark/roc_rnn.png)
![roc_transformer_encoder.png](../../graphs/mood_benchmark/roc_transformer_encoder.png)
