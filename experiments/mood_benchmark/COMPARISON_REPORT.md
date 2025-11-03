# EEG Classification: Mood and Disease — Model Comparison Report

This report compares multiple deep learning methods trained on `data/processed_data.csv` for predicting targets from EEG features (eeg_1..eeg_14 plus eeg_mean).

- Targets covered: `mood` and `disease`
- Data split: stratified 75/25 train/test with random_state=42
- Metrics reported: Accuracy, F1 (macro), ROC-AUC (OvR); per-class precision/recall/f1 available inside each metrics JSON
- Models included: DNN Baseline, CNN 1D, RNN, LSTM, CNN-LSTM, Transformer Encoder, Ensemble SoftVote (DNN+CNN+LSTM) and Ensemble SoftVote (DNN+CNN+LSTM+RNN)

Generated on: 2025-11-03T08:57:18.639120Z


## Mood — Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (OvR) | Params | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 97.20% | 0.963 | 0.936 | 0.949 | 0.9981172752178636 | 121091 | 10.52 |
| CNN-LSTM | 96.40% | 0.965 | 0.906 | 0.930 | 0.993653053270194 | 50499 | 6.63 |
| CNN 1D | 96.40% | 0.965 | 0.906 | 0.930 | 0.998394014891489 | 17475 | 4.86 |
| Ensemble SoftVote (DNN+CNN+LSTM) | 95.50% | 0.955 | 0.881 | 0.909 | 0.9987572295369667 | 0 | 0.00 |
| Ensemble SoftVote (DNN+CNN+LSTM+RNN) | 94.80% | 0.951 | 0.860 | 0.891 | 0.9976721832628592 | 0 | 0.00 |
| DNN Baseline | 93.60% | 0.931 | 0.832 | 0.865 | 0.9912554153562237 | 47235 | 6.47 |
| RNN | 87.20% | 0.583 | 0.660 | 0.619 | 0.9384974035479742 | 34115 | 4.33 |
| Transformer Encoder | 68.00% | 0.449 | 0.496 | 0.470 | 0.7590349247523956 | 10915 | 5.35 |

### Mood — Head-to-head (top vs others)

Main method = LSTM (best F1 0.949, Accuracy 97.20%).

- LSTM vs CNN-LSTM: ΔAccuracy +0.80 pts, ΔF1 +0.019.
- LSTM vs CNN 1D: ΔAccuracy +0.80 pts, ΔF1 +0.019.
- LSTM vs Ensemble SoftVote (DNN+CNN+LSTM): ΔAccuracy +1.70 pts, ΔF1 +0.039.
- LSTM vs Ensemble SoftVote (DNN+CNN+LSTM+RNN): ΔAccuracy +2.40 pts, ΔF1 +0.057.
- LSTM vs DNN Baseline: ΔAccuracy +3.60 pts, ΔF1 +0.083.
- LSTM vs RNN: ΔAccuracy +10.00 pts, ΔF1 +0.329.
- LSTM vs Transformer Encoder: ΔAccuracy +29.20 pts, ΔF1 +0.478.

### Mood — Graphs

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

## Disease — Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (OvR) | Params | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN 1D | 44.00% | 0.299 | 0.308 | 0.297 | 0.7334044128730083 | 17735 | 5.74 |
| CNN-LSTM | 44.40% | 0.336 | 0.308 | 0.295 | 0.7332426344488604 | 447111 | 12.19 |
| DNN Baseline | 43.60% | 0.298 | 0.305 | 0.293 | 0.742645123382245 | 47495 | 4.71 |
| Ensemble SoftVote (DNN+CNN+LSTM+RNN) | 43.20% | 0.288 | 0.302 | 0.290 | 0.7417498248970128 | 0 | 0.00 |
| Transformer Encoder | 44.00% | 0.274 | 0.309 | 0.281 | 0.7486611798187888 | 11175 | 10.07 |
| DNN (mood_encoded) | 42.80% | 0.298 | 0.295 | 0.278 | 0.7221091591053211 | 184967 | 7.86 |
| RNN | 44.00% | 0.234 | 0.322 | 0.245 | 0.7283065199045738 | 34375 | 9.53 |
| LSTM | 42.40% | 0.234 | 0.282 | 0.208 | 0.7405312986642506 | 121351 | 9.15 |

### Disease — Head-to-head (top vs others)

Main method = CNN 1D (best F1 0.297, Accuracy 44.00%).

- CNN 1D vs CNN-LSTM: ΔAccuracy -0.40 pts, ΔF1 +0.002.
- CNN 1D vs DNN Baseline: ΔAccuracy +0.40 pts, ΔF1 +0.004.
- CNN 1D vs Ensemble SoftVote (DNN+CNN+LSTM+RNN): ΔAccuracy +0.80 pts, ΔF1 +0.007.
- CNN 1D vs Transformer Encoder: ΔAccuracy +0.00 pts, ΔF1 +0.016.
- CNN 1D vs DNN (mood_encoded): ΔAccuracy +1.20 pts, ΔF1 +0.019.
- CNN 1D vs RNN: ΔAccuracy +0.00 pts, ΔF1 +0.052.
- CNN 1D vs LSTM: ΔAccuracy +1.60 pts, ΔF1 +0.089.

### Disease — Graphs

![confusion_cnn_1d.png](../../graphs/disease_benchmark/confusion_cnn_1d.png)
![confusion_cnn_lstm.png](../../graphs/disease_benchmark/confusion_cnn_lstm.png)
![confusion_disease_dnn_moodencoded.png](../../graphs/disease_benchmark/confusion_disease_dnn_moodencoded.png)
![confusion_dnn_baseline.png](../../graphs/disease_benchmark/confusion_dnn_baseline.png)
![confusion_ensemble_dnn_cnn_lstm.png](../../graphs/disease_benchmark/confusion_ensemble_dnn_cnn_lstm.png)
![confusion_ensemble_dnn_cnn_lstm_rnn.png](../../graphs/disease_benchmark/confusion_ensemble_dnn_cnn_lstm_rnn.png)
![confusion_lstm.png](../../graphs/disease_benchmark/confusion_lstm.png)
![confusion_rnn.png](../../graphs/disease_benchmark/confusion_rnn.png)
![confusion_transformer_encoder.png](../../graphs/disease_benchmark/confusion_transformer_encoder.png)
![pr_cnn_1d.png](../../graphs/disease_benchmark/pr_cnn_1d.png)
![pr_cnn_lstm.png](../../graphs/disease_benchmark/pr_cnn_lstm.png)
![pr_disease_dnn_moodencoded.png](../../graphs/disease_benchmark/pr_disease_dnn_moodencoded.png)
![pr_dnn_baseline.png](../../graphs/disease_benchmark/pr_dnn_baseline.png)
![pr_ensemble_dnn_cnn_lstm.png](../../graphs/disease_benchmark/pr_ensemble_dnn_cnn_lstm.png)
![pr_ensemble_dnn_cnn_lstm_rnn.png](../../graphs/disease_benchmark/pr_ensemble_dnn_cnn_lstm_rnn.png)
![pr_lstm.png](../../graphs/disease_benchmark/pr_lstm.png)
![pr_rnn.png](../../graphs/disease_benchmark/pr_rnn.png)
![pr_transformer_encoder.png](../../graphs/disease_benchmark/pr_transformer_encoder.png)
![roc_cnn_1d.png](../../graphs/disease_benchmark/roc_cnn_1d.png)
![roc_cnn_lstm.png](../../graphs/disease_benchmark/roc_cnn_lstm.png)
![roc_disease_dnn_moodencoded.png](../../graphs/disease_benchmark/roc_disease_dnn_moodencoded.png)
![roc_dnn_baseline.png](../../graphs/disease_benchmark/roc_dnn_baseline.png)
![roc_ensemble_dnn_cnn_lstm.png](../../graphs/disease_benchmark/roc_ensemble_dnn_cnn_lstm.png)
![roc_ensemble_dnn_cnn_lstm_rnn.png](../../graphs/disease_benchmark/roc_ensemble_dnn_cnn_lstm_rnn.png)
![roc_lstm.png](../../graphs/disease_benchmark/roc_lstm.png)
![roc_rnn.png](../../graphs/disease_benchmark/roc_rnn.png)
![roc_transformer_encoder.png](../../graphs/disease_benchmark/roc_transformer_encoder.png)

## Model Notes

### CNN-LSTM (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.0860

### CNN 1D (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.0686

### CNN-LSTM (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 1.3407

### CNN-LSTM (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 0.9964

### CNN-LSTM (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 0.9648

### CNN-LSTM (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 0.9873

### CNN 1D (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 1.3232

### DNN Baseline (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 1.3878

### DNN (mood_encoded) (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 1.3842

### Ensemble SoftVote (DNN+CNN+LSTM+RNN) (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Ensemble base models: dnn_baseline_disease.h5, cnn_1d_disease.h5, lstm_disease.h5, rnn_disease.h5

### LSTM (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 1.2980

### RNN (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 1.3060

### Transformer Encoder (disease)

- Classes: alzheimer, brain_stroke, brain_tumor, epilepsy, migraine, none, parkinson
- Best validation loss: 1.3023

### DNN Baseline (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.1427

### Ensemble SoftVote (DNN+CNN+LSTM) (mood)

- Classes: happy, neutral, sad
- Ensemble base models: dnn_baseline.h5, cnn_1d.h5, lstm.h5

### Ensemble SoftVote (DNN+CNN+LSTM+RNN) (mood)

- Classes: happy, neutral, sad
- Ensemble base models: dnn_baseline.h5, cnn_1d.h5, lstm.h5, rnn.h5

### LSTM (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.0572

### CNN-LSTM (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.0919

### CNN-LSTM (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.1468

### CNN-LSTM (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.1285

### CNN-LSTM (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.1553

### CNN 1D (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.0648

### DNN Baseline (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.1118

### Ensemble SoftVote (DNN+CNN+LSTM+RNN) (mood)

- Classes: happy, neutral, sad
- Ensemble base models: dnn_baseline_mood.h5, cnn_1d_mood.h5, lstm_mood.h5, rnn_mood.h5

### LSTM (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.0444

### RNN (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.2455

### Transformer Encoder (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.7213

### RNN (mood)

- Classes: happy, neutral, sad
- Best validation loss: 0.2953

### Transformer Encoder (mood)

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

## Overall recommendations

- Default choice: Favor the best-performing method per target above; CNN-LSTM and LSTM are often strong on these EEG features.
- Constrained devices: CNN 1D for competitive accuracy with markedly fewer parameters.
- Quick baselines/ablations: DNN and SimpleRNN for speed and sanity checks.
- With more data/compute: Revisit Transformer with stronger regularization/tuning.
- Reliability-first: Keep SoftVote ensemble for robustness and stable probabilities.
