import os
from . import dnn_baseline, cnn_1d, lstm_model, rnn_model, cnn_lstm, transformer_encoder, ensemble_softvote


def run_for(target: str):
    print(f"Running DNN Baseline ({target})...")
    dnn_baseline.train_and_evaluate(target=target)

    print(f"Running CNN 1D ({target})...")
    cnn_1d.train_and_evaluate(target=target)

    print(f"Running LSTM ({target})...")
    lstm_model.train_and_evaluate(target=target)

    print(f"Running RNN ({target})...")
    rnn_model.train_and_evaluate(target=target)

    print(f"Running CNN-LSTM ({target})...")
    cnn_lstm.train_and_evaluate(target=target)

    print(f"Running Transformer Encoder ({target})...")
    transformer_encoder.train_and_evaluate(target=target)

    print(f"Running Ensemble SoftVote ({target})...")
    ensemble_softvote.train_and_evaluate(target=target)


def main():
    for tgt in ['mood', 'disease']:
        run_for(tgt)
    print("All experiments complete.")


if __name__ == '__main__':
    main()
