import os
from . import dnn_baseline, cnn_1d, lstm_model, rnn_model, cnn_lstm, transformer_encoder, ensemble_softvote


def main():
    print("Running DNN Baseline...")
    dnn_baseline.train_and_evaluate(target='mood')

    print("Running CNN 1D...")
    cnn_1d.train_and_evaluate(target='mood')

    print("Running LSTM...")
    lstm_model.train_and_evaluate(target='mood')

    print("Running RNN...")
    rnn_model.train_and_evaluate(target='mood')

    print("Running CNN-LSTM...")
    cnn_lstm.train_and_evaluate(target='mood')

    print("Running Transformer Encoder...")
    transformer_encoder.train_and_evaluate(target='mood')

    print("Running Ensemble SoftVote...")
    ensemble_softvote.train_and_evaluate(target='mood')

    print("All experiments complete.")


if __name__ == '__main__':
    main()
