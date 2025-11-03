import os
from . import cnn_lstm

SEEDS = [42, 1337, 2025]

def main():
    for tgt in ['mood', 'disease']:
        for s in SEEDS:
            print(f"Running CNN-LSTM with seed={s} for target={tgt}...")
            cnn_lstm.train_and_evaluate(target=tgt, seed=s)
    print("CNN-LSTM multi-seed runs complete.")

if __name__ == '__main__':
    main()
