import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from .utils import load_data, shape_for_sequence, compute_metrics, save_metrics, get_fit_kwargs, set_global_seed


def build_model(timesteps: int, num_classes: int, target: str = 'mood') -> Model:
    inp = Input(shape=(timesteps, 1))
    x = Conv1D(128, 3, padding='same')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Deeper BiLSTM head (slightly larger for mood)
    head1_units = 128 if target == 'mood' else 96
    head2_units = 96 if target == 'mood' else 64
    x = Bidirectional(LSTM(head1_units, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(head2_units, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)
    return model


def _make_loss(target: str):
    if target == 'disease':
        # Sparse categorical focal loss
        def loss_fn(y_true, y_pred, gamma=2.0):
            y_true = tf.cast(y_true, tf.int32)
            y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
            # Avoid log(0)
            eps = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
            pt = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
            loss = -tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
            return tf.reduce_mean(loss)
        return loss_fn
    return 'sparse_categorical_crossentropy'


def train_and_evaluate(target: str = 'mood', seed: int | None = None):
    if seed is not None:
        set_global_seed(seed)
    include_aux_mood = (target == 'disease')
    X_train, X_test, y_train, y_test, encoder, class_names = load_data(
        target=target, include_mean=True, include_aux_mood=include_aux_mood)
    X_train_seq = shape_for_sequence(X_train)
    X_test_seq = shape_for_sequence(X_test)

    model = build_model(X_train_seq.shape[1], len(class_names), target=target)
    # Optimizer with weight decay and cosine restarts
    lr_schedule = CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=200, t_mul=2.0, m_mul=0.8, alpha=1e-4)
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss=_make_loss(target), metrics=['accuracy'])

    # Save best weights by validation accuracy to push top-line accuracy higher.
    weights_path = os.path.join(os.path.dirname(__file__), 'results', f"cnn_lstm_{target}.weights.h5")
    patience = 30 if target == 'mood' else 20
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True),
        ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    ]

    start = time.time()
    fit_kwargs = get_fit_kwargs(target, y_train)
    epochs = 300 if target == 'mood' else 200
    history = model.fit(
        X_train_seq, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
        **fit_kwargs
    )
    train_seconds = time.time() - start

    # Ensure we evaluate with the best weights (ModelCheckpoint wrote to weights_path)
    try:
        model.load_weights(weights_path)
    except Exception:
        pass
    y_prob = model.predict(X_test_seq, verbose=0)
    metrics = compute_metrics(y_test, y_prob, class_names)

    # Allow saving seed-specific metrics to keep multiple runs
    filename = None
    if seed is not None:
        filename = f"{target}_cnn-lstm_seed{seed}_metrics.json"
    out_path = save_metrics(
        model_name='CNN-LSTM',
        target=target,
        class_names=class_names,
        params=model.count_params(),
        train_seconds=train_seconds,
        metrics=metrics,
        extra={'val_best_loss': float(np.min(history.history.get('val_loss', [np.nan])))} ,
        filename=filename
    )
    base_dir = os.path.dirname(out_path)
    model_path = os.path.join(base_dir, f"cnn_lstm_{target}.h5")
    model.save(model_path)
    model.save_weights(os.path.join(base_dir, f"cnn_lstm_{target}.weights.h5"))
    return metrics

if __name__ == '__main__':
    train_and_evaluate()
