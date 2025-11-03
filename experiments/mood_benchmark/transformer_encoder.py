import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .utils import load_data, shape_for_sequence, compute_metrics, save_metrics, get_fit_kwargs


def transformer_block(x, num_heads=2, key_dim=16, ff_dim=64, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dropout(rate)(ff)
    ff = Dense(x.shape[-1], activation=None)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_model(timesteps: int, num_classes: int) -> Model:
    inp = Input(shape=(timesteps, 1))
    # Project to a slightly higher dimension for attention
    x = Dense(32)(inp)
    x = transformer_block(x, num_heads=2, key_dim=16, ff_dim=64, rate=0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(target: str = 'mood'):
    X_train, X_test, y_train, y_test, encoder, class_names = load_data(target=target, include_mean=True)
    X_train_seq = shape_for_sequence(X_train)
    X_test_seq = shape_for_sequence(X_test)

    model = build_model(X_train_seq.shape[1], len(class_names))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    start = time.time()
    fit_kwargs = get_fit_kwargs(target, y_train)
    history = model.fit(
        X_train_seq, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
        **fit_kwargs
    )
    train_seconds = time.time() - start

    y_prob = model.predict(X_test_seq, verbose=0)
    metrics = compute_metrics(y_test, y_prob, class_names)

    out_path = save_metrics(
        model_name='Transformer Encoder',
        target=target,
        class_names=class_names,
        params=model.count_params(),
        train_seconds=train_seconds,
        metrics=metrics,
        extra={'val_best_loss': float(np.min(history.history['val_loss']))}
    )
    base_dir = os.path.dirname(out_path)
    model_path = os.path.join(base_dir, f"transformer_encoder_{target}.h5")
    model.save(model_path)
    model.save_weights(os.path.join(base_dir, f"transformer_encoder_{target}.weights.h5"))
    return metrics

if __name__ == '__main__':
    train_and_evaluate()
