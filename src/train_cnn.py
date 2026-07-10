"""Train a simple 1D-CNN on raw vibration windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from load_data import load_dataset


RANDOM_STATE = 42
MODELS_DIR = Path("models")


def build_1d_cnn(input_shape: tuple[int, int], num_classes: int = 4) -> tf.keras.Model:
    """Build a compact 1D-CNN for raw vibration windows."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=7, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=5, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(128, kernel_size=3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    windows, labels = load_dataset()
    X = windows[..., np.newaxis]
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_1d_cnn(input_shape=X_train.shape[1:])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    model.save(MODELS_DIR / "cnn_1d_model.keras")
    print("Saved 1D-CNN model to models/cnn_1d_model.keras")


if __name__ == "__main__":
    main()
