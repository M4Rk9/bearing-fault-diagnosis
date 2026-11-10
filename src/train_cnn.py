"""Train the STFT + DWT 1D-CNN on the exported clean and noisy features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data_pipeline import CLASSES
from .feature_dataset import FEATURES_DIR, REPO_ROOT, VARIANTS, load_partitions

RANDOM_STATE = 42
MODEL_PATH = REPO_ROOT / "models" / "stft_dwt_clean_noisy_1d_cnn.keras"


def build_1d_cnn(input_shape, normalization, num_classes=len(CLASSES)):
    """Adapt the instructor's two convolution/pooling blocks to feature frames."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        normalization,
        tf.keras.layers.Conv1D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_model(partitions, epochs=10, verbose=2):
    """Fit on both training variants and select weights using both validation variants."""
    x_train, y_train = (np.concatenate([partitions["train"][variant][column] for variant in VARIANTS])
                        for column in (0, 1))
    x_val, y_val = (np.concatenate([partitions["validation"][variant][column] for variant in VARIANTS])
                    for column in (0, 1))
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    normalization = tf.keras.layers.Normalization(axis=-1)
    normalization.adapt(x_train)
    model = build_1d_cnn(x_train.shape[1:], normalization)
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=64, verbose=verbose,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True,
        )],
    )
    return model, history


def evaluate_test(model, test_partition):
    """Measure held-out windows and vote by mean probability per recording."""
    x_test, y_test, groups = test_partition
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    classes = list(CLASSES)
    label_ids = list(CLASSES.values())

    def metrics(actual, predicted):
        return {
            "count": len(actual),
            "accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, labels=label_ids,
                                 average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=label_ids).tolist(),
        }

    record_actual, record_predicted, records = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        actual = int(y_test[mask][0])
        predicted = int(probabilities[mask].mean(axis=0).argmax())
        record_actual.append(actual)
        record_predicted.append(predicted)
        records.append({
            "group_id": str(group), "class": classes[actual],
            "predicted": classes[predicted], "windows": int(mask.sum()),
        })

    return {
        "class_order": classes,
        "windows": metrics(y_test, predictions),
        "recordings": metrics(record_actual, record_predicted),
        "record_predictions": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=FEATURES_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Output .keras model")
    args = parser.parse_args()
    if args.epochs < 1 or args.output.suffix != ".keras":
        parser.error("Use a positive epoch count and an output ending in .keras")

    partitions = load_partitions(args.dataset)
    print("Window pairs:", {split: len(variants["Clean"][1])
                            for split, variants in partitions.items()})
    model, _ = fit_model(partitions, epochs=args.epochs)
    results = {variant: evaluate_test(model, partitions["test"][variant]) for variant in VARIANTS}
    print(json.dumps(results, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
