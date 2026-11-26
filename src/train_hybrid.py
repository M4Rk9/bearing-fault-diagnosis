"""Train a parallel STFT/DWT CNN + Envelope LSTM on aligned CWRU windows."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .data_pipeline import CLASSES, PARTITIONS, digest, raw_path, read_signal
from .feature_dataset import MANIFEST, RAW_DIR, REPO_ROOT, VARIANTS
from .stft_dwt import extract_sequences, SAMPLE_RATE, WINDOW_SAMPLES

DRIVE_CLASSES = {
    "normal": "Healthy", "inner_race": "Inner_Race_IR",
    "ball": "Ball_BR", "outer_race": "Outer_Race_OR_IO",
}
STORAGE_SPLITS = ("train_70", "test_30")
CNN_SHAPE = (25, 263)
ENV_SHAPE = (25, 68)
SEED = 42


def load_array(path, shape):
    x = np.load(path, allow_pickle=False, mmap_mode="r")
    if x.shape != shape or x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError(f"Invalid array: {path}, expected {shape}, got {x.shape}")
    return x


def verify_clean(signal, rows, records, raw_dir, cache):
    """Check Drive's clean windows against the manifest-pinned GitHub MAT files."""
    for record_id in sorted({r["record_id"] for r in rows}):
        record = records[record_id]
        if record_id not in cache:
            path = raw_path(raw_dir, record)
            if digest(path) != record["sha256"]:
                raise ValueError(f"Raw recording hash mismatch: {path}")
            raw = read_signal(path, record["channel_key"])
            source_rate = record["sampling_rate_hz"]
            if source_rate != SAMPLE_RATE:
                g = math.gcd(source_rate, SAMPLE_RATE)
                raw = resample_poly(raw, SAMPLE_RATE // g, source_rate // g,
                                    window=("kaiser", 5.0), padtype="constant")
            cache[record_id] = raw
        raw = cache[record_id]
        for i, row in enumerate(rows):
            if row["record_id"] != record_id:
                continue
            start = int(row["start_sample"])
            window = raw[start:start + WINDOW_SAMPLES]
            if len(window) != WINDOW_SAMPLES:
                raise ValueError(f"Window outside recording: {record_id}:{start}")
            window = window - window.mean()
            if not np.allclose(signal[i], window, rtol=1e-4, atol=2e-6):
                raise ValueError(f"Clean window does not match GitHub raw recording: {record_id}:{start}")


def load_hybrid_partitions(root, raw_dir=RAW_DIR):
    """Regroup both Drive storage folders by the original recording partitions.

    STFT/DWT are regenerated with the GitHub extractor from the exact signal
    windows associated with Envelope frames, including the Drive noisy realization.
    Never zip an independently exported feature dataset by row order.
    """
    records = {r["record_id"]: r for r in json.loads(MANIFEST.read_text())}
    root = Path(root)
    parts = {p: {v: [] for v in VARIANTS} for p in PARTITIONS}
    seen, groups, cache = set(), {}, {}
    for storage in STORAGE_SPLITS:
        for name, folder in DRIVE_CLASSES.items():
            path = root / storage / folder
            with (path / "index.csv").open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(f"Empty index: {path}")
            splits, group_ids = [], []
            for row in rows:
                record = records[row["record_id"]]
                start = int(row["start_sample"])
                identity = (row["record_id"], start)
                split = record["partition"]
                group = record["group_id"]
                if (start < 0 or identity in seen or record["class_name"] != name
                        or row["group_id"] != group or row["partition"] != split
                        or row["split_70_30"] != storage
                        or groups.setdefault(group, split) != split):
                    raise ValueError(f"Duplicate or inconsistent recording metadata: {identity}")
                seen.add(identity)
                splits.append(split)
                group_ids.append(group)
            splits, group_ids = np.asarray(splits), np.asarray(group_ids)
            n = len(rows)
            for variant in VARIANTS:
                signal = load_array(path / variant / "signal.npy", (n, WINDOW_SAMPLES))
                envelope = load_array(path / variant / "envelope_frames.npy", (n, *ENV_SHAPE))
                if variant == "Clean":
                    verify_clean(signal, rows, records, raw_dir, cache)
                cnn = np.concatenate([
                    extract_sequences(signal[i:i+64]) for i in range(0, n, 64)
                ])
                if cnn.shape != (n, *CNN_SHAPE):
                    raise ValueError(f"Unexpected STFT/DWT shape: {cnn.shape}")
                for split in PARTITIONS:
                    mask = splits == split
                    if mask.any():
                        parts[split][variant].append((
                            cnn[mask], np.asarray(envelope[mask], dtype=np.float32),
                            np.full(mask.sum(), CLASSES[name], dtype=np.int64), group_ids[mask],
                        ))
    if set(groups) != {r["group_id"] for r in records.values()}:
        raise ValueError("Drive data does not cover the full locked cohort")
    result = {}
    for split, variants in parts.items():
        result[split] = {}
        for variant, entries in variants.items():
            if not entries:
                raise ValueError(f"No {split}/{variant} data")
            cnn, envelope, labels, group_ids = (np.concatenate(col) for col in zip(*entries))
            if set(labels) != set(CLASSES.values()):
                raise ValueError(f"Missing classes in {split}/{variant}")
            result[split][variant] = (
                {"cnn_features": cnn, "envelope_features": envelope}, labels, group_ids)
    return result


def combine(variants):
    x = {key: np.concatenate([variants[v][0][key] for v in VARIANTS])
         for key in ("cnn_features", "envelope_features")}
    y = np.concatenate([variants[v][1] for v in VARIANTS])
    return x, y


def build_hybrid(x_train):
    import tensorflow as tf

    layers = tf.keras.layers
    cnn_input = layers.Input(CNN_SHAPE, name="cnn_features")
    env_input = layers.Input(ENV_SHAPE, name="envelope_features")
    cnn_norm = layers.Normalization(axis=-1, name="cnn_normalization")
    env_norm = layers.Normalization(axis=-1, name="envelope_normalization")
    cnn_norm.adapt(x_train["cnn_features"])
    env_norm.adapt(x_train["envelope_features"])
    cnn = layers.Conv1D(32, 3, padding="same", activation="relu")(cnn_norm(cnn_input))
    cnn = layers.MaxPooling1D(2)(cnn)
    cnn = layers.Conv1D(64, 3, padding="same", activation="relu")(cnn)
    cnn = layers.GlobalAveragePooling1D()(cnn)
    env = layers.LSTM(64)(env_norm(env_input))
    env = layers.Dropout(0.3)(env)
    fused = layers.Concatenate(name="feature_fusion")([cnn, env])
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dropout(0.4)(fused)
    output = layers.Dense(len(CLASSES), activation="softmax")(fused)
    model = tf.keras.Model([cnn_input, env_input], output, name="parallel_cnn_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Extracted Drive folder containing train_70 and test_30")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models" / "hybrid")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--baseline", action="store_true",
                        help="Also train the original 1D CNN on exactly the same windows")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    partitions = load_hybrid_partitions(args.dataset, args.raw_dir)
    counts = {s: {v: {"windows": len(p[1]), "groups": len(np.unique(p[2]))}
                  for v, p in variants.items()} for s, variants in partitions.items()}
    print(json.dumps(counts, indent=2))
    if args.check_only:
        return

    import tensorflow as tf
    from .train_cnn import evaluate_test, fit_model

    tf.keras.utils.set_random_seed(SEED)
    x_train, y_train = combine(partitions["train"])
    x_val, y_val = combine(partitions["validation"])
    model = build_hybrid(x_train)
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=args.epochs, batch_size=64, callbacks=callbacks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "stft_dwt_envelope_cnn_lstm.keras")
    results = {
        "seed": SEED, "tensorflow_version": tf.__version__,
        "manifest_sha256": digest(MANIFEST), "class_ids": CLASSES,
        "partition_counts": counts,
        "history": {k: [float(a) for a in v] for k, v in history.history.items()},
        "hybrid": {v: evaluate_test(model, partitions["test"][v]) for v in VARIANTS},
    }
    if args.baseline:
        baseline_parts = {s: {v: (p[0]["cnn_features"], p[1], p[2])
                             for v, p in variants.items()} for s, variants in partitions.items()}
        baseline, baseline_history = fit_model(baseline_parts, epochs=args.epochs)
        baseline.save(args.output_dir / "stft_dwt_baseline.keras")
        results["baseline"] = {v: evaluate_test(baseline, baseline_parts["test"][v])
                               for v in VARIANTS}
        results["baseline_history"] = {k: [float(a) for a in v]
                                       for k, v in baseline_history.history.items()}
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("hybrid", "baseline") if k in results}, indent=2))


if __name__ == "__main__":
    main()
