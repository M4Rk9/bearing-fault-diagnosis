"""Export and load paired clean/noisy bearing signals and their STFT/DWT features."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .data_pipeline import CLASSES, PARTITIONS, prepare
from .stft_dwt import (
    DWT_BANDS, DWT_LEVEL, DWT_MODE, FRAME_HOP, FRAME_SAMPLES, FREQUENCY_BINS,
    POWER_FLOOR, SAMPLE_RATE, WAVELET, WINDOW_SAMPLES, extract_sequences,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "Datasets" / "RAW" / "CWRU Dataset"
FEATURES_DIR = REPO_ROOT / "Datasets" / "Features"
MANIFEST = REPO_ROOT / "data" / "manifests" / "cwru_uploaded.lock.json"
CLASS_FOLDERS = {
    "normal": "Healthy", "inner_race": "Inner Race",
    "ball": "Ball", "outer_race": "Outer Race",
}
VARIANTS = ("Clean", "Noisy")
INDEX_COLUMNS = ("partition", "record_id", "group_id", "start_sample")
WINDOW_HOP = WINDOW_SAMPLES // 2
FRAME_COUNT = 1 + (WINDOW_SAMPLES - FRAME_SAMPLES) // FRAME_HOP
DEFAULT_SNR_DB = 10.0
DEFAULT_NOISE_SEED = 42
SCHEMA = "cwru-stft-dwt-paired-v1"


def prepare_uploaded_cohort(raw_dir, output):
    """Validate the uploaded recordings and create grouped vibration windows."""
    rows = json.loads(MANIFEST.read_text())
    return prepare(rows, raw_dir, output, target_rate=SAMPLE_RATE,
                   window_size=WINDOW_SAMPLES, hop_size=WINDOW_HOP)


def add_recording_noise(clean, starts, snr_db, rng):
    """Add one Gaussian noise stream sliced at the recording's window offsets."""
    noise_stream = rng.standard_normal(int(starts.max()) + WINDOW_SAMPLES)
    noise = sliding_window_view(noise_stream, WINDOW_SAMPLES)[starts]
    noise -= noise.mean(axis=1, keepdims=True)
    target_ratio = 10 ** (snr_db / 10)
    gain = np.sqrt(np.mean(clean ** 2) / (target_ratio * np.mean(noise ** 2)))
    return clean + gain * noise


def export_dataset(raw_dir, output, snr_db=DEFAULT_SNR_DB, seed=DEFAULT_NOISE_SEED):
    """Build the four class folders. Publish dataset.json after all arrays are written."""
    if not np.isfinite(snr_db) or not -40 <= snr_db <= 60 or seed < 0:
        raise ValueError("Use a noise SNR between -40 and 60 dB and a nonnegative seed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # An exclusive marker prevents two exporters from mixing their arrays.
    lock = output / ".export.lock"
    with lock.open("x"):
        pass
    try:
        if (output / "dataset.json").exists():
            raise FileExistsError(f"A feature dataset already exists at {output}. Use another --output path.")
        with TemporaryDirectory(dir=output) as temporary:
            staging = Path(temporary)
            summary = prepare_uploaded_cohort(raw_dir, staging / "prepared")
            rng = np.random.default_rng(seed)
            by_class = {label: [] for label in CLASSES.values()}
            for partition in PARTITIONS:
                with np.load(staging / "prepared" / f"{partition}.npz", allow_pickle=False) as data:
                    for group_id in np.unique(data["group_ids"]):
                        mask = data["group_ids"] == group_id
                        label = int(data["labels"][mask][0])
                        clean = data["feature_windows"][mask]
                        starts = data["start_samples"][mask]
                        noisy = add_recording_noise(clean, starts, snr_db, rng)
                        rows = [
                            (partition, record_id, group_id, start)
                            for record_id, start in zip(data["record_ids"][mask], starts)
                        ]
                        by_class[label].append((rows, clean, noisy))

            for class_name, label in CLASSES.items():
                folder = staging / CLASS_FOLDERS[class_name]
                folder.mkdir()
                items = by_class[label]
                with (folder / "index.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(INDEX_COLUMNS)
                    for rows, _, _ in items:
                        writer.writerows(rows)
                for column, variant in enumerate(VARIANTS, start=1):
                    signal = np.concatenate([item[column] for item in items]).astype(np.float32)
                    features = extract_sequences(signal)
                    target = folder / variant
                    target.mkdir()
                    np.savez_compressed(target / "signal.npz", values=signal)
                    np.savez_compressed(target / "stft.npz", values=features[..., :FREQUENCY_BINS])
                    np.savez_compressed(target / "dwt.npz", values=features[..., FREQUENCY_BINS:])

            metadata = {
                "schema": SCHEMA,
                "classes": {name: {"id": CLASSES[name], "folder": folder}
                            for name, folder in CLASS_FOLDERS.items()},
                "manifest_sha256": summary["manifest_sha256"],
                "sample_rate_hz": SAMPLE_RATE,
                "window_samples": WINDOW_SAMPLES,
                "window_hop": WINDOW_HOP,
                "frame_samples": FRAME_SAMPLES,
                "frame_hop": FRAME_HOP,
                "stft_window": "Hann",
                "stft_bins": FREQUENCY_BINS,
                "dwt_wavelet": WAVELET,
                "dwt_level": DWT_LEVEL,
                "dwt_mode": DWT_MODE,
                "dwt_bands": DWT_BANDS,
                "power_floor": POWER_FLOOR,
                "noise": {"kind": "additive white Gaussian", "snr_db": snr_db, "seed": seed},
                "partitions": summary["partitions"],
                "records": summary["records"],
            }
            (staging / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
            for folder_name in CLASS_FOLDERS.values():
                folder = staging / folder_name
                for path in folder.rglob("*"):
                    if path.is_file():
                        destination = output / path.relative_to(staging)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        path.replace(destination)
            (staging / "dataset.json").replace(output / "dataset.json")
        return metadata
    finally:
        lock.unlink(missing_ok=True)


def read_index(folder):
    with (folder / "index.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(INDEX_COLUMNS):
            raise ValueError(f"Unexpected index columns in {folder}")
        return list(reader)


def read_array(folder, name):
    with np.load(folder / f"{name}.npz", allow_pickle=False) as data:
        return data["values"]


def load_partitions(dataset):
    """Read numeric features from disk and verify that recording groups stay together."""
    dataset = Path(dataset)
    metadata_path = dataset / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No complete dataset at {dataset}. Run python -m src.feature_dataset first.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported feature dataset format: {metadata_path}")

    parts = {split: {variant: [] for variant in VARIANTS} for split in PARTITIONS}
    groups_seen, windows_seen = {}, set()
    for class_name, label in CLASSES.items():
        folder = dataset / CLASS_FOLDERS[class_name]
        rows = read_index(folder)
        splits = np.asarray([row["partition"] for row in rows])
        groups = np.asarray([row["group_id"] for row in rows])
        for row in rows:
            group, split = row["group_id"], row["partition"]
            identity = (row["record_id"], int(row["start_sample"]))
            if (not group or not identity[0] or split not in PARTITIONS
                    or identity[1] < 0 or identity in windows_seen
                    or groups_seen.setdefault(group, (split, label)) != (split, label)):
                raise ValueError(f"Invalid split or duplicate recording window in {folder / 'index.csv'}")
            windows_seen.add(identity)
        for variant in VARIANTS:
            path = folder / variant
            stft = read_array(path, "stft")
            dwt = read_array(path, "dwt")
            if (stft.shape != (len(rows), FRAME_COUNT, FREQUENCY_BINS)
                    or dwt.shape != (len(rows), FRAME_COUNT, len(DWT_BANDS))
                    or stft.dtype != np.float32 or dwt.dtype != np.float32
                    or not np.isfinite(stft).all() or not np.isfinite(dwt).all()):
                raise ValueError(f"Invalid or misaligned feature arrays in {path}")
            features = np.concatenate((stft, dwt), axis=-1)
            for split, variants in parts.items():
                mask = splits == split
                if not mask.any():
                    raise ValueError(f"{folder.name} has no {split} windows")
                variants[variant].append((features[mask], np.full(mask.sum(), label), groups[mask]))
    return {
        split: {variant: tuple(np.concatenate(columns) for columns in zip(*entries))
                for variant, entries in variants.items()}
        for split, variants in parts.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FEATURES_DIR)
    parser.add_argument("--snr-db", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--seed", type=int, default=DEFAULT_NOISE_SEED)
    args = parser.parse_args()
    metadata = export_dataset(RAW_DIR, args.output, args.snr_db, args.seed)
    print("Window pairs:", {p: metadata["partitions"][p]["windows"] for p in PARTITIONS})
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
