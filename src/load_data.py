"""Utilities for loading CWRU bearing vibration `.mat` files.

Update FILE_LABEL_MAP with the exact filenames you download from CWRU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from preprocessing import preprocess_window, segment_signal


LABEL_MAP = {
    "normal": 0,
    "inner_race": 1,
    "ball": 2,
    "outer_race": 3,
}

CLASS_NAMES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault",
}

# Example mapping. Replace these with actual downloaded CWRU file names.
FILE_LABEL_MAP = {
    # "97.mat": "normal",
    # "105.mat": "inner_race",
    # "118.mat": "ball",
    # "130.mat": "outer_race",
}


def find_vibration_key(mat_dict: dict) -> str:
    """Find the likely drive-end vibration key in a CWRU .mat file."""
    candidate_keys = [key for key in mat_dict.keys() if "DE_time" in key]
    if candidate_keys:
        return candidate_keys[0]

    numeric_keys = [
        key
        for key, value in mat_dict.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.size > 1000
    ]
    if not numeric_keys:
        raise KeyError("No vibration signal key found in .mat file.")
    return numeric_keys[0]


def load_mat_signal(file_path: str | Path) -> np.ndarray:
    """Load one CWRU .mat file and return a 1D vibration signal."""
    mat_data = sio.loadmat(file_path)
    signal_key = find_vibration_key(mat_data)
    return mat_data[signal_key].ravel().astype(float)


def load_dataset(
    data_dir: str | Path = "data/raw",
    window_size: int = 2048,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load files listed in FILE_LABEL_MAP, segment them and return X, y."""
    data_dir = Path(data_dir)
    all_windows = []
    all_labels = []

    if not FILE_LABEL_MAP:
        raise ValueError(
            "FILE_LABEL_MAP is empty. Add your downloaded CWRU filenames and labels in src/load_data.py."
        )

    for filename, label_name in FILE_LABEL_MAP.items():
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        signal = load_mat_signal(file_path)
        windows = segment_signal(signal, window_size=window_size, overlap=overlap)
        windows = np.asarray([preprocess_window(w) for w in windows])
        labels = np.full(len(windows), LABEL_MAP[label_name])

        all_windows.append(windows)
        all_labels.append(labels)

    return np.vstack(all_windows), np.concatenate(all_labels)


if __name__ == "__main__":
    X, y = load_dataset()
    print("Windows:", X.shape)
    print("Labels:", y.shape)
