"""Utilities for loading CWRU bearing vibration `.mat` files.

Update FILE_LABEL_MAP with the exact filenames you download from CWRU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

try:
    from .preprocessing import preprocess_window, segment_signal
except ImportError:
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
    if len(candidate_keys) != 1:
        raise KeyError("Expected exactly one DE_time key; use the M4 manifest for explicit channel selection.")
    return candidate_keys[0]


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
    """Disabled legacy API: callers must preserve recording-level partitions."""
    raise RuntimeError(
        "Legacy window-level dataset loading is disabled: it loses recording partitions. "
        "Run python -m src.data_pipeline prepare --help and use the partitioned NPZ files. "
        "Model integration belongs to M5-M7."
    )


if __name__ == "__main__":
    X, y = load_dataset()
    print("Windows:", X.shape)
    print("Labels:", y.shape)
