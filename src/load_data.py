"""Utilities for loading CWRU bearing vibration `.mat` files.

Legacy helpers only. Use src.data_pipeline for manifest-driven ingestion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

try:
    from .data_pipeline import read_signal
except ImportError:
    from data_pipeline import read_signal


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



def find_vibration_key(mat_dict: dict) -> str:
    """Require exactly one DE channel for legacy single-file callers."""
    candidate_keys = [key for key in mat_dict.keys() if key.endswith("_DE_time")]
    if len(candidate_keys) != 1:
        raise KeyError("Expected exactly one DE_time key; use the M4 manifest for explicit channel selection.")
    return candidate_keys[0]


def load_mat_signal(file_path: str | Path) -> np.ndarray:
    """Load one CWRU .mat file and return a 1D vibration signal."""
    mat_data = sio.loadmat(file_path)
    signal_key = find_vibration_key(mat_data)
    return read_signal(file_path, signal_key)


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
