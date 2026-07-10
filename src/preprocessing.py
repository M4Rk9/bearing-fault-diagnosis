"""Signal preprocessing utilities for bearing vibration data."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def remove_mean(signal: np.ndarray) -> np.ndarray:
    """Remove DC offset from a vibration signal."""
    signal = np.asarray(signal, dtype=float)
    return signal - np.mean(signal)


def normalize_signal(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize a signal."""
    signal = np.asarray(signal, dtype=float)
    return (signal - np.mean(signal)) / (np.std(signal) + eps)


def segment_signal(signal: np.ndarray, window_size: int = 2048, overlap: float = 0.5) -> np.ndarray:
    """Split a 1D vibration signal into overlapping fixed-length windows.

    Args:
        signal: 1D vibration array.
        window_size: Number of samples per segment.
        overlap: Fractional overlap between consecutive windows.

    Returns:
        Array with shape (num_windows, window_size).
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the range [0, 1).")

    signal = np.asarray(signal, dtype=float).ravel()
    step = int(window_size * (1 - overlap))
    if step <= 0:
        raise ValueError("window_size and overlap produce an invalid step size.")

    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start : start + window_size])

    return np.asarray(windows)


def bandpass_filter(
    signal: np.ndarray,
    fs: int = 12000,
    lowcut: float = 20.0,
    highcut: float = 5000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    This is optional. Start without filtering, then add filtering during experiments.
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def preprocess_window(window: np.ndarray) -> np.ndarray:
    """Basic preprocessing for one signal window."""
    return normalize_signal(remove_mean(window))
