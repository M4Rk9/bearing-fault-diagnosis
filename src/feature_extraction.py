"""Feature extraction for bearing fault diagnosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew


def extract_time_features(window: np.ndarray) -> dict[str, float]:
    """Extract time-domain vibration features from one window."""
    x = np.asarray(window, dtype=float).ravel()
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "variance": float(np.var(x)),
        "rms": float(rms),
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),
        "peak": float(peak),
        "peak_to_peak": float(np.ptp(x)),
        "crest_factor": float(peak / (rms + 1e-8)),
    }


def extract_frequency_features(window: np.ndarray, fs: int = 12000) -> dict[str, float]:
    """Extract frequency-domain features using FFT."""
    x = np.asarray(window, dtype=float).ravel()
    fft_values = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)

    spectral_energy = np.sum(fft_values**2)
    spectral_centroid = np.sum(freqs * fft_values) / (np.sum(fft_values) + 1e-8)
    dominant_idx = int(np.argmax(fft_values))

    return {
        "dominant_frequency": float(freqs[dominant_idx]),
        "max_fft_amplitude": float(fft_values[dominant_idx]),
        "spectral_energy": float(spectral_energy),
        "spectral_centroid": float(spectral_centroid),
    }


def extract_envelope_features(window: np.ndarray, fs: int = 12000) -> dict[str, float]:
    """Extract simple envelope-spectrum features.

    Envelope analysis is useful for weak impulsive fault signatures.
    """
    x = np.asarray(window, dtype=float).ravel()
    envelope = np.abs(hilbert(x))
    envelope_fft = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), d=1 / fs)
    peak_idx = int(np.argmax(envelope_fft[1:]) + 1) if len(envelope_fft) > 1 else 0

    return {
        "envelope_peak_frequency": float(freqs[peak_idx]),
        "envelope_peak_amplitude": float(envelope_fft[peak_idx]),
        "envelope_energy": float(np.sum(envelope_fft**2)),
    }


def extract_all_features(window: np.ndarray, fs: int = 12000) -> dict[str, float]:
    """Extract all engineered features from one vibration window."""
    features = {}
    features.update(extract_time_features(window))
    features.update(extract_frequency_features(window, fs=fs))
    features.update(extract_envelope_features(window, fs=fs))
    return features


def build_feature_table(windows: np.ndarray, labels: np.ndarray, fs: int = 12000) -> pd.DataFrame:
    """Convert signal windows and labels into a feature DataFrame."""
    rows = []
    for window, label in zip(windows, labels):
        row = extract_all_features(window, fs=fs)
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)
