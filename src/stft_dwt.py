"""Compute time-aligned STFT and DWT features for vibration windows."""

import numpy as np
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import stft

SAMPLE_RATE = 12_000
WINDOW_SAMPLES = 2048
FRAME_SAMPLES = 512
FRAME_HOP = 64
FREQUENCY_BINS = FRAME_SAMPLES // 2 + 1
WAVELET = "db4"
DWT_LEVEL = 5
DWT_MODE = "symmetric"
DWT_BANDS = ("A5", "D5", "D4", "D3", "D2", "D1")
POWER_FLOOR = 1e-12


def extract_sequences(windows):
    """Return (windows, time frames, 257 STFT bins + 6 DWT bands).

    Input windows are mean-centered, amplitude-preserving samples at 12 kHz.
    Each DWT uses the same 512 raw samples as its corresponding Hann STFT
    frame. DWT bands contain mean squared coefficients, with no Hann taper.
    Both sets of powers are converted to dB relative to one squared unit.
    """
    windows = np.asarray(windows)
    if (windows.ndim != 2 or windows.shape[1] != WINDOW_SAMPLES
            or not len(windows) or windows.dtype.kind not in "iuf"
            or not np.isfinite(windows).all()):
        raise ValueError("Expected nonempty, finite real vibration windows of 2048 samples")

    _, _, spectrum = stft(
        windows, fs=SAMPLE_RATE, window="hann", nperseg=FRAME_SAMPLES,
        noverlap=FRAME_SAMPLES - FRAME_HOP, nfft=FRAME_SAMPLES,
        boundary=None, padded=False, scaling="spectrum",
    )
    stft_power = np.abs(spectrum.swapaxes(1, 2)) ** 2
    frames = sliding_window_view(windows, FRAME_SAMPLES, axis=-1)[:, ::FRAME_HOP]
    coeffs = pywt.wavedec(frames, WAVELET, mode=DWT_MODE, level=DWT_LEVEL, axis=-1)
    dwt_power = np.stack([np.mean(band ** 2, axis=-1) for band in coeffs], axis=-1)
    power = np.concatenate((stft_power, dwt_power), axis=-1)
    return (10 * np.log10(np.maximum(power, POWER_FLOOR))).astype(np.float32)
