"""General plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def plot_waveform(signal: np.ndarray, fs: int = 12000):
    """Plot time-domain waveform."""
    time = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, signal)
    ax.set_title("Time-Domain Vibration Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    return fig


def plot_fft(signal: np.ndarray, fs: int = 12000):
    """Plot FFT magnitude spectrum."""
    fft_values = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, fft_values)
    ax.set_title("FFT Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    return fig


def plot_spectrogram(signal: np.ndarray, fs: int = 12000):
    """Plot STFT spectrogram."""
    f, t, sxx = spectrogram(signal, fs=fs)

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-12), shading="gouraud")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="Power/Frequency (dB/Hz)")
    return fig
