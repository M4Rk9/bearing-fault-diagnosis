"""Streamlit dashboard for bearing fault diagnosis."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from feature_extraction import extract_all_features
from preprocessing import preprocess_window
from utils import plot_fft, plot_spectrogram, plot_waveform


CLASS_NAMES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault",
}

st.set_page_config(page_title="Bearing Fault Diagnosis", layout="wide")

st.title("Bearing Fault Diagnosis from Vibration Signals")
st.write(
    "Upload a single-column CSV vibration signal or use a synthetic sample to visualize waveform, FFT, spectrogram, and model prediction."
)

uploaded_file = st.file_uploader("Upload vibration signal CSV", type=["csv"])
fs = st.sidebar.number_input("Sampling frequency (Hz)", min_value=1000, max_value=50000, value=12000, step=1000)
window_size = st.sidebar.number_input("Window size", min_value=512, max_value=8192, value=2048, step=512)


def load_signal_from_upload(file) -> np.ndarray:
    df = pd.read_csv(file)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("CSV must contain at least one numeric column.")
    return numeric_df.iloc[:, 0].dropna().to_numpy(dtype=float)


def synthetic_signal(size: int = 2048, fs: int = 12000) -> np.ndarray:
    time = np.arange(size) / fs
    base = 0.5 * np.sin(2 * np.pi * 50 * time)
    fault_impulses = np.zeros_like(time)
    fault_impulses[::150] = 2.0
    noise = 0.1 * np.random.default_rng(42).normal(size=size)
    return base + fault_impulses + noise


if uploaded_file is not None:
    try:
        signal = load_signal_from_upload(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()
else:
    st.info("No CSV uploaded. Showing a synthetic demo signal.")
    signal = synthetic_signal(size=int(window_size), fs=int(fs))

if len(signal) < window_size:
    st.error("Signal length is smaller than selected window size.")
    st.stop()

window = signal[: int(window_size)]
window = preprocess_window(window)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Time-Domain Waveform")
    st.pyplot(plot_waveform(window, fs=int(fs)))

with col2:
    st.subheader("FFT Spectrum")
    st.pyplot(plot_fft(window, fs=int(fs)))

st.subheader("Spectrogram")
st.pyplot(plot_spectrogram(window, fs=int(fs)))

st.subheader("Extracted Features")
features = extract_all_features(window, fs=int(fs))
features_df = pd.DataFrame([features])
st.dataframe(features_df, use_container_width=True)

st.subheader("Prediction")
model_path = PROJECT_ROOT / "models" / "random_forest.pkl"

if model_path.exists():
    model = joblib.load(model_path)
    prediction = int(model.predict(features_df)[0])
    st.success(f"Predicted condition: {CLASS_NAMES.get(prediction, prediction)}")
else:
    st.warning(
        "No trained model found at models/random_forest.pkl. Train the model first using `python src/train_ml.py`."
    )
