# Project Plan: Bearing Fault Diagnosis from Vibration Signals

## Objective

Classify bearing condition from vibration signals using signal processing, classical ML, and deep learning.

## Classification Labels

| Label | Class |
|---:|---|
| 0 | Normal |
| 1 | Inner Race Fault |
| 2 | Ball Fault |
| 3 | Outer Race Fault |

## Phase 1: Dataset Setup

- Download CWRU Bearing Data Center `.mat` files.
- Start with 12 kHz drive-end data.
- Select normal, inner-race fault, ball fault, and outer-race fault files.
- Place files in `data/raw/`.
- Update `FILE_LABEL_MAP` in `src/load_data.py`.

## Phase 2: Signal Processing

- Load `.mat` files.
- Extract drive-end vibration signal.
- Segment long signals into fixed windows.
- Use window size of 2048 samples and 50% overlap.
- Normalize every window.

## Phase 3: Feature Engineering

Extract:

- RMS
- Kurtosis
- Skewness
- Crest factor
- FFT spectral energy
- Dominant frequency
- Envelope-spectrum peaks

## Phase 4: Baseline ML Models

Train:

- Random Forest
- Support Vector Machine

Evaluate using:

- Accuracy
- Macro-F1
- Confusion matrix

## Phase 5: Deep Learning Extension

Train:

- 1D-CNN on raw vibration windows

Optional:

- 2D-CNN on spectrogram/STFT images

## Phase 6: Robustness Testing

Perform two experiments:

1. Random train-test split.
2. Load-wise split, such as training on 0 HP, 1 HP, 2 HP and testing on 3 HP.

## Phase 7: Dashboard

Build a Streamlit dashboard showing:

- Uploaded vibration signal
- Time-domain waveform
- FFT spectrum
- Spectrogram
- Extracted features
- Predicted bearing condition

## Phase 8: Final Deliverables

- Code repository
- Literature survey
- Feature extraction pipeline
- Trained ML models
- Evaluation results
- Dashboard
- Final report
- Presentation
