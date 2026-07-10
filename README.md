# Bearing Fault Diagnosis from Vibration Signals

A fifth-semester AIML minor project for classifying bearing health from vibration signals using signal processing, classical machine learning, and deep learning.

## Problem Statement

The aim is to classify whether a bearing is healthy or faulty using measured vibration signals. If faulty, the system identifies the fault type:

- Normal / Healthy
- Inner race fault
- Ball fault
- Outer race fault

The project uses vibration signal windows as inputs and predicts the corresponding bearing condition as a supervised classification task.

## Dataset

Primary dataset:

- Case Western Reserve University Bearing Data Center dataset
- Recommended starting point: 12 kHz drive-end bearing data
- Classes: normal, inner race, ball, outer race
- Loads: 0 HP, 1 HP, 2 HP, 3 HP

Optional external validation:

- NASA IMS bearing dataset

> Note: Dataset files are not committed to GitHub. Place downloaded `.mat` files inside `data/raw/` locally.

## Project Pipeline

```text
Raw CWRU .mat files
        ↓
Signal loading
        ↓
Window segmentation
        ↓
Preprocessing / normalization
        ↓
Time-domain and frequency-domain feature extraction
        ↓
SVM / Random Forest baseline models
        ↓
1D-CNN on raw vibration windows
        ↓
Evaluation and visualization dashboard
```

## Features Extracted

Time-domain features:

- Mean
- Standard deviation
- RMS
- Variance
- Skewness
- Kurtosis
- Peak value
- Crest factor

Frequency-domain features:

- Dominant frequency
- Spectral energy
- Spectral centroid
- Maximum FFT amplitude

## Models

Baseline models:

- Support Vector Machine
- Random Forest

Advanced model:

- 1D Convolutional Neural Network on raw vibration windows

Optional extension:

- 2D-CNN on spectrogram images

## Evaluation Metrics

- Accuracy
- Macro-F1 score
- Confusion matrix
- Load-wise robustness testing

## Repository Structure

```text
bearing-fault-diagnosis/
├── data/
│   ├── raw/
│   └── processed/
├── dashboard/
│   └── app.py
├── docs/
│   └── project_plan.md
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── literature_survey.md
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_ml.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

For Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Usage

1. Download CWRU `.mat` files.
2. Place them in `data/raw/`.
3. Edit the file-label mapping in `src/load_data.py`.
4. Extract features and train classical ML models:

```bash
python src/train_ml.py
```

5. Run dashboard:

```bash
streamlit run dashboard/app.py
```

## Expected Deliverables

- Signal-processing pipeline
- Feature extraction module
- Trained SVM / Random Forest baseline
- 1D-CNN model extension
- Accuracy, macro-F1 and confusion matrix results
- Vibration visualization dashboard
- Final report and presentation

## Team Notes

This project is especially suitable for ECE students because it combines AIML with vibration signals, sensors, FFT, STFT, envelope analysis, and operating-condition robustness.
