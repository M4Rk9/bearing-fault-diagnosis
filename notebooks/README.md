# Notebooks

Use this folder for step-by-step experimentation. The [CNN pilot notebook](05_cnn_model.ipynb)
runs on the uploaded CWRU cohort. The other notebook names below are planned deliverables.
Follow the [roadmap](../docs/roadmap.md) and [evaluation protocol](../docs/data_and_evaluation_protocol.md).

Suggested notebook order:

1. `01_data_loading.ipynb` - inspect CWRU `.mat` files, validate the record manifest and assign original recording groups to splits before windowing (M4).
2. `02_signal_visualization.ipynb` - plot waveform, FFT, and spectrogram.
3. `03_feature_extraction.ipynb` - validate amplitude-preserving statistics, FFT/envelope features, STFT and DWT settings (M5).
4. `04_ml_models.ipynb` - train and evaluate SVM/Random Forest using the saved group splits (M6).
5. [`05_cnn_model.ipynb`](05_cnn_model.ipynb) - audit the plots, regenerate numeric STFT/DWT features and train the 1D-CNN on recording-level splits (M5/M7 pilot).
6. `06_robustness.ipynb` - cross-load, noise and measurement-path experiments (M8).

Keep heavy outputs and dataset files out of git.
