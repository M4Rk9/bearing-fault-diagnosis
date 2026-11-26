# Notebooks

Use this folder for step-by-step experimentation. The [CNN pilot notebook](05_cnn_model.ipynb)
runs on the uploaded CWRU cohort. The [hybrid experiment notebook](06_hybrid_cnn_lstm.ipynb) was executed locally on the extracted numerical feature dataset; [measured results](../reports/hybrid/2026-11-26/README.md). Other names below may be planned deliverables.
Follow the [roadmap](../docs/roadmap.md) and [evaluation protocol](../docs/data_and_evaluation_protocol.md).

Suggested notebook order:

1. `01_data_loading.ipynb` - inspect CWRU `.mat` files, validate the record manifest and assign original recording groups to splits before windowing (M4).
2. `02_signal_visualization.ipynb` - plot waveform, FFT, and spectrogram.
3. `03_feature_extraction.ipynb` - validate amplitude-preserving statistics, FFT/envelope features, STFT and DWT settings (M5).
4. `04_ml_models.ipynb` - train and evaluate SVM/Random Forest using the saved group splits (M6).
5. [`05_cnn_model.ipynb`](05_cnn_model.ipynb) - generate paired clean/noisy STFT and DWT features and train the 1D-CNN on recording-level splits (M5/M7 pilot).
6. [`06_hybrid_cnn_lstm.ipynb`](06_hybrid_cnn_lstm.ipynb) - validate aligned inputs, train both models and generate results. Use DATASET pointing to the extracted split_70_30 folder or a compatible ZIP.
7. Additional cross-load, noise and measurement-path experiments — future work.

Keep heavy outputs and dataset files out of git.
