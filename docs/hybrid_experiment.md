# Final hybrid experiment

This experiment builds on PR #9 and uses the supplied numerical ZIP archive. The original Features.zip and image archive are not needed for training this hybrid: STFT/DWT are regenerated with the existing GitHub code from the aligned numerical vibration windows. The image archive is for documentation.

From the repository root (Python 3.11 is recommended):

    python -m pip install -r requirements.txt
    python -m src.run_hybrid_experiment --dataset "Bearing_Fault_Feature_Dataset(1).zip" --check-only
    python -m src.run_hybrid_experiment --dataset "Bearing_Fault_Feature_Dataset(1).zip" --epochs 30

Alternatively, use notebooks/06_hybrid_cnn_lstm.ipynb in Jupyter. Point DATASET to the numerical ZIP. Raw MAT recordings must still be present at the repository's manifest-pinned paths.

The ZIP runner extracts only the required index, vibration and envelope arrays into a temporary directory; checks fail closed if required members or raw-window provenance are missing. No input data are uploaded or committed. Choose a new output path for each experiment.

Expected outputs in models/hybrid_experiment:
- stft_dwt_envelope_cnn_lstm.keras and stft_dwt_baseline.keras
- results.json, with histories and clean/noisy held-out metrics
- report/metrics.csv and per_class_metrics.csv
- report/*.png, learning curves, confusion matrices, comparison chart
- report/results_and_discussion.md, generated from measured results

Interpret the results in terms of load-held-out performance, the clean-to-noisy accuracy change, class-level confusion and learning curves. The single-seed comparison is exploratory: independent recordings are limited and the hybrid changes architecture and input features at the same time. Do not claim an isolated LSTM or Envelope effect without ablation, or a general industrial performance guarantee.

The notebook and pipeline have not yet been executed against the supplied ZIP. Do not fill in numerical results until the validation and training complete.
