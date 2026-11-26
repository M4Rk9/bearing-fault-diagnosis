# Final hybrid experiment

This experiment builds on PR #9 and uses the supplied numerical ZIP archive. The original Features.zip and image archive are not needed for training this hybrid: STFT/DWT are regenerated with the existing GitHub code from the aligned numerical vibration windows. The image archive is for documentation.

From the repository root (Python 3.11 is recommended):

    python -m pip install -r requirements.txt
    python -m src.run_hybrid_experiment --dataset "Bearing_Fault_Feature_Dataset(1).zip" --check-only
    python -m src.run_hybrid_experiment --dataset "Bearing_Fault_Feature_Dataset(1).zip" --epochs 30

Alternatively, use notebooks/06_hybrid_cnn_lstm.ipynb in Jupyter. Point DATASET to the numerical ZIP or extracted dataset folder. The successful local run used the extracted [split_70_30 folder](https://drive.google.com/drive/folders/1qR96FcLrIDfTck5qRwqEBCpYKuxXntJG). If using an archive, verify it contains train_70 and test_30; the earlier unsplit archive is incompatible. Raw MAT recordings must still be present at the repository's manifest-pinned paths.

The ZIP runner extracts only the required index, vibration and envelope arrays into a temporary directory; checks fail closed if required members or raw-window provenance are missing. No input data are uploaded or committed. Choose a new output path for each experiment.

Expected outputs in models/hybrid_experiment:
- stft_dwt_envelope_cnn_lstm.keras and stft_dwt_baseline.keras
- results.json, with histories and clean/noisy held-out metrics
- report/metrics.csv and per_class_metrics.csv
- report/*.png, learning curves, confusion matrices, comparison chart
- report/results_and_discussion.md, generated from measured results

Interpret the results in terms of load-held-out performance, the clean-to-noisy accuracy change, class-level confusion and learning curves. The single-seed comparison is exploratory: independent recordings are limited and the hybrid changes architecture and input features at the same time. Do not claim an isolated LSTM or Envelope effect without ablation, or a general industrial performance guarantee.

## Completed experiment — 26 November 2026

The team ran the notebook successfully on the extracted numerical dataset. The measured run is archived in [the experimental report](../reports/hybrid/2026-11-26/README.md), with the original [Drive artifacts](https://drive.google.com/drive/folders/1RCVT_pJwR7g_njQP1bRmwgwDp3ilPkZb). Both trained models were saved. The hybrid achieved 100.00% clean and 96.70% noisy window accuracy, versus 97.87% and 94.29% for the matched baseline. These values are from the saved experiment rather than a new reproduction by the documentation update. Keep existing results separate from any future run.
