# Bearing Fault Diagnosis from Vibration Signals

An ongoing fifth-semester ECE project at **Birla Institute of Technology, Mesra**, combining bearing physics, MATLAB vibration analysis and a Python machine-learning pipeline.

The objective is to identify **healthy, inner-race, ball and outer-race** conditions from CWRU vibration recordings. Defect-size prediction and generalization to other conditions are later experiments.

## Where we are

**Updated 10 November 2026.** M1–M3 study deliverables and the M4 16-record ingestion cohort are complete. The [STFT and DWT audit](docs/stft_dwt_audit.md) identifies labeling errors in the uploaded plots and documents a reproducible four-class CNN pilot.

| Milestone | Deliverable | Status |
|---|---|---|
| M1 | Motor operation and bearing-fault foundations | Complete — study |
| M2 | CWRU dataset selection and literature study | Complete — study |
| M3 | MATLAB signal interpretation and time-frequency study | Complete — exploratory study |
| M4 | Reproducible ingestion and preprocessing | **Complete — 16 recordings, 1,821 windows, passing CI** |
| M5 | Validated statistical, spectral and wavelet features | STFT/DWT pilot implemented. Broader feature validation pending. |
| M6 | Evaluated SVM and Random Forest baselines | Pending — training scripts exist |
| M7 | Evaluated 1D-CNN and baseline comparison | 0.007-inch CNN pilot evaluated. Baseline comparison pending. |
| M8 | Cross-condition robustness and error analysis | Pending |
| M9 | Validated inference and dashboard integration | Pending — prototype exists |
| M10 | Reproducible academic release and presentation | Pending |

See the [roadmap](docs/roadmap.md) for completion criteria, [study progress](docs/study_progress.md) for findings, and [source inventory](docs/source_inventory.md) for the supplied reports and figures reviewed. Study completion does not mean that proposed algorithms have been implemented or evaluated.

## Evidence and implementation

| Area | Available evidence | Remaining work |
|---|---|---|
| Engineering and dataset study | Motor reports, dataset comparison and detailed CWRU study | Resolve source inconsistencies against recording metadata |
| MATLAB exploration | Classification reports and original IR014, IR021 and IR028 plots | Preserve generating scripts and raw-file mapping |
| Time-frequency methods | STFT, CWT and DWT study guide, plus reproducible STFT/DWT for the pilot | Recover settings and provenance for legacy plots |
| Python features | 9 statistical, 4 FFT and 3 Hilbert-envelope features | Validate feature definitions and add band/wavelet features |
| Models | SVM and Random Forest starter scripts, plus TensorFlow 1D-CNN pilot | Compare models and test beyond the pilot cohort |
| Dashboard | CSV/synthetic visualization and optional RF prediction | Integrate a validated model and enforce its input contract |

## M4 implementation update — 14 September 2026

A [manifest-driven ingestion pipeline](docs/m4_implementation.md) now validates recording groups, file integrity, channels and signal quality, exports amplitude-preserving feature windows and separately normalized CNN windows, and retains window provenance. It includes anti-alias resampling, tests and a real-record CI example.

**M4 is complete for this cohort:** all 16 official recordings were acquired and hash-checked, yielding 881 train / 469 validation / 471 test windows across isolated recording groups. Healthy DE rates are documented from experimental literature and resampled from 48 to 12 kHz. All 13 M4 tests and the full real-data CI job passed.

See the [retained full-cohort summary](reports/m4/cohort/summary.json) and [real-record plot](reports/m4/first-real-record/record.png).

```bash
pip install -r requirements-m4.txt
python -m unittest discover -s tests -v
```

See [M4 reproduction commands and acceptance evidence](docs/m4_implementation.md). The legacy training loader now stops explicitly rather than allowing a random split of overlapping windows. M5–M7 will integrate partitioned artifacts with validated features and models.

## Setup and visualization

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Linux/macOS, then:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard displays a synthetic demonstration or numeric CSV without a trained model. Predictions require `models/random_forest.pkl`; synthetic input is not evidence of diagnostic accuracy.

Build the [four-class feature dataset](Datasets/Features) with `python -m src.feature_dataset`, then run `python -m src.train_cnn` from the repository root. The model trains on the exported clean and noisy STFT/DWT features and reports test metrics for each variant. The [CNN notebook](notebooks/05_cnn_model.ipynb) also plots paired examples for all four classes. The SVM and Random Forest starter still uses the disabled legacy loader.

## Project resources

- [Roadmap and acceptance criteria](docs/roadmap.md)
- [Completed study and signal observations](docs/study_progress.md)
- [Dataset, preprocessing and evaluation requirements](docs/data_and_evaluation_protocol.md)
- [Literature survey](reports/literature_survey.md)
- [Reviewed reports and images](docs/source_inventory.md)
- [Resume wording for current progress](docs/resume_points.md)
- [Notebook plan](notebooks/README.md)
- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)

The uploaded recordings and exploratory plots are under [Datasets](Datasets). Generated model files and prepared arrays stay outside Git.
