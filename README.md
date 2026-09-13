# Bearing Fault Diagnosis from Vibration Signals

An ongoing fifth-semester ECE project at **Birla Institute of Technology, Mesra**, combining bearing physics, MATLAB vibration analysis and a Python machine-learning pipeline.

The objective is to identify **healthy, inner-race, ball and outer-race** conditions from CWRU vibration recordings. Defect-size prediction and generalization to other conditions are later experiments.

## Where we are

**Reviewed 13 September 2026: M1–M3 complete for their study deliverables; M4 in progress.** We have studied the motor and bearings, documented the dataset and literature, and interpreted MATLAB plots. Initial feature-extraction, training and dashboard code exists, but there are no verified model-training results.

| Milestone | Deliverable | Status |
|---|---|---|
| M1 | Motor operation and bearing-fault foundations | Complete — study |
| M2 | CWRU dataset selection and literature study | Complete — study |
| M3 | MATLAB signal interpretation and time-frequency study | Complete — exploratory study |
| M4 | Reproducible ingestion and preprocessing | **In progress — current milestone** |
| M5 | Validated statistical, spectral and wavelet features | Pending — initial features exist |
| M6 | Evaluated SVM and Random Forest baselines | Pending — training scripts exist |
| M7 | Evaluated 1D-CNN and baseline comparison | Pending — model script exists |
| M8 | Cross-condition robustness and error analysis | Pending |
| M9 | Validated inference and dashboard integration | Pending — prototype exists |
| M10 | Reproducible academic release and presentation | Pending |

See the [roadmap](docs/roadmap.md) for completion criteria, [study progress](docs/study_progress.md) for findings, and [source inventory](docs/source_inventory.md) for the supplied reports and figures reviewed. Study completion does not mean that proposed algorithms have been implemented or evaluated.

## Evidence and implementation

| Area | Available evidence | Remaining work |
|---|---|---|
| Engineering and dataset study | Motor reports, dataset comparison and detailed CWRU study | Resolve source inconsistencies against recording metadata |
| MATLAB exploration | Classification reports and original IR014, IR021 and IR028 plots | Preserve generating scripts and raw-file mapping |
| Time-frequency methods | STFT, CWT and DWT study guide | Reproduce transforms and validate settings |
| Python features | 9 statistical, 4 FFT and 3 Hilbert-envelope features | Preserve amplitude and validate band/wavelet features |
| Models | SVM, Random Forest and TensorFlow 1D-CNN scripts | Train and evaluate on independent recordings |
| Dashboard | CSV/synthetic visualization and optional RF prediction | Integrate a validated model and enforce its input contract |

## Current execution limitations

- `FILE_LABEL_MAP` in `src/load_data.py` is empty. Training needs downloaded `.mat` files and verified labels.
- Training randomly splits windows **after** segmentation with 50% overlap. Those scores must not be presented as independent-recording or cross-load validation. M4 will assign recording groups before windowing.
- The loader normalizes each window before feature extraction. This makes RMS, standard deviation and variance approximately constant, removing amplitude information. M4–M5 will separate amplitude-preserving features from normalized CNN inputs.
- The pipeline defaults to 12 kHz without a per-recording sampling-rate manifest. Verify each file/channel, including the healthy baseline; a renamed filename does not establish sampling rate.
- Trained models, experiment metrics, completed notebooks, DWT/CWT implementation and cross-load tests are absent from the reviewed revision.

The [data and evaluation protocol](docs/data_and_evaluation_protocol.md) defines the next implementation. This documentation update records the issues; it does not claim to fix training code.

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

Existing training entry points are `python src/train_ml.py` and `python src/train_cnn.py`. These remain starter scripts subject to the limitations above, rather than the final evaluation procedure.

## Project resources

- [Roadmap and acceptance criteria](docs/roadmap.md)
- [Completed study and signal observations](docs/study_progress.md)
- [Dataset, preprocessing and evaluation requirements](docs/data_and_evaluation_protocol.md)
- [Literature survey](reports/literature_survey.md)
- [Reviewed reports and images](docs/source_inventory.md)
- [Resume wording for current progress](docs/resume_points.md)
- [Notebook plan](notebooks/README.md)
- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)

Raw datasets, generated models and original uploaded reports/images remain outside Git. This update publishes original repository documentation summarizing the supplied study, plus a source inventory; it does not redistribute the attachments.
