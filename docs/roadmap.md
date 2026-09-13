# Milestone roadmap

Status baseline: **13 September 2026**, based on code at `f7a8270b8da4f742f09f44ebe4a8b213e69bae63` and the supplied study inventory.

## How progress is measured

Complete means the milestone's scoped deliverables have evidence. M1–M3 are study milestones: they do not certify reproducible numerical experiments, trained models or mastery of every paper. In progress means work exists but acceptance criteria are unmet. Pending can include starter code without validation.

**Current milestone: M4.** Three of ten milestones have completed their study deliverables; this is not a percentage of effort or product readiness. Later-stage scripts do not bypass the data-validation gates.

## M1 — Engineering foundations

**Status: Complete — study.**

- [x] Document motor construction, rotating magnetic field, induction, torque and slip.
- [x] Explain shaft support, races, rolling elements and cage.
- [x] Survey localized faults and lubrication, corrosion, electrical and mounting damage.
- [x] Define the initial target: healthy, inner race, ball and outer race.

Evidence: [motor report](source_inventory.md), [fault and dataset study](source_inventory.md), [summary](study_progress.md#engineering-foundations). This does not claim construction of a physical test rig.

## M2 — Dataset and literature study

**Status: Complete — study. Depends on M1.**

- [x] Select CWRU and compare alternative datasets.
- [x] Document load, RPM, fault size, channel, bearing end and sampling metadata.
- [x] Study the progression from engineered features to CNNs, transfer learning and transformers.
- [x] Document seeded-defect limitations, measurement paths and leakage concerns.

Evidence: [detailed CWRU study](source_inventory.md), [literature survey](../reports/literature_survey.md). Actual record acquisition and validation belong to M4.

## M3 — Exploratory signal and time-frequency study

**Status: Complete — exploratory study. Depends on M2.**

- [x] Interpret the team's MATLAB healthy, IR, ball and OR plots.
- [x] Compare selected defect sizes, sampling rates and measurement paths.
- [x] Study FFT, envelope analysis, STFT, CWT and DWT principles.
- [x] Explain why peak amplitude alone is insufficient for severity.

Evidence: [classification report](source_inventory.md), [STFT/DWT guide](source_inventory.md), [original plots](source_inventory.md). Generating scripts and verified transform experiments were not supplied; reproducing these is part of M4–M5. This does not imply a wavelet classifier exists.

## M4 — Reproducible ingestion and preprocessing

**Status: In progress — current milestone. Depends on M3.**

Existing work: [MAT loader](../src/load_data.py), [preprocessing helpers](../src/preprocessing.py), [implementation protocol](data_and_evaluation_protocol.md).

- [ ] Obtain actual CWRU recordings from the official source.
- [ ] Populate a validated manifest: file hash, source URL, class, defect size, load, RPM, sampling rate, channel and bearing end.
- [ ] Verify healthy-record sampling metadata; detect missing keys, nonfinite values, duplicates, constant and short signals.
- [ ] Assign complete recording groups to train/validation/test before segmentation; keep channels and copies of one recording together.
- [ ] Preserve amplitude for engineered features; normalize CNN inputs separately.
- [ ] Implement sampling-rate handling, anti-alias resampling where needed, and duration-aware window settings.
- [ ] Retain processing scripts and a reproducible raw-record-to-plot example.
- [ ] Test metadata validation, group isolation, window boundaries and normalization behaviour.

**Acceptance evidence:** manifest metadata, ingestion summary, split membership, hashes, commands, meaningful passing tests and a reproducible real-record example. Raw data can remain outside Git. No classifier score is needed to finish M4.

## M5 — Validated feature extraction

**Status: Pending — 16 initial features exist. Depends on M4.**

- [ ] Verify the nine statistical, four FFT and three envelope features against reference calculations.
- [ ] Calculate amplitude-sensitive features on amplitude-preserving inputs with documented units.
- [ ] Add RPM/bearing-aware frequency bands, envelope harmonics and sidebands.
- [ ] Reproduce STFT and DWT; document wavelet family, level, boundaries, scaling and reconstruction.
- [ ] Evaluate wavelet-band features and denoising without assuming an SNR gain.
- [ ] Export features with recording/window identifiers and the processing schema.

**Acceptance evidence:** feature checks, finite-value validation, parameterized plots, an ablation-ready table and a reproducible extraction command. CWT is an optional comparison; a 2D image classifier is not required.

## M6 — Classical machine-learning baselines

**Status: Pending — SVM/RF scripts exist. Depends on M5.**

- [ ] Integrate M4 partitions and fit scalers using training data only.
- [ ] Select hyperparameters on validation data without examining final test scores.
- [ ] Report accuracy, macro-F1, class-wise precision/recall/F1 and confusion matrices.
- [ ] Save model, feature order, classes, preprocessing, split identity and random seed together.
- [ ] Compare time-only, spectral/envelope and combined features using the same split.

**Acceptance evidence:** real-data experiment report, machine-readable metrics, dataset/split identifiers and reproducible trained-artifact retrieval. An arbitrary accuracy target cannot replace valid evaluation.

## M7 — Deep-learning comparison

**Status: Pending — 1D-CNN script exists. Depends on M6.**

- [ ] Train the compact CNN using the same recording groups as M6.
- [ ] Use an explicit group-separated validation set and documented early stopping.
- [ ] Report macro-F1, class-wise metrics, confusion matrix and learning curves.
- [ ] Compare results, model size and inference cost with SVM/RF.
- [ ] Document errors and save reproducible artifacts/configuration.

**Acceptance evidence:** a comparable CNN experiment report. A 2D-CNN on STFT/CWT images is optional after the baseline and is not currently implemented.

## M8 — Robustness and error analysis

**Status: Pending. Depends on M6 and M7.**

- [ ] Hold out entire loads; initially train on 0–2 HP and test on 3 HP if verified coverage supports it.
- [ ] Test noise sensitivity with a recorded injection procedure and SNR definition.
- [ ] Study held-out defect sizes and sensor paths where class coverage permits valid comparisons.
- [ ] Keep faulted bearing end, sensor identity and bearing geometry distinct.
- [ ] Report failures and variability across training seeds without tuning on final test data.

**Acceptance evidence:** robustness tables, split definitions, class coverage and failure analysis. Cross-dataset testing is optional and requires compatible tasks/labels; NASA IMS is not automatically a four-class CWRU test set.

## M9 — Validated dashboard and inference

**Status: Pending — Streamlit prototype exists. Depends on M8.**

- [ ] Load a versioned model bundle with matching features and preprocessing.
- [ ] Validate CSV layout, channel, sampling rate, length and invalid values.
- [ ] Enforce compatible windows and define aggregation across multiple windows.
- [ ] Display waveform, FFT, spectrogram, features and results consistently.
- [ ] Label synthetic demonstrations and handle absent models clearly.
- [ ] Check inference against reference inputs and measure latency.

**Acceptance evidence:** real-data demonstration, input-contract tests and a launch command. Deployment and live hardware acquisition are optional work with separate validation requirements.

## M10 — Academic release and handoff

**Status: Pending. Depends on M9.**

- [ ] Consolidate methodology, experiments, limitations, citations and contributions.
- [ ] Produce a presentation and viva preparation grounded in actual results.
- [ ] Reproduce the selected experiment and inference demonstration from a fresh setup.
- [ ] Record dependency versions, provenance and artifact checksums.
- [ ] Tag the academic release and update README/resume claims to match evidence.

**Acceptance evidence:** release tag, final report, presentation, reproduction instructions and completed evidence table. Industrial reliability, remaining useful life and maintenance savings are not established by CWRU classification alone.

## Next concrete work

Begin M4 by obtaining raw recordings, resolving their metadata and implementing the manifest/group split. Do not treat scores from the current overlapping-window random split as completion of M6.

To close a milestone, link its implementation commit, commands, tests and report here, record the completion date, and update the README table in the same change.
