# Data and evaluation protocol

This is the implementation contract for M4–M8, not a description of functionality already delivered. The current loader and training scripts do not yet meet it.

## Record manifest

Create one manifest row per selected recording/channel, retaining a shared recording group across all channels, renamed copies and derivatives. Do not populate metadata from guesses or illustrative report filenames.

| Required field | Meaning and validation |
|---|---|
| record_id, group_id | Stable original-recording identity and split group |
| relative_path, sha256 | Raw-file location and integrity hash |
| source_url | Official dataset record/download provenance |
| class_name | normal, inner_race, ball or outer_race |
| fault_diameter_in | Published defect size; null for healthy |
| fault_depth_in, manufacturer | Preserve known confounders; unknown values explicitly null |
| load_hp, rpm | Recorded operating condition; distinguish actual RPM from table approximation |
| faulted_bearing_end | DE, FE, or not applicable for healthy |
| channel_key, sensor_end | Actual MAT variable and accelerometer position |
| sampling_rate_hz | Verified rate for the selected channel, not inferred from its new filename |
| outer_race_position | Published angular position or not applicable |
| units, calibration_source | Verified units, or explicitly unknown |
| partition | train, validation or test at the group level |

Validate missing arrays, unexpected dimensionality, nonnumeric/nonfinite samples, constants, length, duplicate hashes and inconsistent metadata. Do not fall back to an arbitrary large numeric array when the requested channel is missing. Keep the raw data unchanged.

## Partition before segmentation

1. Choose a feasible class/condition subset after inspecting coverage. A useful initial target is normal plus 0.007-inch IR/ball/OR records across available loads, keeping OR position controlled.
2. Assign original recording groups to partitions before filtering/windowing. All windows, resampled copies, image representations and channels of the same recording stay together.
3. Persist the partition manifest and assert disjoint groups/hashes. A random seed does not solve leakage from overlapping windows.
4. With very few independent records per class, explicitly acknowledge limited independent evidence. If needed, use a load-held-out design rather than inventing a stratified group split that the data cannot support.
5. Apply overlap only within a partition. Use independent validation groups for early stopping and hyperparameter selection.

Initial window setting: 2048 samples with 50% overlap at a verified 12 kHz target. This is approximately 0.1707 s. At 48 kHz, 2048 samples span only 0.0427 s; 8192 samples give the same duration as 2048 at 12 kHz. Record both sample count and physical duration.

Resampling requires anti-alias filtering and retained source/target rates. Fit learned preprocessing only on training data. Fixed signal filters still require documented cutoffs, boundary treatment and physical justification.

## Two preprocessing branches

**Engineered-feature branch:** preserve physical amplitude after documented calibration, optional filtering and mean removal. Extract RMS, standard deviation, variance and peak quantities before any per-window variance normalization. Standardize the resulting feature columns with a scaler fitted on training rows only where appropriate.

**CNN branch:** per-window normalization may be used to emphasize shape, with documented epsilon and handling of zero-variance inputs. Apply the same transformation at inference. Evaluate amplitude dependence as an explicit design choice.

The existing path calls normalize_signal before all features. For nonconstant windows this makes mean approximately zero and standard deviation, RMS and variance approximately one. It also changes energy and peak semantics. Retain that limitation until the two branches are implemented and tested.

## Feature and transform validation

- Verify the existing 16 outputs and define units, FFT scaling, window/taper and kurtosis convention (the current SciPy default is excess kurtosis).
- Document whether spectral centroid is magnitude- or power-weighted; the existing implementation uses magnitude weights.
- The current envelope energy includes its DC component. Define whether later versions remove the envelope mean, and version the feature schema when semantics change.
- Select resonance bands before envelope analysis using training evidence and physics; save the choice rather than tuning against test labels.
- Calculate BPFI/BPFO/FTF and rolling-element bands using the faulted bearing geometry and RPM. Resolve the BSF versus 2×BSF naming issue documented in the study summary.
- Save STFT window type/length, overlap, FFT length, scaling and display limits.
- For DWT, save wavelet family, level, extension mode, selected subbands and reconstruction rule. Check reconstruction error and boundary behaviour. Evaluate band features and denoising; do not assume improvement.
- Use identical image normalization rules across partitions for any later spectrogram/CWT classifier.

## Models and evaluation

SVM/RF and CNN must use the same underlying recording partitions. Do not retain the current window-level train_test_split or CNN validation_split as the final experimental design. Use explicit validation arrays from held-out groups.

Report accuracy, macro-F1, per-class precision/recall/F1, class support and confusion matrices. Record numbers of independent recordings as well as windows. Report variability across training seeds where feasible without treating correlated windows as independent replicates.

For cross-load evaluation, an initial design is train on 0–2 HP and test on 3 HP, subject to class coverage. Preserve a separate validation design inside the development data. Noise experiments must document signal power, injected-noise power, SNR and random seed. Sensor-end comparisons must distinguish sensor location from faulted bearing location.

Do not select models by repeatedly checking the final test set. Do not compare scores from different class definitions or split protocols as if they measured the same task.

## Reproducible evidence to save

- Code revision, dependency versions, commands and random seeds.
- Dataset manifest/hash and train/validation/test group membership.
- Feature schema, units, processing configuration and class mapping.
- Metrics in a machine-readable format, plots and a concise interpretation.
- Model bundle location/checksum and inference input requirements.

Raw data and large models can remain outside Git, but their provenance and reproduction path must be recorded. No accuracy value is filled in until an actual experiment produces it.

## Focused implementation checks

Test metadata/channel selection, duplicate/group isolation, short/constant/nonfinite inputs, window boundaries, known feature values and amplitude preservation. Check that changing only test data cannot change fitted scaler parameters. Compare training and dashboard features on the same input. These checks belong to the implementation milestones; this documentation-only update does not claim they exist.
