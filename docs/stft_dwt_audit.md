# STFT and DWT audit and 1‑D CNN pilot

## Findings from the uploaded graphs

The feature folder contains 77 PNGs in 11 sets. I compared the decoded RGB pixels and checked the waveform channel titles against the raw files. Several PNGs contain identical pixels even though their file hashes differ.

| Uploaded plot set | Evidence | Disposition |
|---|---|---|
| Ball, 0.007 inch | All seven plots match the inner-race 0.007-inch set pixel for pixel. The waveform title shows `X107_DE_time`, which belongs to `IR007_2.mat`. | Regenerate from `Ball/B007_2.mat` using `X120_DE_time`. |
| Inner race, 0.007 inch | The plotted channel `X107_DE_time` matches `Inner Race/IR007_2.mat`. | Correct channel for this label. |
| Ball and inner race, 0.014 and 0.021 inch | Waveform titles identify ball channels `X187_DE_time` and `X224_DE_time`, and inner-race channels `X171_DE_time` and `X211_DE_time`. Each exists in the corresponding uploaded 2 HP file. | Channel titles match. Exact legacy transform settings remain unverified. |
| Outer race, centered, opposite and orthogonal | For each of the seven plot types, all three folders contain identical pixels. The waveform title shows `X132_DE_time`, which belongs to `OR007@6_2.mat`. | The centered set has the matching source. The opposite and orthogonal sets need their own @12 and @3 recordings. |
| Ball and inner race, 0.028 inch | Waveform titles show `X050_DE_time` and `X058_DE_time`. Those channels and matching raw recordings are absent from the upload. | Hold these plots out until the original recordings and settings can be matched. |
| Healthy | Healthy recordings are present. The feature graph folder has no healthy set. | The notebook plots healthy features from `Normal_2.mat` with the verified `X099_DE_time` channel. |

The existing STFT axes are consistent with 12 kHz for the fault plots shown. The 32- and 64-sample windows span about 2.67 ms and 5.33 ms. Those windows can show broad bands and give limited frequency resolution for closely spaced components. Their different lengths also make direct comparisons across plot sets difficult. The DWT coefficient counts and nominal bands are plausible for a four-level dyadic decomposition at 12 kHz. The separate frequency-axis scalogram does not establish the DWT parameters. Exact numerical agreement for the old exports requires the generating script and settings, including the wavelet, boundary mode, STFT overlap and FFT size.

All 16 pilot raw files match the existing locked SHA-256 hashes. `Normal_2.mat` contains both `X099_DE_time` and a copy of `X098_DE_time` that is sample-for-sample identical to the channel in `Normal_1.mat`. The locked manifest selects `X099_DE_time` for the 2 HP recording, and M4 checks selected channel content for duplicates across groups. The run follows the existing manifest's literature-sourced 48 kHz healthy drive-end rate and resamples healthy signals to 12 kHz with an anti-alias filter. The CWRU tables identify the 12 kHz fault cohort and do not list a per-file healthy sampling rate.

## Reproduce the pilot

From the repository root, install `python -m pip install -r requirements.txt` and run `python -m src.train_cnn`. You can also run [the notebook](../notebooks/05_cnn_model.ipynb) to inspect the waveform, STFT and DWT plots for all four classes. The CLI prints held-out metrics and saves `models/stft_dwt_1d_cnn.keras`. Inference with this model takes the output of `src.stft_dwt.extract_sequences` using the same vibration-window preparation.

The first experiment uses the previously locked cohort of 16 recordings. Faults are 0.007 inch and the outer-race position is fixed at 6 o'clock. The 0 and 1 HP recordings train the model, the 2 HP recordings select weights by validation loss, and the 3 HP recordings supply the test. All overlapping windows from an original recording stay together.

Each 2,048-sample window becomes a `(15, 134)` sequence. At each of 15 frame times, the model receives 129 Hann STFT log-power bins and five db4 DWT log mean-square coefficients (A4, D4, D3, D2, D1). The frames are 256 samples long with a 128-sample hop, and the DWT uses symmetric extension. The CNN convolves along time. It retains the instructor's two convolution/pooling blocks and fits its feature normalization only on training data.

Report window metrics and original-recording metrics separately. The test has four original recordings. Results apply to the selected CWRU cohort and do not establish performance on other bearings, fault sizes, positions or machines.

## Pilot run on the uploaded recordings

The run used Python 3.12.14, TensorFlow CPU 2.21.0, PyWavelets 1.10.0 and random seed 42. It produced 881 training windows, 469 validation windows and 471 test windows from 8, 4 and 4 recordings, respectively. After 10 epochs it classified 471/471 test windows and 4/4 test recordings correctly. Macro-F1 was 1.00 at both levels. Overlapping windows share signal samples, so the independent test count is four recordings. The test uses a different motor load within the same rig and experimental cohort. The design also cannot isolate any cues tied to how the healthy and fault data were acquired.

Sources for metadata include the [CWRU data index](https://engineering.case.edu/bearingdatacenter/download-data-file), [12 kHz drive-end fault table](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data), [healthy baseline table](https://engineering.case.edu/bearingdatacenter/normal-baseline-data) and the healthy-rate paper cited in the [locked manifest](../data/manifests/cwru_007.lock.json).
