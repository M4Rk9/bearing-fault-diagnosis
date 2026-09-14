# Study progress and findings

Review date: 13 September 2026. This consolidates the supplied reports, guides and images. The [source inventory](source_inventory.md) identifies the reviewed originals; explanations here distinguish observations, calculations and proposed experiments. Original attachments are not published in this change.

Academic context: Project I, EC24300, Department of Electronics and Communication Engineering, Birla Institute of Technology, Mesra. The reports describe a three-member project group. Findings are group work unless individual contributions are explicitly established.

## Engineering foundations

The motor study covers stator and rotor construction, three-phase rotating magnetic fields, induced rotor current and torque. Synchronous speed is Ns = 120f/P, with supply frequency f and pole count P; fractional slip is s = (Ns − Nr)/Ns. For a four-pole, 60 Hz example, Ns = 1800 rpm. The RPM difference is slip speed; it is not itself the dimensionless slip ratio. Three-phase induction motors are self-starting in the torque sense, which does not mean switching, protection or starting-current control is unnecessary.

Bearings support the shaft, maintain alignment and allow low-friction rotation. Local damage on a race or ball produces impacts and structural ringing. The wider fault survey includes cage damage, corrosion, inadequate lubrication, electrical erosion, mounting damage and misalignment. Those mechanisms are background knowledge; they are not all labelled classes in the initial CWRU classifier.

## Dataset selection and experimental context

The selected primary source is the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome). Its [apparatus description](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures) documents a 2 HP motor, torque transducer/encoder, dynamometer, control electronics and accelerometers. Defects were seeded using EDM. The team is studying this existing dataset; the evidence does not establish that it built or instrumented the rig.

The reports compare CWRU with NASA IMS, Paderborn, PRONOSTIA, XJTU-SY, Ottawa and HUST. CWRU fits an initial supervised fault-location task. Run-to-failure datasets support different questions, including degradation, and require separate label/task design before external validation.

The explored records are mostly labelled load 2, approximately 1750 rpm. The [12 kHz drive-end table](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) lists 0–3 HP load conditions and shows that fault-size and outer-race-position coverage is not uniform. A motor's nameplate rating and an experimental load label are different metadata.

Keep three concepts separate: the bearing containing the fault, the accelerometer channel used, and the sampling rate of that channel. The supplied FE-folder plot titles explicitly show a DE channel. They must not automatically be described as measurements from the fan-end sensor.

## MATLAB signal observations

These are qualitative observations from supplied plots, not new measurements calculated from raw recordings. The detailed report inventories 18 figures; only three were also supplied as standalone IR plot images.

| Condition | Observed pattern in the supplied study | Diagnostic interpretation |
|---|---|---|
| Healthy | Relatively small background vibration without the strongest repeating bursts | Baseline mechanical vibration is not zero |
| Inner race | Repeated impacts with varying amplitudes and ringing | A rotating defect can move through different loading regions |
| Ball | Modulated and sometimes intermittent activity | Spin and orbital motion complicate the signal |
| Outer race | Several examples show pronounced repeating ring-down bursts | A fixed defect can create a more stationary impact pattern |

### Original inner-race examples

IR014 contains separated sharp excursions against a quieter background, useful for studying impulsiveness.

IR021 contains repeated bursts with varying amplitude and ringing.

IR028 appears denser over the displayed interval. It is not simply a scaled copy of the smaller-defect traces. The CWRU apparatus documentation identifies NTN equivalents for larger faults versus SKF bearings for smaller faults, adding a confounding variable.

All three supplied charts display the first 0.2 s and label amplitude as g. Those labels are preserved as source content, not independently verified calibration. The source arrays and plotting code were not supplied.

### Comparisons already studied

- Selected IR/ball defects range from 0.007 to 0.028 inch; the reported OR comparison uses 0.007, 0.014 and 0.021 inch.
- Larger diameter does not consistently imply a larger plotted peak. Load, depth, manufacturer, mounting and measurement path affect amplitude.
- The report compares 12 kHz and 48 kHz plots. Sampling changes observable bandwidth and samples per transient; it does not change the physical fault label.
- FE-dataset/DE-channel figures illustrate measurement-path differences, not proof of classifier generalization.

## Characteristic frequencies

At an illustrative shaft speed of 1750 rpm, fr = 1750/60 ≈ 29.17 Hz. Using the [official drive-end bearing multipliers](https://engineering.case.edu/bearingdatacenter/bearing-information) gives the following calculated references, not measured FFT peaks:

| Reference | Multiplier × fr | Approximate value |
|---|---|---|
| Inner-race BPFI | 5.4152 | 157.94 Hz |
| Outer-race BPFO | 3.5848 | 104.56 Hz |
| Cage FTF | 0.39828 | 11.62 Hz |
| CWRU rolling-element defect reference | 4.7135 | 137.48 Hz |

The classification report labels about 68.7 Hz as BSF and 137.5 Hz as 2×BSF, whereas the detailed dataset report abbreviates the official 4.7135× rolling-element reference as BSF. Preserve this distinction and state the convention when implementing bands. Do not use a single ambiguous BSF field. Fan-end bearings have different multipliers even if a drive-end accelerometer observes them.

## Signal-processing study

| Method | Purpose covered | Implementation evidence |
|---|---|---|
| FFT | Frequency distribution, harmonics and spectral peaks | Python functions and supplied spectrum discussion |
| Statistical features | Energy, waveform shape and impulsiveness | Python RMS, variance, skewness, kurtosis and crest-factor functions |
| Hilbert envelope | Demodulation of impact-related amplitude modulation | Simple envelope features in Python; tuned resonance-band analysis pending |
| STFT | Local spectra and the window-length resolution trade-off | Guide plus Python spectrogram visualization helper |
| CWT | Time-scale representation of transient signals | Study guide; no executable CWT implementation in the reviewed repository |
| DWT | Multiresolution filtering, approximation/detail bands and denoising concepts | Study guide; implementation and validation pending |

For a nominal three-level dyadic DWT at 12 kHz, D1 approximately covers 3–6 kHz, D2 1.5–3 kHz, D3 0.75–1.5 kHz and A3 0–0.75 kHz. These are idealized bands: real filters have transition regions and boundary effects. The guide's db4/level-three choice is a study example, not an experimentally established optimum.

### Clarifications before implementation

1. FFT remains useful on transient signals; a global magnitude spectrum does not localize individual events in time. It should not be described as universally failing on nonstationary data.
2. Wavelets vary time/frequency resolution with scale; they do not remove the uncertainty trade-off. See [MathWorks Fourier-to-wavelet explanation](https://www.mathworks.com/help/wavelet/gs/from-fourier-analysis-to-wavelet-analysis.html).
3. Discarding low-frequency coefficients or thresholding does not guarantee a noise-free signal or improved SNR. Estimate useful bands and measure the effect.
4. A fixed band is not automatically a universal SKF bearing resonance: the housing, mounting and measurement path contribute to resonance.
5. Verify original sampling rates, particularly healthy records. Labels such as Healthy_2_12k are not sufficient provenance.
6. For outer-race position, the 12 kHz drive-end table states a load zone centred at 6 o'clock; the general apparatus prose has a conflicting position description. Retain the actual table/record source and avoid silently combining those descriptions.

## Current software state

The repository contains a loader, 2048-sample segmentation with 50% overlap, normalization, 16 feature calculations, SVM/RF training, a compact 1D-CNN, evaluation helpers and a Streamlit prototype. The notebooks folder has a suggested sequence, not completed experiments.

No raw recordings, populated filename mapping, fitted models or measured accuracy/F1 results were provided. The training scripts split correlated windows randomly and normalize away amplitude before extracting amplitude-sensitive features. These are tracked in [M4 and M5](roadmap.md), not claimed as fixed here.

The correct current description is **completed foundational and exploratory study, with a starter implementation awaiting reproducible data preparation and model validation**.

## Implementation follow-up — 14 September 2026

The study review above remains a historical snapshot. See [M4 implementation and evidence](m4_implementation.md) for the new validated ingestion code, tests, reproduction commands and unresolved healthy-channel sampling metadata. No classifier result is claimed.
