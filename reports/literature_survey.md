# Literature study and implementation relevance

Updated 13 September 2026 from the supplied [detailed CWRU review](../docs/source_inventory.md), [classification report](../docs/source_inventory.md), and the repository's original reading list.

This is a synthesis of the team's supplied literature materials. It does not claim independent full-text verification or reproduction of every cited paper. Bibliographic details below are transcribed from those materials; verify them against the publisher before final academic submission. No published accuracy is presented as a result achieved by this project.

## What the study supports

Engineered time, frequency and envelope features provide interpretable baselines. Raw-signal CNNs learn local patterns; spectrogram/wavelet image models introduce another representation; transfer learning and self-supervision target limited labels and changing conditions. A strong evaluation protocol is necessary before a high benchmark score is meaningful.

| Study in supplied review | Method or focus | Application to our roadmap |
|---|---|---|
| Neupane and Seok, 2020 | Review of deep learning on CWRU | Dataset and model context, M2 |
| Zhang et al., 2020 | Comprehensive bearing-diagnostics deep-learning review | Compare representation choices, M2 |
| Zhang, Zhao and Lin, 2021 | CWRU machine-learning review | Understand differences in published protocols, M2 |
| Hendriks, Dumond and Knox, 2022 | Better CWRU benchmarking | Recording/condition separation and robustness, M4/M8 |
| Wang and Lin, 2011 | Wavelet packets and spectral kurtosis | Physically motivated feature selection, M5 |
| Li et al., 2013 | Wavelet-filter spectral kurtosis and envelope analysis | Resonance-band and envelope features, M5 |
| Mishra et al., 2021 | Multi-domain features and SVM | Classical baseline, M5/M6 |
| Alonso-Gonzalez et al., 2023 | Envelope analysis with machine learning | Interpretable vibration features, M5/M6 |
| Li et al., 2020 | Raw-signal 1D residual CNN | Deep-model comparison, M7 |
| Mohiuddin et al., 2021 | CNN performance on bearing faults | Classification design, M7 |
| Bapir and Aydin, 2022 | Comparative 1D-CNN analysis | Architecture choices, M7 |
| Deveci et al., 2021 | Spectrogram transfer learning | Optional 2D extension after M7 |
| Chang et al., 2023 | CNN-LSTM hybrid | Optional temporal-model comparison |
| Shen et al., 2015 | SVD and transfer learning | Cross-condition motivation, M8 |
| Zhou and Barati Farimani, 2024 | FaultFormer and self-supervised pretraining | Optional future work after reliable baselines |
| Tuo et al., 2024 | CWT and convolutional attention under noise | Optional representation/noise study |
| Bouchareb et al., 2024 | Spectrograms and pretrained GoogLeNet | Optional 2D transfer-learning study |

The original reading list also included sparse-autoencoder condition monitoring and stacked-autoencoder hydraulic-pump diagnosis. Retain these as background candidates; the supplied repository list did not provide enough bibliographic information to invent a verified citation. Neither approach is implemented here.

## Reading order for the next implementation

1. CWRU documentation and the existing dataset review for labels, channels and conditions.
2. Benchmarking work and the [data protocol](../docs/data_and_evaluation_protocol.md) before creating model splits.
3. Statistical, spectral and envelope methods for M5–M6.
4. Compact 1D-CNN methods for comparison on the same data split.
5. Transfer learning, attention and transformers only after the baseline and robustness experiments.

For each paper, retain the task, dataset subset, class definition, preprocessing, split unit, overlap, metrics, limitations and relevance. Scores using different subsets or evaluation protocols are not directly comparable.

## References recorded in the supplied reports

- Neupane and Seok, *Bearing Fault Detection and Diagnosis Using Case Western Reserve University Dataset With Deep Learning Approaches: A Review* (2020). DOI: [10.1109/ACCESS.2020.2990528](https://doi.org/10.1109/ACCESS.2020.2990528).
- S. Zhang et al., *Deep Learning Algorithms for Bearing Fault Diagnostics — A Comprehensive Review* (2020). DOI: [10.1109/ACCESS.2020.2972859](https://doi.org/10.1109/ACCESS.2020.2972859).
- X. Zhang, Zhao and Lin, *Machine Learning Based Bearing Fault Diagnosis Using the Case Western Reserve University Data: A Review* (2021). DOI: [10.1109/ACCESS.2021.3128669](https://doi.org/10.1109/ACCESS.2021.3128669).
- Hendriks, Dumond and Knox, *Towards better benchmarking using the CWRU bearing fault dataset* (2022). DOI: [10.1016/j.ymssp.2021.108732](https://doi.org/10.1016/j.ymssp.2021.108732).
- Wang and Lin, *Fault diagnosis of rolling bearings based on wavelet packet and spectral kurtosis* (2011). DOI: [10.1109/ICICTA.2011.173](https://doi.org/10.1109/ICICTA.2011.173).
- W. Li et al., *Envelope analysis by wavelet-filter based spectral kurtosis for bearing health monitoring* (2013). DOI: [10.1109/I2MTC.2013.6555709](https://doi.org/10.1109/I2MTC.2013.6555709).
- Mishra et al., *Multi-domain Bearing Fault Diagnosis using Support Vector Machine* (2021). DOI: [10.1109/GUCON50781.2021.9573613](https://doi.org/10.1109/GUCON50781.2021.9573613).
- Alonso-Gonzalez et al., *Bearing Fault Diagnosis With Envelope Analysis and Machine Learning Approaches Using CWRU Dataset* (2023). DOI: [10.1109/ACCESS.2023.3283466](https://doi.org/10.1109/ACCESS.2023.3283466).
- C. Li et al., *A Novel Bearing Fault Diagnosis of Raw Signals Based on 1D Residual Convolution Neural Network* (2020). DOI: [10.1109/HPBDIS49115.2020.9130567](https://doi.org/10.1109/HPBDIS49115.2020.9130567).
- Mohiuddin, Islam and Kabir, *Performance Analysis of Bearing fault diagnosis using Convolutional Neural Network* (2021). DOI: [10.1109/GUCON50781.2021.9573710](https://doi.org/10.1109/GUCON50781.2021.9573710).
- Bapir and Aydin, *A comparative Analysis of 1D Convolutional Neural Networks for Bearing Fault Diagnosis* (2022). DOI: [10.1109/DASA54658.2022.9765229](https://doi.org/10.1109/DASA54658.2022.9765229).
- Deveci et al., *A Comparison of Deep Transfer Learning Methods on Bearing Fault Detection* (2021). DOI: [10.1109/FiCloud49777.2021.00048](https://doi.org/10.1109/FiCloud49777.2021.00048).
- Chang, Chen and Chen, *Bearing Fault Diagnosis Based on an Advanced Method: ID-CNN-LSTM* (2023; title as supplied). DOI: [10.1109/ECEI57668.2023.10105356](https://doi.org/10.1109/ECEI57668.2023.10105356).
- Shen et al., *Bearing fault diagnosis based on SVD feature extraction and transfer learning classification* (2015). DOI: [10.1109/PHM.2015.7380088](https://doi.org/10.1109/PHM.2015.7380088).
- Zhou and Barati Farimani, *FaultFormer: Pretraining Transformers for Adaptable Bearing Fault Classification* (2024). DOI: [10.1109/ACCESS.2024.3399670](https://doi.org/10.1109/ACCESS.2024.3399670).
- Tuo et al., *Bearing Fault Diagnosis in Noisy Environment Based on Continuous Wavelet Transform and Convolutional Attention Fusion Network* (2024). DOI: [10.1109/CAC63892.2024.10865779](https://doi.org/10.1109/CAC63892.2024.10865779).
- Bouchareb et al., *Enhanced Bearing Fault Diagnosis Using Transfer Learning and Spectrograms with Pretrained GoogLeNet* (2024). DOI: [10.1109/ICTIS62692.2024.10894215](https://doi.org/10.1109/ICTIS62692.2024.10894215).

## Primary dataset and transform documentation

- [CWRU overview](https://engineering.case.edu/bearingdatacenter/welcome)
- [Apparatus and procedures](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures)
- [Data-file conventions](https://engineering.case.edu/bearingdatacenter/download-data-file)
- [Bearing geometry and defect multipliers](https://engineering.case.edu/bearingdatacenter/bearing-information)
- [Fault specifications](https://engineering.case.edu/bearingdatacenter/fault-specifications)
- [12 kHz drive-end table](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data)
- [48 kHz drive-end table](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)
- [12 kHz fan-end table](https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data)
- [Normal baseline](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)
- [MathWorks Fourier-to-wavelet explanation](https://www.mathworks.com/help/wavelet/gs/from-fourier-analysis-to-wavelet-analysis.html)
- [MathWorks continuous and discrete wavelet transforms](https://www.mathworks.com/help/wavelet/gs/continuous-and-discrete-wavelet-transforms.html)
