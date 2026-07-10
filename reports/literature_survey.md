# Literature Survey

## Topic

Bearing Fault Diagnosis from Vibration Signals using signal processing, machine learning, and deep learning.

## Summary

The literature in bearing fault diagnosis shows a gradual shift from handcrafted signal-processing features to learned deep representations. Classical approaches use time-domain features, FFT, envelope analysis, wavelet packet decomposition, and spectral kurtosis to highlight fault-sensitive vibration components. More recent methods use CNNs, transfer learning, sparse autoencoders, and domain-adaptation methods to improve robustness under changing loads and operating conditions.

## Papers to Review

| No. | Paper | Core Method | Relevance |
|---:|---|---|---|
| 1 | Fault diagnosis of rolling bearings based on wavelet packet and spectral kurtosis | Wavelet packet + spectral kurtosis | Shows signal-processing-based fault feature extraction |
| 2 | Envelope analysis by wavelet-filter based spectral kurtosis for bearing health monitoring | Envelope analysis + spectral kurtosis | Useful for weak impulsive fault signatures |
| 3 | Bearing fault diagnosis based on SVD feature extraction and transfer learning classification | SVD + transfer learning | Useful for cross-load / cross-condition robustness |
| 4 | Intelligent condition based monitoring of rotating machines using sparse auto-encoders | Sparse autoencoder | Shows learned feature extraction for rotating machinery |
| 5 | Fault diagnosis of hydraulic pump based on stacked autoencoders | Stacked autoencoder | Mechanical fault-diagnosis analogue using deep learning |

## How These Papers Support This Project

- Papers 1 and 2 justify engineered signal features and envelope analysis.
- Paper 3 supports testing robustness across different operating conditions.
- Papers 4 and 5 support deep feature-learning extensions.
- The project implementation starts with interpretable SVM/Random Forest baselines and then extends to CNN-based models.

## Notes for Final Report

In the final report, each paper should be summarized using:

1. Objective
2. Dataset
3. Methodology
4. Results
5. Limitations
6. Relevance to our project
