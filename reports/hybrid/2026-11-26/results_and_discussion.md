# Results and discussion

All values below are derived from the saved experiment; no simulated scores.

## Test performance

| Model | Condition | Level | N | Accuracy | Macro-F1 | Log loss |
|---|---|---|---:|---:|---:|---:|
| hybrid | Clean | windows | 1454 | 100.00% | 1.0000 | 0.0018 |
| hybrid | Clean | recordings | 10 | 100.00% | 1.0000 | 0.0022 |
| hybrid | Noisy | windows | 1454 | 96.70% | 0.9657 | 0.0644 |
| hybrid | Noisy | recordings | 10 | 100.00% | 1.0000 | 0.0571 |
| baseline | Clean | windows | 1454 | 97.87% | 0.9778 | 0.0532 |
| baseline | Clean | recordings | 10 | 100.00% | 1.0000 | 0.0375 |
| baseline | Noisy | windows | 1454 | 94.29% | 0.9400 | 0.2399 |
| baseline | Noisy | recordings | 10 | 90.00% | 0.9143 | 0.1181 |

## Observations

- **hybrid:** Minimum validation loss was 0.0001 at epoch 10. Training and validation accuracy at that epoch were 100.00% and 100.00%; the observed gap was +0.00 percentage points. Inspect both learning curves; a gap alone cannot establish overfitting because validation recordings also differ in load.
- hybrid / Noisy: largest off-diagonal window error was inner_race classified as outer_race (48 windows).
- hybrid: clean-to-noisy accuracy change was -3.30 percentage points.
- **baseline:** Minimum validation loss was 0.0009 at epoch 12. Training and validation accuracy at that epoch were 100.00% and 99.97%; the observed gap was +0.03 percentage points. Inspect both learning curves; a gap alone cannot establish overfitting because validation recordings also differ in load.
- baseline / Clean: largest off-diagonal window error was inner_race classified as outer_race (29 windows).
- baseline / Noisy: largest off-diagonal window error was inner_race classified as outer_race (83 windows).
- baseline: clean-to-noisy accuracy change was -3.58 percentage points.
- Clean windows: hybrid minus baseline accuracy +2.13 percentage points; macro-F1 change +0.0222.
- Clean recordings: hybrid minus baseline accuracy +0.00 percentage points; macro-F1 change +0.0000.
- Noisy windows: hybrid minus baseline accuracy +2.41 percentage points; macro-F1 change +0.0256.
- Noisy recordings: hybrid minus baseline accuracy +10.00 percentage points; macro-F1 change +0.0857.

## Limitations

The test set is a held-out motor load from the same CWRU laboratory cohort. Overlapping windows and clean/noisy pairs are correlated; the number of independent recordings is much smaller than the window count. The experiment uses one seed and does not establish statistical significance or deployment accuracy. Hybrid-versus-baseline differences combine architecture and feature changes; an ablation study is required to isolate the contribution of Envelope or LSTM alone. Report the actual noise configuration only when independently verified. A small gap or high accuracy does not by itself rule out overfitting or underfitting.
