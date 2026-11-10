# Bearing vibration features

The four class folders are [Healthy](Healthy), [Inner Race](Inner%20Race), [Outer Race](Outer%20Race) and [Ball](Ball). Each folder includes a preview of the clean and noisy versions of the same validation window.

Generate the complete dataset from the repository root after installing the dependencies.

```bash
python -m pip install -r requirements.txt
python -m src.feature_dataset
python -m src.train_cnn
```

Alternatively, extract the prepared archive into `Datasets` so that `Datasets/Features/dataset.json` exists, then run the training command. Use `--output` with the export command and `--dataset` with the training command if you want to keep the data elsewhere. The complete arrays are reproducible from the raw recordings and are stored separately from the Git-tracked previews.

## Data in each class folder

| Path | Contents | Shape per class |
| --- | --- | --- |
| `Clean/signal.npz`, `Noisy/signal.npz` | Paired mean-centered vibration windows | `(windows, 2048)` |
| `Clean/stft.npz`, `Noisy/stft.npz` | Hann STFT log-power spectrum on each frame | `(windows, 25, 257)` |
| `Clean/dwt.npz`, `Noisy/dwt.npz` | Log mean squared db4 coefficients for A5, D5, D4, D3, D2 and D1 | `(windows, 25, 6)` |
| `index.csv` | Original record ID, group, partition and window start | One row per aligned clean/noisy pair |

The NPZ array key is `values`. All arrays use the same row order within a class. The dataset uses 12 kHz windows and 512-sample transform frames (42.67 ms) with a 64-sample hop (5.33 ms). The noise is white Gaussian with seed 42 and a target SNR of 10 dB per recording, measured across its windows. Noise is drawn once per recording, sliced at the same offsets as the clean windows, then mean-centered and scaled. Export another noise level with `--snr-db` and a separate `--output` directory.

The dataset uses all 40 uploaded CWRU recordings. Fault diameters are 0.007, 0.014 and 0.021 inch, and the uploaded outer-race recordings use the centered position. The export yields 4,643 clean/noisy pairs. The 2,291 training pairs come from 0 and 1 HP, the 1,174 validation pairs from 2 HP, and the 1,178 test pairs from 3 HP. Both members of each pair stay in the partition assigned to the original recording. The CNN concatenates STFT and DWT features along the channel axis, yielding 263 features per time frame. It trains on both variants and reports held-out clean and noisy performance separately.

The detailed export uses a 512-point Hann STFT and a five-level db4 DWT. The STFT bin spacing is 23.4375 Hz. The longer frame improves frequency discrimination and averages over a longer time interval. The 64-sample hop provides a denser time grid. The loader also accepts the previous v1 dataset. When both datasets are present, pass the detailed export directory to `--dataset`.
