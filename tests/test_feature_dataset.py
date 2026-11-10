import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.data_pipeline import PARTITIONS
from src.feature_dataset import (
    CLASS_FOLDERS, FRAME_COUNT, INDEX_COLUMNS, SCHEMA, VARIANTS, add_recording_noise, load_partitions,
)
from src.stft_dwt import DWT_BANDS, FREQUENCY_BINS, WINDOW_SAMPLES


class FeatureDatasetTests(unittest.TestCase):
    def test_noise_is_reproducible_at_target_snr_with_consistent_overlap(self):
        clean = np.ones((2, WINDOW_SAMPLES))
        start = WINDOW_SAMPLES // 2
        starts = np.array([0, start])

        def add_noise():
            return add_recording_noise(clean, starts, 10, np.random.default_rng(42)) - clean

        noise = add_noise()
        np.testing.assert_array_equal(noise, add_noise())
        self.assertAlmostEqual(10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2)), 10)
        overlap = noise[0, start:] - noise[1, :start]
        np.testing.assert_allclose(overlap, overlap[0], atol=1e-12)

    def test_pair_loader_rejects_recording_in_more_than_one_split(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset.json").write_text(json.dumps({"schema": SCHEMA}))
            for folder_name in CLASS_FOLDERS.values():
                folder = root / folder_name
                folder.mkdir()
                with (folder / "index.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(INDEX_COLUMNS)
                    for split in PARTITIONS:
                        group = f"{folder_name}-{split}"
                        writer.writerow((split, group, group, 0))
                for variant in VARIANTS:
                    target = folder / variant
                    target.mkdir()
                    for key, channels in (("stft", FREQUENCY_BINS), ("dwt", len(DWT_BANDS))):
                        values = np.zeros((len(PARTITIONS), FRAME_COUNT, channels), dtype=np.float32)
                        np.savez(target / f"{key}.npz", values=values)

            splits = load_partitions(root)
            for split in PARTITIONS:
                clean, noisy = (splits[split][variant] for variant in VARIANTS)
                np.testing.assert_array_equal(clean[2], noisy[2])
                self.assertEqual(len(clean[2]), 4)

            index = root / "Healthy" / "index.csv"
            index.write_text(index.read_text().replace("Healthy-validation,Healthy-validation,",
                                                      "Healthy-validation,Healthy-train,"))
            with self.assertRaisesRegex(ValueError, "Invalid split"):
                load_partitions(root)
