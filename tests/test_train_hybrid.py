"""Focused unit tests for the aligned hybrid loader and training inputs."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import train_hybrid as h


class HybridDatasetTests(unittest.TestCase):
    def make_dataset(self, root):
        records = []
        for name, folder in h.DRIVE_CLASSES.items():
            for p in h.PARTITIONS:
                rid = f"{name}-{p}"
                records.append(dict(record_id=rid, group_id=rid,
                                    partition=p, class_name=name))
            for i, storage in enumerate(h.STORAGE_SPLITS):
                path = root / storage / folder
                path.mkdir(parents=True)
                with (path / "index.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "record_id", "group_id", "partition", "start_sample", "split_70_30"])
                    writer.writeheader()
                    for p in h.PARTITIONS:
                        rid = f"{name}-{p}"
                        writer.writerow(dict(record_id=rid, group_id=rid, partition=p,
                                             start_sample=i*2048, split_70_30=storage))
                for v in h.VARIANTS:
                    target = path / v
                    target.mkdir()
                    np.save(target / "signal.npy", np.ones((3, 2048), np.float32))
                    np.save(target / "envelope_frames.npy",
                            np.ones((3, *h.ENV_SHAPE), np.float32))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(records))
        return manifest

    def test_group_partition_and_aligned_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_dataset(root)
            with patch.object(h, "MANIFEST", manifest), patch.object(h, "verify_clean"), \
                 patch.object(h, "extract_sequences",
                              side_effect=lambda x: np.ones((len(x), *h.CNN_SHAPE), np.float32)):
                parts = h.load_hybrid_partitions(root)
            groups = {}
            for p in h.PARTITIONS:
                x, y = h.combine(parts[p])
                self.assertEqual(x["cnn_features"].shape, (16, *h.CNN_SHAPE))
                self.assertEqual(x["envelope_features"].shape, (16, *h.ENV_SHAPE))
                self.assertEqual(set(y), set(h.CLASSES.values()))
                groups[p] = set(parts[p]["Clean"][2])
                self.assertEqual(groups[p], set(parts[p]["Noisy"][2]))
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])

    def test_misaligned_array_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npy"
            np.save(path, np.zeros((2, 25, 68), np.float32))
            with self.assertRaises(ValueError):
                h.load_array(path, (3, *h.ENV_SHAPE))

    def test_duplicate_window_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_dataset(root)
            index = root / "test_30" / "Healthy" / "index.csv"
            index.write_text(index.read_text().replace("2048,test_30", "0,test_30"))
            with patch.object(h, "MANIFEST", manifest), patch.object(h, "verify_clean"), \
                 patch.object(h, "extract_sequences",
                              side_effect=lambda x: np.ones((len(x), *h.CNN_SHAPE), np.float32)):
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    h.load_hybrid_partitions(root)


if __name__ == "__main__":
    unittest.main()
