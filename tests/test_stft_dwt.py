import unittest

import numpy as np

from src.stft_dwt import (
    FRAME_SAMPLES, FREQUENCY_BINS, POWER_FLOOR, WINDOW_SAMPLES, extract_sequences,
)


class SequenceTests(unittest.TestCase):
    def test_stft_frequency_bin_and_time_axis(self):
        bin_index = 32  # 1500 Hz at 12 kHz
        samples = np.arange(WINDOW_SAMPLES)
        signal = np.sin(2 * np.pi * bin_index * samples / FRAME_SAMPLES)
        features = extract_sequences(signal[None, :])

        self.assertEqual(features.shape, (1, 15, 134))
        stft = features[0, :, :FREQUENCY_BINS]
        np.testing.assert_array_equal(stft.argmax(axis=-1), bin_index)
        np.testing.assert_allclose(stft[:, bin_index], 10 * np.log10(0.25), atol=1e-5)

    def test_dwt_bands_and_stft_share_the_same_frame_support(self):
        signal = np.zeros((1, WINDOW_SAMPLES))
        signal[0, :FRAME_SAMPLES] = 1.0
        features = extract_sequences(signal)[0]
        floor_db = 10 * np.log10(POWER_FLOOR)

        self.assertAlmostEqual(features[0, 0], 0.0, places=5)
        self.assertAlmostEqual(features[0, FREQUENCY_BINS], 10 * np.log10(16), places=5)
        np.testing.assert_allclose(features[0, FREQUENCY_BINS + 1:], floor_db)
        np.testing.assert_allclose(features[2:], floor_db)

    def test_rejects_incorrect_window_size_and_nonfinite_samples(self):
        with self.assertRaises(ValueError):
            extract_sequences(np.zeros((1, WINDOW_SAMPLES - 1)))
        invalid = np.zeros((1, WINDOW_SAMPLES))
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_sequences(invalid)


if __name__ == "__main__":
    unittest.main()
