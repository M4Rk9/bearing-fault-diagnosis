import unittest

import numpy as np

from src.stft_dwt import (
    FRAME_HOP, FRAME_SAMPLES, FREQUENCY_BINS, POWER_FLOOR, WINDOW_SAMPLES, extract_sequences,
)


class SequenceTests(unittest.TestCase):
    def test_stft_frequency_bin_and_time_axis(self):
        bin_index = 64  # 1500 Hz at 12 kHz
        samples = np.arange(WINDOW_SAMPLES)
        signal = np.sin(2 * np.pi * bin_index * samples / FRAME_SAMPLES)
        features = extract_sequences(signal[None, :])

        self.assertEqual(features.shape, (1, 25, 263))
        stft = features[0, :, :FREQUENCY_BINS]
        np.testing.assert_array_equal(stft.argmax(axis=-1), bin_index)
        np.testing.assert_allclose(stft[:, bin_index], 10 * np.log10(0.25), atol=1e-5)

    def test_stft_separates_two_nearby_tones(self):
        samples = np.arange(WINDOW_SAMPLES)
        signal = (np.sin(2 * np.pi * 64 * samples / FRAME_SAMPLES)
                  + np.sin(2 * np.pi * 67 * samples / FRAME_SAMPLES))
        spectrum = extract_sequences(signal[None, :])[0, 0, :FREQUENCY_BINS]
        self.assertGreater(spectrum[64], spectrum[65])
        self.assertGreater(spectrum[67], spectrum[66])
        np.testing.assert_allclose(spectrum[[64, 67]], 10 * np.log10(0.25), atol=1e-5)

    def test_dwt_bands_and_stft_share_the_same_frame_support(self):
        signal = np.zeros((1, WINDOW_SAMPLES))
        signal[0, :FRAME_SAMPLES] = 1.0
        features = extract_sequences(signal)[0]
        floor_db = 10 * np.log10(POWER_FLOOR)

        self.assertAlmostEqual(features[0, 0], 0.0, places=5)
        self.assertAlmostEqual(features[0, FREQUENCY_BINS], 10 * np.log10(32), places=5)
        np.testing.assert_allclose(features[0, FREQUENCY_BINS + 1:], floor_db)
        np.testing.assert_allclose(features[FRAME_SAMPLES // FRAME_HOP:], floor_db)

    def test_rejects_incorrect_window_size_and_nonfinite_samples(self):
        with self.assertRaises(ValueError):
            extract_sequences(np.zeros((1, WINDOW_SAMPLES - 1)))
        invalid = np.zeros((1, WINDOW_SAMPLES))
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_sequences(invalid)


if __name__ == "__main__":
    unittest.main()
