import unittest

import numpy as np

from emicart.analysis.fft import apply_frequency_domain_window_by_rbw


class FFTWindowingByRBWTests(unittest.TestCase):
    def test_windowing_does_not_cross_rbw_discontinuity(self):
        n = 12
        freqs = np.arange(1, n + 1, dtype=float) * 1_000.0
        mags_db = np.full(n, -200.0, dtype=float)
        mags_db[5] = 0.0  # High-energy bin at end of first RBW segment.

        rbw = np.array([3_000.0] * 6 + [6_000.0] * 6, dtype=float)
        out = apply_frequency_domain_window_by_rbw(
            mags_db=mags_db,
            freqs_hz=freqs,
            target_rbw_hz=rbw,
            window_name="boxcar",
        )

        # If the kernel crossed the discontinuity at index 6, this bin would rise.
        self.assertLess(out[6], -180.0)
        # Segment-local smoothing should still affect neighboring bins in segment 1.
        self.assertGreater(out[4], -80.0)

    def test_invalid_or_missing_rbw_returns_original(self):
        freqs = np.arange(1, 8, dtype=float) * 1_000.0
        mags_db = np.array([10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0], dtype=float)
        out = apply_frequency_domain_window_by_rbw(
            mags_db=mags_db,
            freqs_hz=freqs,
            target_rbw_hz=[None] * len(mags_db),
            window_name="hamming",
        )
        np.testing.assert_allclose(out, mags_db, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
