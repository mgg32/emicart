import tempfile
import unittest
from pathlib import Path

import numpy as np

from emicart.ui.import_export import (
    build_binary_payload,
    read_csv_import,
    read_mat_import,
    read_npz_import,
    write_csv_export,
)


class ImportExportRoundtripTests(unittest.TestCase):
    def _sample_traces(self):
        return [
            {
                "label": "Roundtrip Trace",
                "window": "Hann",
                "effective_rbw_hz": 10000.0,
                "color": "#1D4ED8",
                "probe_name": "Direct Voltage (No Probe)",
                "probe_units": "dBuV",
                "probe_impedance_ohms": None,
                "probe_v_per_m_gain": None,
                "freqs": np.array([1.0e4, 2.0e4, 5.0e4], dtype=float),
                "original": np.array([90.0, 85.0, 80.0], dtype=float),
                "windowed": np.array([89.0, 84.2, 79.7], dtype=float),
                "volts": np.array([0.01, -0.01, 0.005], dtype=float),
                "sample_rate": 1.0e6,
            }
        ]

    def _assert_trace(self, traces):
        self.assertEqual(len(traces), 1)
        t = traces[0]
        self.assertEqual(t["label"], "Roundtrip Trace")
        self.assertEqual(t["probe_units"], "dBuV")
        np.testing.assert_allclose(t["freqs"], np.array([1.0e4, 2.0e4, 5.0e4], dtype=float))
        np.testing.assert_allclose(t["original"], np.array([90.0, 85.0, 80.0], dtype=float))
        np.testing.assert_allclose(t["windowed"], np.array([89.0, 84.2, 79.7], dtype=float))

    def test_csv_and_npz_roundtrip(self):
        traces = self._sample_traces()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            csv_path = base / "sample.csv"
            npz_path = base / "sample.npz"
            meta_expected = {
                "Standard": "No Standard",
                "LimitCurve": "No Limit Curve",
                "Probe": "Direct Voltage (No Probe)",
                "WindowSelection": "Raw FFT (No RBW Correction)",
            }

            write_csv_export(
                csv_path=csv_path,
                traces=traces,
                standard=meta_expected["Standard"],
                limit_curve=meta_expected["LimitCurve"],
                selected_probe=meta_expected["Probe"],
                window_selection=meta_expected["WindowSelection"],
            )
            meta_csv, traces_csv = read_csv_import(
                str(csv_path),
                default_probe_snapshot={
                    "probe_name": meta_expected["Probe"],
                    "probe_units": "dBuV",
                    "probe_impedance_ohms": None,
                    "probe_v_per_m_gain": None,
                },
            )
            for key, val in meta_expected.items():
                self.assertEqual(meta_csv.get(key, ""), val)
            self._assert_trace(traces_csv)

            np.savez_compressed(
                npz_path,
                **build_binary_payload(
                    traces=traces,
                    standard=meta_expected["Standard"],
                    limit_curve=meta_expected["LimitCurve"],
                    selected_probe=meta_expected["Probe"],
                    window_selection=meta_expected["WindowSelection"],
                ),
            )
            meta_npz, traces_npz = read_npz_import(str(npz_path))
            for key, val in meta_expected.items():
                self.assertEqual(meta_npz.get(key, ""), val)
            self._assert_trace(traces_npz)

    def test_mat_roundtrip_if_scipy_present(self):
        try:
            from scipy.io import savemat  # type: ignore
        except Exception:
            self.skipTest("scipy not installed")

        traces = self._sample_traces()
        with tempfile.TemporaryDirectory() as tmp:
            mat_path = Path(tmp) / "sample.mat"
            savemat(
                mat_path,
                build_binary_payload(
                    traces=traces,
                    standard="No Standard",
                    limit_curve="No Limit Curve",
                    selected_probe="Direct Voltage (No Probe)",
                    window_selection="Raw FFT (No RBW Correction)",
                ),
                do_compression=True,
            )
            meta_mat, traces_mat = read_mat_import(str(mat_path))
            self.assertEqual(meta_mat.get("Standard"), "No Standard")
            self._assert_trace(traces_mat)


if __name__ == "__main__":
    unittest.main()
