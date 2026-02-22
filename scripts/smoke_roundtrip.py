from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emicart.ui.import_export import (
    build_binary_payload,
    read_csv_import,
    read_mat_import,
    read_npz_import,
    write_csv_export,
)


def _sample_traces() -> list[dict]:
    freqs = np.array([1.0e4, 2.0e4, 5.0e4, 1.0e5], dtype=float)
    original = np.array([90.0, 87.5, 84.0, 81.0], dtype=float)
    windowed = np.array([89.2, 86.9, 83.6, 80.7], dtype=float)
    volts = np.array([0.01, -0.02, 0.015, -0.005], dtype=float)
    return [
        {
            "label": "Smoke Trace 1",
            "window": "Hamming",
            "effective_rbw_hz": 10000.0,
            "color": "#1D4ED8",
            "probe_name": "Direct Voltage (No Probe)",
            "probe_units": "dBuV",
            "probe_impedance_ohms": None,
            "probe_v_per_m_gain": None,
            "freqs": freqs,
            "original": original,
            "windowed": windowed,
            "volts": volts,
            "sample_rate": 1.0e6,
        }
    ]


def _assert_import(meta: dict, traces: list[dict], expected_window: str) -> None:
    assert meta.get("Standard", "") == "No Standard"
    assert meta.get("LimitCurve", "") == "No Limit Curve"
    assert meta.get("Probe", "") == "Direct Voltage (No Probe)"
    assert meta.get("WindowSelection", "") == expected_window
    assert len(traces) == 1
    t = traces[0]
    assert t["label"] == "Smoke Trace 1"
    assert t["window"] == "Hamming"
    assert t["probe_units"] == "dBuV"
    np.testing.assert_allclose(np.asarray(t["freqs"], dtype=float), np.array([1.0e4, 2.0e4, 5.0e4, 1.0e5]))
    np.testing.assert_allclose(np.asarray(t["original"], dtype=float), np.array([90.0, 87.5, 84.0, 81.0]))
    np.testing.assert_allclose(np.asarray(t["windowed"], dtype=float), np.array([89.2, 86.9, 83.6, 80.7]))


def main() -> int:
    traces = _sample_traces()
    standard = "No Standard"
    limit_curve = "No Limit Curve"
    selected_probe = "Direct Voltage (No Probe)"
    window_selection = "Raw FFT (No RBW Correction)"
    default_probe_snapshot = {
        "probe_name": selected_probe,
        "probe_units": "dBuV",
        "probe_impedance_ohms": None,
        "probe_v_per_m_gain": None,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)

        # CSV roundtrip
        csv_path = out_dir / "smoke.csv"
        write_csv_export(
            csv_path=csv_path,
            traces=traces,
            standard=standard,
            limit_curve=limit_curve,
            selected_probe=selected_probe,
            window_selection=window_selection,
        )
        meta_csv, traces_csv = read_csv_import(str(csv_path), default_probe_snapshot=default_probe_snapshot)
        _assert_import(meta_csv, traces_csv, window_selection)

        # NPZ roundtrip
        npz_path = out_dir / "smoke.npz"
        np.savez_compressed(
            npz_path,
            **build_binary_payload(
                traces=traces,
                standard=standard,
                limit_curve=limit_curve,
                selected_probe=selected_probe,
                window_selection=window_selection,
            ),
        )
        meta_npz, traces_npz = read_npz_import(str(npz_path))
        _assert_import(meta_npz, traces_npz, window_selection)

        # MAT roundtrip (optional if scipy is available)
        try:
            from scipy.io import savemat  # type: ignore
        except Exception:
            print("PASS: CSV+NPZ roundtrip succeeded (MAT skipped: scipy not installed).")
            return 0

        mat_path = out_dir / "smoke.mat"
        savemat(
            mat_path,
            build_binary_payload(
                traces=traces,
                standard=standard,
                limit_curve=limit_curve,
                selected_probe=selected_probe,
                window_selection=window_selection,
            ),
            do_compression=True,
        )
        meta_mat, traces_mat = read_mat_import(str(mat_path))
        _assert_import(meta_mat, traces_mat, window_selection)

    print("PASS: CSV, NPZ, and MAT roundtrip succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
