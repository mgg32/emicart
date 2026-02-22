from __future__ import annotations

from csv import reader as csv_reader
from csv import writer as csv_writer
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def build_binary_payload(
    traces: List[dict],
    standard: str,
    limit_curve: str,
    selected_probe: str,
    window_selection: str,
) -> dict:
    return {
        "standard": np.array([standard], dtype=object),
        "limit_curve": np.array([limit_curve], dtype=object),
        "selected_probe": np.array([selected_probe], dtype=object),
        "window_selection": np.array([window_selection], dtype=object),
        "trace_labels": np.array([t["label"] for t in traces], dtype=object),
        "trace_windows": np.array([t["window"] for t in traces], dtype=object),
        "trace_effective_rbw_hz": np.array(
            [np.nan if t.get("effective_rbw_hz") is None else float(t["effective_rbw_hz"]) for t in traces],
            dtype=float,
        ),
        "trace_colors": np.array([t.get("color", "") for t in traces], dtype=object),
        "trace_probe_names": np.array([t.get("probe_name", "") for t in traces], dtype=object),
        "trace_probe_units": np.array([t.get("probe_units", "") for t in traces], dtype=object),
        "trace_probe_impedance_ohms": np.array(
            [np.nan if t.get("probe_impedance_ohms") is None else float(t.get("probe_impedance_ohms")) for t in traces],
            dtype=float,
        ),
        "trace_probe_v_per_m_gain": np.array(
            [np.nan if t.get("probe_v_per_m_gain") is None else float(t.get("probe_v_per_m_gain")) for t in traces],
            dtype=float,
        ),
        "trace_freqs_hz": np.array([np.asarray(t["freqs"], dtype=float) for t in traces], dtype=object),
        "trace_original_dbuv": np.array([np.asarray(t["original"], dtype=float) for t in traces], dtype=object),
        "trace_windowed_dbuv": np.array([np.asarray(t["windowed"], dtype=float) for t in traces], dtype=object),
        "trace_volts_v": np.array([np.asarray(t.get("volts", []), dtype=float) for t in traces], dtype=object),
        "trace_sample_rate_hz": np.array(
            [np.nan if t.get("sample_rate") is None else float(t.get("sample_rate")) for t in traces],
            dtype=float,
        ),
    }


def write_csv_export(
    csv_path: Path,
    traces: List[dict],
    standard: str,
    limit_curve: str,
    selected_probe: str,
    window_selection: str,
) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv_writer(f)
        w.writerow(["Standard", standard])
        w.writerow(["LimitCurve", limit_curve])
        w.writerow(["Probe", selected_probe])
        w.writerow(["WindowSelection", window_selection])
        w.writerow([])
        w.writerow(
            [
                "CaptureLabel",
                "Window",
                "Effective_RBW_Hz",
                "TraceColor",
                "ProbeName",
                "ProbeUnits",
                "ProbeImpedanceOhms",
                "ProbeVperMperV",
                "Frequency(Hz)",
                "OriginalFFT(dBuV)",
                "WindowedFFT(dBuV)",
            ]
        )
        for trace in traces:
            rbw_text = "N/A" if trace["effective_rbw_hz"] is None else f"{trace['effective_rbw_hz']:.6g}"
            for i in range(len(trace["freqs"])):
                w.writerow(
                    [
                        trace["label"],
                        trace["window"],
                        rbw_text,
                        trace.get("color", ""),
                        trace.get("probe_name", ""),
                        trace.get("probe_units", ""),
                        "" if trace.get("probe_impedance_ohms") is None else f"{trace.get('probe_impedance_ohms'):.12g}",
                        "" if trace.get("probe_v_per_m_gain") is None else f"{trace.get('probe_v_per_m_gain'):.12g}",
                        f"{trace['freqs'][i]:.12g}",
                        f"{trace['original'][i]:.12g}",
                        f"{trace['windowed'][i]:.12g}",
                    ]
                )

        w.writerow([])
        w.writerow(
            [
                "CaptureLabelTD",
                "Window",
                "Effective_RBW_Hz",
                "TraceColor",
                "ProbeName",
                "ProbeUnits",
                "ProbeImpedanceOhms",
                "ProbeVperMperV",
                "SampleRate(Hz)",
                "SampleIndex",
                "Voltage(V)",
            ]
        )
        for trace in traces:
            if trace.get("sample_rate") is None or len(trace.get("volts", [])) == 0:
                continue
            rbw_text = "N/A" if trace["effective_rbw_hz"] is None else f"{trace['effective_rbw_hz']:.6g}"
            sample_rate_text = f"{float(trace['sample_rate']):.12g}"
            volts = np.asarray(trace["volts"], dtype=float)
            for idx, v in enumerate(volts):
                w.writerow(
                    [
                        trace["label"],
                        trace["window"],
                        rbw_text,
                        trace.get("color", ""),
                        trace.get("probe_name", ""),
                        trace.get("probe_units", ""),
                        "" if trace.get("probe_impedance_ohms") is None else f"{trace.get('probe_impedance_ohms'):.12g}",
                        "" if trace.get("probe_v_per_m_gain") is None else f"{trace.get('probe_v_per_m_gain'):.12g}",
                        sample_rate_text,
                        str(idx),
                        f"{float(v):.12g}",
                    ]
                )


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    return [v]


def _optional_float(values, idx: int):
    if idx >= len(values):
        return None
    try:
        val = float(values[idx])
        return val if np.isfinite(val) else None
    except Exception:
        return None


def read_npz_import(path: str) -> Tuple[Dict[str, str], List[dict]]:
    data = np.load(path, allow_pickle=True)
    trace_labels = np.atleast_1d(data.get("trace_labels", np.array([], dtype=object)))
    trace_windows = np.atleast_1d(data.get("trace_windows", np.array([], dtype=object)))
    trace_rbws = np.atleast_1d(data.get("trace_effective_rbw_hz", np.array([], dtype=float)))
    trace_colors = np.atleast_1d(data.get("trace_colors", np.array([], dtype=object)))
    trace_probe_names = np.atleast_1d(data.get("trace_probe_names", np.array([], dtype=object)))
    trace_probe_units = np.atleast_1d(data.get("trace_probe_units", np.array([], dtype=object)))
    trace_probe_imps = np.atleast_1d(data.get("trace_probe_impedance_ohms", np.array([], dtype=float)))
    trace_probe_gains = np.atleast_1d(data.get("trace_probe_v_per_m_gain", np.array([], dtype=float)))
    trace_freqs = np.atleast_1d(data.get("trace_freqs_hz", np.array([], dtype=object)))
    trace_original = np.atleast_1d(data.get("trace_original_dbuv", np.array([], dtype=object)))
    trace_windowed = np.atleast_1d(data.get("trace_windowed_dbuv", np.array([], dtype=object)))
    trace_volts = np.atleast_1d(data.get("trace_volts_v", np.array([], dtype=object)))
    trace_sample_rates = np.atleast_1d(data.get("trace_sample_rate_hz", np.array([], dtype=float)))

    meta = {
        "Standard": str(np.atleast_1d(data.get("standard", np.array([""])))[0] or ""),
        "LimitCurve": str(np.atleast_1d(data.get("limit_curve", np.array([""])))[0] or ""),
        "Probe": str(np.atleast_1d(data.get("selected_probe", np.array([""])))[0] or ""),
        "WindowSelection": str(np.atleast_1d(data.get("window_selection", np.array([""])))[0] or ""),
    }

    traces: List[dict] = []
    for i in range(int(len(trace_labels))):
        rbw_val = _optional_float(trace_rbws, i)
        imp_val = _optional_float(trace_probe_imps, i)
        gain_val = _optional_float(trace_probe_gains, i)
        sr_val = _optional_float(trace_sample_rates, i)
        trace = {
            "label": str(trace_labels[i]),
            "window": str(trace_windows[i]) if i < len(trace_windows) else "Imported",
            "effective_rbw_hz": rbw_val,
            "color": str(trace_colors[i]) if i < len(trace_colors) else "",
            "probe_name": str(trace_probe_names[i]) if i < len(trace_probe_names) else "",
            "probe_units": str(trace_probe_units[i]) if i < len(trace_probe_units) else "",
            "probe_impedance_ohms": imp_val,
            "probe_v_per_m_gain": gain_val,
            "freqs": np.asarray(trace_freqs[i], dtype=float) if i < len(trace_freqs) else np.array([], dtype=float),
            "original": np.asarray(trace_original[i], dtype=float) if i < len(trace_original) else np.array([], dtype=float),
            "windowed": np.asarray(trace_windowed[i], dtype=float) if i < len(trace_windowed) else np.array([], dtype=float),
            "volts": np.asarray(trace_volts[i], dtype=float) if i < len(trace_volts) else np.array([], dtype=float),
            "sample_rate": sr_val,
        }
        traces.append(trace)
    return meta, traces


def read_mat_import(path: str) -> Tuple[Dict[str, str], List[dict]]:
    from scipy.io import loadmat  # type: ignore

    mat = loadmat(path, simplify_cells=True)
    trace_labels = _as_list(mat.get("trace_labels"))
    trace_windows = _as_list(mat.get("trace_windows"))
    trace_rbws = _as_list(mat.get("trace_effective_rbw_hz"))
    trace_colors = _as_list(mat.get("trace_colors"))
    trace_probe_names = _as_list(mat.get("trace_probe_names"))
    trace_probe_units = _as_list(mat.get("trace_probe_units"))
    trace_probe_imps = _as_list(mat.get("trace_probe_impedance_ohms"))
    trace_probe_gains = _as_list(mat.get("trace_probe_v_per_m_gain"))
    trace_freqs = _as_list(mat.get("trace_freqs_hz"))
    trace_original = _as_list(mat.get("trace_original_dbuv"))
    trace_windowed = _as_list(mat.get("trace_windowed_dbuv"))
    trace_volts = _as_list(mat.get("trace_volts_v"))
    trace_sample_rates = _as_list(mat.get("trace_sample_rate_hz"))

    meta = {
        "Standard": str(mat.get("standard", "") or ""),
        "LimitCurve": str(mat.get("limit_curve", "") or ""),
        "Probe": str(mat.get("selected_probe", "") or ""),
        "WindowSelection": str(mat.get("window_selection", "") or ""),
    }

    trace_count = int(len(trace_labels))

    def _trace_array(values, idx: int):
        if trace_count == 1:
            return np.asarray(values, dtype=float)
        if idx < len(values):
            return np.asarray(values[idx], dtype=float)
        return np.array([], dtype=float)

    traces: List[dict] = []
    for i in range(trace_count):
        trace = {
            "label": str(trace_labels[i]),
            "window": str(trace_windows[i]) if i < len(trace_windows) else "Imported",
            "effective_rbw_hz": _optional_float(trace_rbws, i),
            "color": str(trace_colors[i]) if i < len(trace_colors) else "",
            "probe_name": str(trace_probe_names[i]) if i < len(trace_probe_names) else "",
            "probe_units": str(trace_probe_units[i]) if i < len(trace_probe_units) else "",
            "probe_impedance_ohms": _optional_float(trace_probe_imps, i),
            "probe_v_per_m_gain": _optional_float(trace_probe_gains, i),
            "freqs": _trace_array(trace_freqs, i),
            "original": _trace_array(trace_original, i),
            "windowed": _trace_array(trace_windowed, i),
            "volts": _trace_array(trace_volts, i),
            "sample_rate": _optional_float(trace_sample_rates, i),
        }
        traces.append(trace)
    return meta, traces


def read_csv_import(path: str, default_probe_snapshot: dict) -> Tuple[Dict[str, str], List[dict]]:
    with open(path, "r", newline="") as f:
        rows = list(csv_reader(f))

    if not rows:
        return {}, []

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "CaptureLabel":
            header_idx = i
            break

    metadata: Dict[str, str] = {}
    meta_scan_end = header_idx if header_idx is not None else min(len(rows), 20)
    for row in rows[:meta_scan_end]:
        if len(row) < 2:
            continue
        key = row[0].strip()
        value = row[1].strip()
        if key:
            metadata[key] = value

    if header_idx is not None:
        td_header_idx = None
        for i, row in enumerate(rows):
            if row and row[0].strip() == "CaptureLabelTD":
                td_header_idx = i
                break

        grouped = {}
        fft_rows_end = td_header_idx if td_header_idx is not None else len(rows)
        for row in rows[header_idx + 1 : fft_rows_end]:
            if len(row) < 7:
                continue
            label = row[0].strip()
            if not label:
                continue
            if len(row) >= 11:
                probe_name = row[4].strip()
                probe_units = row[5].strip()
                probe_imp = row[6].strip()
                probe_gain = row[7].strip()
                freq_idx, orig_idx, win_idx = 8, 9, 10
            else:
                probe_name = default_probe_snapshot.get("probe_name", "")
                probe_units = default_probe_snapshot.get("probe_units", "")
                probe_imp_value = default_probe_snapshot.get("probe_impedance_ohms")
                probe_gain_value = default_probe_snapshot.get("probe_v_per_m_gain")
                probe_imp = "" if probe_imp_value is None else str(probe_imp_value)
                probe_gain = "" if probe_gain_value is None else str(probe_gain_value)
                freq_idx, orig_idx, win_idx = 4, 5, 6
            try:
                freq = float(row[freq_idx])
                orig = float(row[orig_idx])
                win = float(row[win_idx])
            except ValueError:
                continue

            key = (
                label,
                row[1].strip(),
                row[2].strip(),
                row[3].strip(),
                probe_name,
                probe_units,
                probe_imp,
                probe_gain,
            )
            grouped.setdefault(key, {"freqs": [], "orig": [], "win": []})
            grouped[key]["freqs"].append(freq)
            grouped[key]["orig"].append(orig)
            grouped[key]["win"].append(win)

        waveform_grouped = {}
        if td_header_idx is not None:
            for row in rows[td_header_idx + 1 :]:
                if len(row) < 7:
                    continue
                label = row[0].strip()
                if not label:
                    continue
                if len(row) >= 11:
                    probe_name = row[4].strip()
                    probe_units = row[5].strip()
                    probe_imp = row[6].strip()
                    probe_gain = row[7].strip()
                    sample_rate_idx, sample_idx_idx, voltage_idx = 8, 9, 10
                else:
                    probe_name = default_probe_snapshot.get("probe_name", "")
                    probe_units = default_probe_snapshot.get("probe_units", "")
                    probe_imp_value = default_probe_snapshot.get("probe_impedance_ohms")
                    probe_gain_value = default_probe_snapshot.get("probe_v_per_m_gain")
                    probe_imp = "" if probe_imp_value is None else str(probe_imp_value)
                    probe_gain = "" if probe_gain_value is None else str(probe_gain_value)
                    sample_rate_idx, sample_idx_idx, voltage_idx = 4, 5, 6
                key = (
                    label,
                    row[1].strip(),
                    row[2].strip(),
                    row[3].strip(),
                    probe_name,
                    probe_units,
                    probe_imp,
                    probe_gain,
                )
                try:
                    sample_rate = float(row[sample_rate_idx])
                    sample_idx = int(float(row[sample_idx_idx]))
                    voltage = float(row[voltage_idx])
                except ValueError:
                    continue
                if sample_idx < 0:
                    continue
                bucket = waveform_grouped.setdefault(
                    key, {"sample_rate": sample_rate, "samples": []}
                )
                bucket["samples"].append((sample_idx, voltage))

        traces: List[dict] = []
        for key, vals in grouped.items():
            (
                label,
                window_text,
                rbw_text,
                color_text,
                probe_name_text,
                probe_units_text,
                probe_imp_text,
                probe_gain_text,
            ) = key
            if rbw_text in {"", "N/A"}:
                rbw_val = None
            else:
                try:
                    rbw_val = float(rbw_text)
                except ValueError:
                    rbw_val = None
            volts_arr = np.array([])
            sample_rate_val = None
            if key in waveform_grouped:
                wf = waveform_grouped[key]
                sample_rate_val = float(wf["sample_rate"])
                ordered = sorted(wf["samples"], key=lambda pair: pair[0])
                volts_arr = np.array([v for _, v in ordered], dtype=float)
            if probe_units_text in {"dBuV", "dBuA", "V/m"}:
                parsed_probe_units = probe_units_text
            else:
                parsed_probe_units = default_probe_snapshot.get("probe_units", "dBuV")
            try:
                parsed_probe_imp = float(probe_imp_text) if probe_imp_text.strip() else None
            except ValueError:
                parsed_probe_imp = None
            try:
                parsed_probe_gain = float(probe_gain_text) if probe_gain_text.strip() else None
            except ValueError:
                parsed_probe_gain = None
            if probe_name_text.strip():
                parsed_probe_name = probe_name_text.strip()
            else:
                parsed_probe_name = default_probe_snapshot.get("probe_name", "")
            trace = {
                "label": label,
                "window": window_text or "Imported",
                "effective_rbw_hz": rbw_val,
                "color": color_text or "",
                "volts": volts_arr,
                "sample_rate": sample_rate_val,
                "freqs": np.array(vals["freqs"], dtype=float),
                "original": np.array(vals["orig"], dtype=float),
                "windowed": np.array(vals["win"], dtype=float),
                "probe_name": parsed_probe_name,
                "probe_units": parsed_probe_units,
                "probe_impedance_ohms": parsed_probe_imp,
                "probe_v_per_m_gain": parsed_probe_gain,
            }
            traces.append(trace)
        return metadata, traces

    freqs = []
    orig = []
    win = []
    for row in rows:
        if len(row) < 3:
            continue
        try:
            freq = float(row[0])
            o = float(row[1])
            w = float(row[2])
        except ValueError:
            continue
        freqs.append(freq)
        orig.append(o)
        win.append(w)

    traces = []
    if freqs:
        stem = Path(path).stem
        trace = {
            "label": f"Imported {stem}",
            "window": "Imported",
            "effective_rbw_hz": None,
            "color": "",
            "volts": np.array([]),
            "sample_rate": None,
            "freqs": np.array(freqs, dtype=float),
            "original": np.array(orig, dtype=float),
            "windowed": np.array(win, dtype=float),
            **default_probe_snapshot,
        }
        traces.append(trace)
    return metadata, traces
