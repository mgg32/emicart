import numpy as np
from scipy.signal import get_window
from typing import Any, cast


def _resolve_window_name(window_name):
    if window_name is None:
        return "boxcar"
    key = str(window_name).strip().lower()
    aliases = {
        "rectangular": "boxcar",
        "square": "boxcar",
        "boxcar": "boxcar",
        "none": "boxcar",
        "hann": "hann",
        "hanning": "hann",
        "hamming": "hamming",
        "blackman": "blackman",
        "bartlett": "bartlett",
        "flat top": "flattop",
        "flattop": "flattop",
    }
    return aliases.get(key, key)


def get_window_array(window_name, n):
    if n is None or int(n) <= 0:
        raise ValueError("Window length n must be a positive integer.")
    return get_window(_resolve_window_name(window_name), n)


def get_effective_rbw_hz(sample_rate, n, window_name):
    if sample_rate is None or float(sample_rate) <= 0:
        raise ValueError("sample_rate must be > 0.")
    window = get_window_array(window_name, n)
    bin_width_hz = sample_rate / n
    enbw_bins = n * np.sum(window**2) / (np.sum(window) ** 2)
    return bin_width_hz * enbw_bins


def apply_rbw_correction_db(mags_db, target_rbw_hz, effective_rbw_hz):
    corrected = np.array(mags_db, dtype=float, copy=True)
    if target_rbw_hz is None or effective_rbw_hz <= 0:
        return corrected

    if np.isscalar(target_rbw_hz):
        try:
            scalar_target_rbw_hz = float(cast(Any, target_rbw_hz))
        except (TypeError, ValueError):
            return corrected
        if scalar_target_rbw_hz > 0:
            corrected += 10.0 * np.log10(scalar_target_rbw_hz / effective_rbw_hz)
        return corrected

    target = np.array([np.nan if v is None else float(v) for v in target_rbw_hz], dtype=float)
    valid = np.isfinite(target) & (target > 0)
    corrected[valid] += 10.0 * np.log10(target[valid] / effective_rbw_hz)
    return corrected


def t2f(data, dt, window_name="hann"):
    if dt is None or float(dt) <= 0:
        raise ValueError("dt must be > 0.")
    n = len(data)
    if n <= 0:
        raise ValueError("data must contain at least one sample.")
    window = get_window_array(window_name, n)
    data_windowed = data * window
    fft_vals = np.fft.rfft(data_windowed)
    fft_freqs = np.fft.rfftfreq(n, dt)
    return fft_freqs, np.abs(fft_vals)


def compute_single_sided_fft_db(
    volts,
    sample_rate,
    window_name="rectangular",
    amplitude_offset_db=120,
):
    n = len(volts)
    if n <= 0:
        raise ValueError("volts must contain at least one sample.")
    if sample_rate is None or float(sample_rate) <= 0:
        raise ValueError("sample_rate must be > 0.")
    window = get_window_array(window_name, n)
    signal = volts * window

    fft_data = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n, d=1 / sample_rate)

    mask = fft_freqs > 0
    freqs = fft_freqs[mask]
    mags = np.abs(fft_data[mask]) * 2 / np.sum(window)
    mags_db = 20 * np.log10(mags + 1e-12) + amplitude_offset_db
    return freqs, mags_db


def apply_frequency_domain_window_by_rbw(
    mags_db,
    freqs_hz,
    target_rbw_hz,
    window_name="hann",
):
    """
    Apply frequency-domain windowing in RBW-sized chunks, honoring RBW discontinuities.

    - Each contiguous run of equal target RBW is processed independently.
    - Window frame length for a segment is derived from that segment RBW and FFT bin width.
    - Frames do not cross segment boundaries.
    """
    mags = np.asarray(mags_db, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float)
    if mags.ndim != 1 or freqs.ndim != 1 or len(mags) != len(freqs):
        raise ValueError("mags_db and freqs_hz must be 1D arrays of equal length.")
    n = len(mags)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array(mags, copy=True)

    # Convert to a target RBW vector.
    if np.isscalar(target_rbw_hz):
        try:
            rbw_vals = np.full(n, float(cast(Any, target_rbw_hz)), dtype=float)
        except (TypeError, ValueError):
            return np.array(mags, copy=True)
    else:
        rbw_vals = np.array([np.nan if v is None else float(v) for v in target_rbw_hz], dtype=float)
        if rbw_vals.shape[0] != n:
            raise ValueError("target_rbw_hz length must match mags_db/freqs_hz length.")

    # Positive frequency bins from FFT are uniformly spaced.
    bin_width_hz = float(np.median(np.diff(freqs)))
    if not np.isfinite(bin_width_hz) or bin_width_hz <= 0:
        return np.array(mags, copy=True)

    # Process in a linear power-like domain, then map back to dB.
    power = 10.0 ** (mags / 10.0)
    out_power = np.array(power, copy=True)

    valid = np.isfinite(rbw_vals) & (rbw_vals > 0)
    if not np.any(valid):
        return np.array(mags, copy=True)

    idx = 0
    while idx < n:
        if not valid[idx]:
            idx += 1
            continue

        segment_rbw = rbw_vals[idx]
        j = idx + 1
        while j < n and valid[j] and np.isclose(rbw_vals[j], segment_rbw, rtol=0.0, atol=1e-12):
            j += 1

        seg_len = j - idx
        win_bins = max(1, int(round(float(segment_rbw) / bin_width_hz)))
        win_bins = min(win_bins, seg_len)
        if win_bins % 2 == 0 and win_bins > 1:
            win_bins -= 1

        if win_bins > 1:
            w = get_window_array(window_name, win_bins)
            w = np.asarray(w, dtype=float)
            w_sum = float(np.sum(w))
            if w_sum > 0:
                w = w / w_sum
                seg = power[idx:j]
                smoothed = np.convolve(seg, w, mode="same")
                out_power[idx:j] = smoothed

        idx = j

    out_db = 10.0 * np.log10(np.maximum(out_power, 1e-30))
    return out_db
