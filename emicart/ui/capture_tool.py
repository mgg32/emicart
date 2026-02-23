import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np

from emicart.limits import registry as curve
from emicart.analysis.fft import compute_single_sided_fft_db
from emicart.instruments.tektronix import (
    configure_timebase,
    connect_to_scope,
    download_waveform,
    get_record_length,
)

DEFAULT_LIMIT_STANDARD = None
DEFAULT_LIMIT_CURVE = None


def get_freqs_from_gui():
    def browse_file():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Select file to save waveform data",
        )
        if file_path:
            file_path_var.set(file_path)

    def submit():
        try:
            min_freq = float(min_freq_entry.get())
            max_freq = float(max_freq_entry.get())
            file_path = file_path_var.get()

            if min_freq <= 0 or max_freq <= 0 or max_freq <= min_freq:
                raise ValueError("Invalid frequency values")
            if not file_path:
                raise ValueError("No file path selected")

            root.quit()
            root.destroy()
            result.append((min_freq, max_freq, file_path))
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    result = []
    root = tk.Tk()
    root.title("Tektronix FFT Capture Tool")

    instructions = (
        "Enter the minimum and maximum frequency (in Hz, supports scientific notation).\n"
        "The tool will configure the oscilloscope to capture data accordingly,\n"
        "perform FFT analysis, and save the waveform to a CSV file."
    )
    tk.Label(root, text=instructions, justify="left", wraplength=400).grid(
        row=0, column=0, columnspan=2, padx=10, pady=(10, 0)
    )

    tk.Label(root, text="Min Frequency (Hz):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    min_freq_entry = tk.Entry(root)
    min_freq_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Max Frequency (Hz):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    max_freq_entry = tk.Entry(root)
    max_freq_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="Save CSV as:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    file_path_var = tk.StringVar()
    file_entry = tk.Entry(root, textvariable=file_path_var, width=30)
    file_entry.grid(row=3, column=1, padx=(10, 0), pady=5, sticky="w")
    browse_btn = tk.Button(root, text="Browse...", command=browse_file)
    browse_btn.grid(row=3, column=1, padx=(0, 10), pady=5, sticky="e")

    submit_btn = tk.Button(root, text="Submit", command=submit)
    submit_btn.grid(row=4, column=0, columnspan=2, pady=15)

    root.mainloop()

    if result:
        return result[0]
    raise SystemExit("No valid input provided.")


def plot_fft(
    volts,
    sample_rate,
    min_freq=None,
    max_freq=None,
    limit_standard=DEFAULT_LIMIT_STANDARD,
    limit_curve_name=DEFAULT_LIMIT_CURVE,
):
    if limit_standard is None:
        standards = curve.get_standards()
        limit_standard = standards[0] if standards else None
    if limit_curve_name is None and limit_standard is not None:
        available = curve.get_curves_for_standard(limit_standard)
        limit_curve_name = available[0].name if available else None

    freqs, mags_db = compute_single_sided_fft_db(volts, sample_rate)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, mags_db, label="FFT Magnitude (dB)")

    selected_curve = curve.get_curve_by_name(limit_curve_name, standard=limit_standard)
    if selected_curve is None:
        available = curve.get_curves_for_standard(limit_standard)
        if available:
            selected_curve = available[0]

    if selected_curve is not None:
        limit_y = selected_curve.get_curve(freqs)
        limit_y = np.array([np.nan if v is None else v for v in limit_y], dtype=float)
        plt.plot(freqs, limit_y, label=selected_curve.name, color="green")

    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    y_units = selected_curve.units if selected_curve is not None else "dBuV"
    plt.ylabel(f"Amplitude ({y_units})")
    plt.title("FFT Spectrum (Log Frequency Scale + Hann Window)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()

    if min_freq:
        plt.axvline(min_freq, color="orange", linestyle="--", label=f"Min freq: {min_freq:.2e} Hz")
    if max_freq:
        plt.axvline(max_freq, color="red", linestyle="--", label=f"Max freq: {max_freq:.2e} Hz")

    if min_freq and max_freq:
        plt.xlim(min_freq / 2, max_freq * 2)
    else:
        plt.xlim(freqs[0], freqs[-1])

    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    try:
        min_freq, max_freq, save_path = get_freqs_from_gui()
        oversample_factor = 10
        limit_standard = DEFAULT_LIMIT_STANDARD
        limit_curve_name = DEFAULT_LIMIT_CURVE

        print(f"Limit standard: {limit_standard}")
        print(f"Limit curve: {limit_curve_name}")

        min_sample_rate = max_freq * oversample_factor
        min_time_span = 20 / min_freq
        record_length = int(min_sample_rate * min_time_span)
        time_per_div = min_time_span / 10

        print()
        print("Requested parameters")
        print(f"Frequency range: {min_freq:.2e} Hz -> {max_freq:.2e} Hz")
        print(f"Sample rate: {min_sample_rate:.2e} samples/sec")
        print(f"Record length: {record_length} samples")
        print(f"Time span: {min_time_span:.6e} s ({time_per_div:.2e} s/div)")
        print(f"Resolution: {min_sample_rate / record_length:.2f} Hz")
        print()

        scope = connect_to_scope()
        configure_timebase(scope, time_per_div=time_per_div, record_length=record_length)

        xincr = float(scope.query("WFMPRE:XINCR?"))
        sample_rate = 1 / xincr
        actual_record_length = get_record_length(scope)
        actual_time_span = xincr * actual_record_length
        actual_time_per_div = actual_time_span / 10
        f_min_fft = 1 / actual_time_span
        f_max_fft = sample_rate / 2

        print()
        print("Actual parameters")
        print(f"Frequency range: {f_min_fft:.2e} Hz -> {f_max_fft:.2e} Hz")
        print(f"Sample rate: {sample_rate:.2e} samples/sec")
        print(f"Record length: {actual_record_length} samples")
        print(f"Time span: {actual_time_span:.6e} s ({actual_time_per_div:.2e} s/div)")
        print(f"Resolution: {sample_rate / actual_record_length:.2f} Hz")
        print()

        if f_min_fft > min_freq:
            print(
                f"WARNING: FFT cannot resolve min_freq {min_freq:.2f} Hz; "
                f"FFT starts at {f_min_fft:.2f} Hz."
            )

        volts = download_waveform(scope, channel="CH1", num_points=actual_record_length)
        print(f"Downloaded {len(volts)} points from CH1.")

        np.savetxt(save_path, volts, delimiter=",", header="Voltage (V)", comments="")
        print(f"Saved waveform to {save_path}")
        print()

        plot_fft(
            volts,
            sample_rate,
            min_freq,
            max_freq,
            limit_standard=limit_standard,
            limit_curve_name=limit_curve_name,
        )
    except Exception as e:
        print(f"ERROR: {e}")
