"""Standalone diagnostic script (NOT part of the main app) to discover which
SCPI commands actually work for horizontal/sample-rate control on your real
scope, and how they respond as time/div changes.

This exists because the app's autoscale logic has repeatedly guessed wrong
about which commands (ACQuire:MAXSamplerate?, ACQuire:SAMPLERATE?,
HORIZONTAL:RECORDLENGTH?, etc.) are supported and how fast they respond on a
Tektronix MSO24. Run this once, with the scope connected, and share the
printed output -- it will tell us exactly what to use instead of guessing.

Usage (from the repo root, with the venv activated):
    python scripts/diagnose_timebase.py
"""
import sys
import time

sys.path.insert(0, ".")

from emicart.instruments.tektronix import connect_to_scope  # noqa: E402

# Candidate queries to probe, in (label, command) pairs. We try many because
# the exact supported syntax varies across Tektronix scope families.
CANDIDATE_QUERIES = [
    ("IDN", "*IDN?"),
    ("ACQuire:MAXSamplerate?", "ACQuire:MAXSamplerate?"),
    ("ACQ:MAXS?", "ACQ:MAXS?"),
    ("ACQuire:SAMPLERATE?", "ACQuire:SAMPLERATE?"),
    ("ACQ:SAMPLERATE?", "ACQ:SAMPLERATE?"),
    ("HORizontal:MODE:SAMPLERATE?", "HORizontal:MODE:SAMPLERATE?"),
    ("HORizontal:SAMPLERATE?", "HORizontal:SAMPLERATE?"),
    ("HORizontal:MODE?", "HORizontal:MODE?"),
    ("HORizontal:MODE:RECOrdlength?", "HORizontal:MODE:RECOrdlength?"),
    ("HORIZONTAL:RECORDLENGTH?", "HORIZONTAL:RECORDLENGTH?"),
    ("HOR:RECO?", "HOR:RECO?"),
    ("HORizontal:SCAle?", "HORizontal:SCAle?"),
    ("HOR:MAIN:SCALE?", "HOR:MAIN:SCALE?"),
    ("WFMOutpre:XINCR?", "WFMOutpre:XINCR?"),
    ("WFMO:XINCR?", "WFMO:XINCR?"),
    ("WFMPRE:XINCR?", "WFMPRE:XINCR?"),
]

# Time/div candidates (seconds) to sweep through while re-checking the queries
# above, to see which values actually change and how.
TIME_PER_DIV_SWEEP = [1e-6, 1e-5, 4e-5, 1e-4, 1e-3]
RECORD_LENGTH = 250_000


def probe_once(scope, label_prefix=""):
    print(f"--- {label_prefix} ---")
    for label, command in CANDIDATE_QUERIES:
        start = time.time()
        try:
            value = scope.query(command).strip()
            elapsed = time.time() - start
            print(f"  {label:35s} -> {value!r}  ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  {label:35s} -> FAILED: {e}  ({elapsed:.2f}s)")


def try_write(scope, command):
    try:
        scope.write(command)
        print(f"  WRITE {command!r} -> ok")
    except Exception as e:
        print(f"  WRITE {command!r} -> FAILED: {e}")


def main():
    print("Connecting to scope...")
    scope = connect_to_scope()
    print("Connected.\n")

    probe_once(scope, "Initial state")

    print("\nDisabling CH2/CH3/CH4, enabling CH1 only...")
    try_write(scope, "SELECT:CH1 ON")
    try_write(scope, "SELECT:CH2 OFF")
    try_write(scope, "SELECT:CH3 OFF")
    try_write(scope, "SELECT:CH4 OFF")
    time.sleep(1)

    probe_once(scope, "After disabling unused channels")

    for time_per_div in TIME_PER_DIV_SWEEP:
        print(f"\nSetting time/div={time_per_div}, record_length={RECORD_LENGTH}...")
        try_write(scope, f"HORIZONTAL:SCALE {time_per_div}")
        try_write(scope, f"HORIZONTAL:RECORDLENGTH {RECORD_LENGTH}")
        time.sleep(1)
        probe_once(scope, f"time/div={time_per_div}")

    print("\nDone. Please copy/paste all of the output above.")


if __name__ == "__main__":
    main()
