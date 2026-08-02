"""Standalone diagnostic script (NOT part of the main app) that runs the real
`apply_autoscale` max-points logic against your scope with step-by-step
instrumentation, so we can see exactly what the scope is reporting at each
stage instead of guessing.

Run this with the scope connected and share the full printed output.

Usage (from the repo root, with the venv activated):
    python scripts/diagnose_autoscale.py
"""
import sys

sys.path.insert(0, ".")

from emicart.instruments.tektronix import (  # noqa: E402
    apply_autoscale,
    connect_to_scope,
    get_record_length,
    get_time_per_div,
)

DEFAULT_MAX_POINTS = 1_000_000


def on_step(event, info):
    print(f"  [{event}] {info}")


def main():
    print("Connecting to scope...")
    scope = connect_to_scope()
    print("Connected.")

    print(f"\nBefore autoscale: time/div={get_time_per_div(scope):g}, record_length={get_record_length(scope)}")

    print("\nRunning apply_autoscale (max_points scenario, no limit curve)...")
    info = apply_autoscale(
        scope,
        record_length=DEFAULT_MAX_POINTS,
        required_max_frequency_hz=None,
        on_step=on_step,
    )

    print("\n--- Result ---")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(
        f"\nAfter autoscale: time/div={get_time_per_div(scope):g}, record_length={get_record_length(scope)}"
    )
    print("\nDone. Please copy/paste all of the output above.")


if __name__ == "__main__":
    main()
