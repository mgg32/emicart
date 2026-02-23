# EmiCart

EmiCart is a Python toolkit for scope-based EMI spectrum capture and limit-curve comparison.

## Run

Run the GUI with:

- `python -m emicart`

Scope selection defaults to automatic VISA discovery. If multiple instruments are connected, set
`EMICART_SCOPE_RESOURCE` to force a specific Tektronix resource string.

## Data Formats

Export supports:

- `PNG` for plot image only.
- `CSV` for human-readable data + metadata.
- `NPZ` for compact NumPy binary data + metadata.
- `MAT` for MATLAB-compatible binary data + metadata (`scipy` required).

Import supports:

- `CSV`, `NPZ`, and `MAT` (all restore metadata when available).
- Metadata keys stored in export files: `Standard`, `LimitCurve`, `Probe`, `WindowSelection`.

Per-trace saved data includes:

- Label, window name, effective RBW, color
- Probe snapshot (name, units, impedance, V->V/m gain)
- Frequency vector, original FFT data, windowed FFT data
- Optional time-domain waveform + sample rate (when available)

## Maintenance Helpers

- Smoke test roundtrip export/import:
  - `python scripts/smoke_roundtrip.py`
- Reset user registries (limits/probes):
  - `python scripts/reset_user_data.py`
- Reset user registries and outputs:
  - `python scripts/reset_user_data.py --include-outputs --yes`

## Raspberry Pi Deploy

Recommended on Raspberry Pi OS (Bookworm/Bullseye):

1. Copy/clone this repo onto the Pi.
2. Run:
   - `chmod +x scripts/install_pi.sh`
   - `./scripts/install_pi.sh`
3. Start EmiCart:
   - `source .venv/bin/activate`
   - `python -m emicart`

The install script:

- Installs OS packages (`python3-venv`, `python3-pip`, `python3-tk`, `libatlas-base-dev`)
- Creates `.venv`
- Installs runtime Python deps (`numpy`, `matplotlib`, `scipy`, `pyvisa`, `pyvisa-py`, `psutil`, `zeroconf`)
- Installs this package (`pip install .`)
- Creates launchers at `~/Desktop/EmiCart.desktop` and `~/.local/share/applications/EmiCart.desktop`

Optional package build on Pi (for wheel deployment):

- `source .venv/bin/activate`
- `pip install build`
- `python -m build`
- Install wheel: `pip install dist/<wheel-file>.whl`

To install with MATLAB export support:

- `pip install .[mat]`

## Windows Deploy

Recommended on Windows 10/11:

1. Open PowerShell in the repo root.
2. Run:
   - `powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1`
3. Start EmiCart:
   - `.\.venv\Scripts\Activate.ps1`
   - `python -m emicart`

If you have multiple Python installs, you can pick one:

- `powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1 -PythonExe "py -3.11"`

The install script:

- Creates `.venv`
- Upgrades `pip`
- Installs runtime Python deps (`numpy`, `matplotlib`, `scipy`, `pyvisa`, `pyvisa-py`, `psutil`, `zeroconf`)
- Installs this package (`pip install .`)
- Creates a Desktop shortcut (`EmiCart.lnk`)

Optional package build on Windows (for wheel deployment):

- `.\.venv\Scripts\Activate.ps1`
- `pip install build`
- `python -m build`
- Install wheel: `pip install .\dist\<wheel-file>.whl`

To install with MATLAB export support:

- `pip install .[mat]`

The codebase is package-first. Runtime entrypoint and application modules live under `emicart/`.

## Project Layout

- `emicart/ui/curve_viewer.py`
  - Tkinter + Matplotlib UI for plotting selected standard/curve limits.
- `emicart/ui/capture_tool.py`
  - End-to-end capture workflow (GUI input, scope acquire, FFT plot, CSV save).
- `emicart/ui/files.py`
  - Shared save-path and filename increment helpers.
- `emicart/instruments/tektronix.py`
  - Shared PyVISA connection and waveform acquisition helpers.
- `emicart/analysis/fft.py`
  - FFT helper functions and magnitude-to-dB conversion helpers.
- `emicart/analysis/units.py`
  - Unit-conversion helpers (for example, `dBuV <-> dBuA` through probe impedance).
- `emicart/limits/registry.py`
  - Standard-aware curve registry and lookup API.
  - Standards/curves are persisted to `~/Documents/EmiCart/limits.json`.
  - First-run defaults are seeded from `emicart/data/default_limits.json`.
- `emicart/probes/registry.py`
  - Probe registry (probe names, measured units, and probe impedance for conversions).
  - Probes are persisted to `~/Documents/EmiCart/probes.json`.
  - First-run defaults are seeded from `emicart/data/default_probes.json`.
- `emicart/data/default_limits.json`
  - Initial seeded standard/curve definitions.
- `emicart/data/default_probes.json`
  - Initial seeded probe definitions.

## Entry Point

- `emicart/__main__.py` launches `emicart.ui.curve_viewer.main`

## Where to Edit Going Forward

- Add or modify seeded limits: `emicart/data/default_limits.json`
- Add or modify seeded probes: `emicart/data/default_probes.json`
- Change curve registry behavior: `emicart/limits/registry.py`
- Change Tek scope behavior: `emicart/instruments/tektronix.py`
- Change FFT/math behavior: `emicart/analysis/fft.py`
- Change UIs: `emicart/ui/*.py`
  - In the main GUI Source section, use `Manage Probes` to add/edit/delete probes.
  - Use `Manage Standards` to add/edit/delete standards and limit curves.

## Dev

- Run tests: `python -m unittest discover -s tests -v`
- Optional dev extras: `pip install .[dev]`
