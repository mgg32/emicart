import json
import logging
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
SUPPORTED_PROBE_UNITS = {"dBuV", "dBuA", "dBuV/m", "V/m"}  # V/m is accepted only for legacy migration.

@dataclass(frozen=True)
class Probe:
    name: str
    measured_units: str
    impedance_ohms: Optional[float] = None
    # (frequency in Hz, additive correction in dB).  For an antenna this is
    # normally the antenna factor in dB/m.
    frequency_correction_factors: Tuple[Tuple[float, float], ...] = ()
    min_frequency_hz: Optional[float] = None
    max_frequency_hz: Optional[float] = None

    def supports_frequency(self, frequency_hz: float) -> bool:
        return (
            (self.min_frequency_hz is None or frequency_hz >= self.min_frequency_hz)
            and (self.max_frequency_hz is None or frequency_hz <= self.max_frequency_hz)
        )
    description: str = ""

    def can_convert(self, from_units: str, to_units: str) -> bool:
        if from_units == to_units:
            return True
        pair = {from_units, to_units}
        if pair == {"dBuV", "dBuA"}:
            return self.impedance_ohms is not None and self.impedance_ohms > 0
        if from_units == "dBuV" and to_units == "dBuV/m":
            return bool(self.frequency_correction_factors)
        return False

    def convert_db_level(self, value_db: float, from_units: str, to_units: str) -> float:
        if from_units == to_units:
            return value_db

        if {from_units, to_units} == {"dBuV", "dBuA"}:
            if self.impedance_ohms is None or self.impedance_ohms <= 0:
                raise ValueError(
                    f"Probe '{self.name}' requires a positive impedance to convert "
                    f"{from_units} to {to_units}."
                )
            offset = 20.0 * math.log10(self.impedance_ohms)
            if from_units == "dBuV" and to_units == "dBuA":
                return value_db - offset
            return value_db + offset

        raise ValueError(f"Unsupported unit conversion: {from_units} -> {to_units}")

    def correction_db_at(self, frequency_hz: float) -> float:
        """Return the additive dB correction, linearly interpolated by frequency."""
        if self.frequency_correction_factors:
            frequencies, corrections = zip(*self.frequency_correction_factors)
            return float(np.interp(float(frequency_hz), frequencies, corrections))
        return 0.0


def _get_probe_store_path() -> Path:
    base = Path.home() / "Documents" / "EmiCart"
    base.mkdir(parents=True, exist_ok=True)
    return base / "probes.json"


def _get_default_probe_template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "default_probes.json"


def _build_default_probe_registry() -> Dict[str, Probe]:
    template_path = _get_default_probe_template_path()
    try:
        payload = json.loads(template_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load probe defaults from %s: %s", template_path, e)
        payload = []
    registry: Dict[str, Probe] = {}
    invalid_count = 0
    if isinstance(payload, list):
        for item in payload:
            probe = _dict_to_probe(item)
            if probe is not None:
                registry[probe.name] = probe
            else:
                invalid_count += 1
    if invalid_count:
        logger.warning("Dropped %d invalid probe entries from default template.", invalid_count)
    if registry:
        return registry
    logger.warning("Probe default template had no valid entries: %s", template_path)
    return {
        "Direct Voltage (No Probe)": Probe(
            name="Direct Voltage (No Probe)",
            measured_units="dBuV",
            impedance_ohms=None,
            description="No conversion. Scope voltage is plotted as dBuV.",
        )
    }


def _probe_to_dict(probe: Probe) -> dict:
    return {
        "name": probe.name,
        "measured_units": probe.measured_units,
        "impedance_ohms": probe.impedance_ohms,
        "frequency_correction_factors": [list(point) for point in probe.frequency_correction_factors],
        "min_frequency_hz": probe.min_frequency_hz,
        "max_frequency_hz": probe.max_frequency_hz,
        "description": probe.description,
    }


def _dict_to_probe(raw: dict) -> Optional[Probe]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    measured_units = str(raw.get("measured_units", "")).strip()
    if not name or measured_units not in SUPPORTED_PROBE_UNITS:
        return None
    impedance = raw.get("impedance_ohms", None)
    if impedance in ("", None):
        impedance_ohms = None
    else:
        try:
            impedance_ohms = float(impedance)
        except (TypeError, ValueError):
            return None
    if impedance_ohms is not None and impedance_ohms <= 0:
        return None
    factors_raw = raw.get("frequency_correction_factors", [])
    if factors_raw is None:
        factors_raw = []
    if not isinstance(factors_raw, list):
        return None
    factors = []
    for point in factors_raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            frequency, factor = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if frequency <= 0:
            return None
        factors.append((frequency, factor))
    factors.sort()
    if any(a[0] == b[0] for a, b in zip(factors, factors[1:])):
        return None
    def optional_frequency(key):
        value = raw.get(key)
        if value in (None, ""):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    min_frequency_hz = optional_frequency("min_frequency_hz")
    max_frequency_hz = optional_frequency("max_frequency_hz")
    if min_frequency_hz is not None and max_frequency_hz is not None and min_frequency_hz >= max_frequency_hz:
        return None
    if measured_units == "dBuA" and (impedance_ohms is None or impedance_ohms <= 0):
        return None
    # Upgrade the first implementation's linear V/m-per-V field to an
    # equivalent one-point antenna factor in dBµV/m.
    if measured_units == "V/m":
        gain = raw.get("volts_to_v_per_m_gain")
        try:
            gain = float(gain)
        except (TypeError, ValueError):
            return None
        if gain <= 0:
            return None
        measured_units = "dBuV/m"
        factors = [(1.0, 20.0 * math.log10(gain))]
    description = str(raw.get("description", "")).strip()
    return Probe(
        name=name,
        measured_units=measured_units,
        impedance_ohms=impedance_ohms,
        frequency_correction_factors=tuple(factors),
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        description=description,
    )


def _save_probe_registry(registry: Dict[str, Probe]) -> None:
    path = _get_probe_store_path()
    payload = [_probe_to_dict(registry[name]) for name in sorted(registry.keys())]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_probe_registry() -> Dict[str, Probe]:
    path = _get_probe_store_path()
    if not path.exists():
        seeded = _build_default_probe_registry()
        _save_probe_registry(seeded)
        return seeded
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse probe registry %s: %s", path, e)
        payload = []
    if not isinstance(payload, list):
        payload = []

    registry: Dict[str, Probe] = {}
    invalid_count = 0
    for item in payload:
        probe = _dict_to_probe(item)
        if probe is None:
            invalid_count += 1
            continue
        registry[probe.name] = probe
    if invalid_count:
        logger.warning("Dropped %d invalid probe entries from %s.", invalid_count, path)

    if not registry:
        registry = _build_default_probe_registry()
        _save_probe_registry(registry)
        return registry

    # Add newly introduced default probes without overwriting user-edited entries.
    defaults = _build_default_probe_registry()
    added = False
    for name, probe in defaults.items():
        if name not in registry:
            registry[name] = probe
            added = True
    if added:
        _save_probe_registry(registry)
    return registry


_probe_registry = _load_probe_registry()


def reload_probes() -> None:
    global _probe_registry
    _probe_registry = _load_probe_registry()


def get_probe_names() -> List[str]:
    return sorted(_probe_registry.keys())


def get_probe_by_name(name: str) -> Optional[Probe]:
    return _probe_registry.get(name)


def get_default_probe() -> Probe:
    direct = _probe_registry.get("Direct Voltage (No Probe)")
    if direct is not None:
        return direct
    first = next(iter(_probe_registry.values()), None)
    if first is not None:
        return first
    fallback = Probe(name="Direct Voltage (No Probe)", measured_units="dBuV")
    _probe_registry[fallback.name] = fallback
    _save_probe_registry(_probe_registry)
    return fallback


def upsert_probe(
    name: str,
    measured_units: str,
    impedance_ohms: Optional[float] = None,
    frequency_correction_factors: Sequence[Tuple[float, float]] = (),
    min_frequency_hz: Optional[float] = None,
    max_frequency_hz: Optional[float] = None,
    description: str = "",
) -> Probe:
    probe_name = (name or "").strip()
    units = (measured_units or "").strip()
    if not probe_name:
        raise ValueError("Probe name is required.")
    if units not in SUPPORTED_PROBE_UNITS:
        raise ValueError("Units must be one of: dBuV, dBuA, dBuV/m.")
    if units == "dBuA":
        if impedance_ohms is None:
            raise ValueError("dBuA probes require impedance (ohms).")
        if impedance_ohms <= 0:
            raise ValueError("Probe impedance must be > 0 ohms.")
    try:
        factors = tuple(sorted((float(frequency), float(factor)) for frequency, factor in frequency_correction_factors))
    except (TypeError, ValueError):
        raise ValueError("Each correction factor must contain numeric frequency and factor values.")
    if any(frequency <= 0 for frequency, _ in factors):
        raise ValueError("Correction frequencies must be > 0.")
    if any(a[0] == b[0] for a, b in zip(factors, factors[1:])):
        raise ValueError("Correction frequencies must be unique.")
    for label, value in (("Minimum frequency", min_frequency_hz), ("Maximum frequency", max_frequency_hz)):
        if value is not None and value <= 0:
            raise ValueError(f"{label} must be > 0 Hz.")
    if min_frequency_hz is not None and max_frequency_hz is not None and min_frequency_hz >= max_frequency_hz:
        raise ValueError("Minimum frequency must be below maximum frequency.")
    elif impedance_ohms is not None and impedance_ohms <= 0:
        raise ValueError("Probe impedance must be > 0 ohms.")
    if impedance_ohms is not None and impedance_ohms <= 0:
        raise ValueError("Probe impedance must be > 0 ohms.")

    probe = Probe(
        name=probe_name,
        measured_units=units,
        impedance_ohms=impedance_ohms,
        frequency_correction_factors=factors,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        description=(description or "").strip(),
    )
    _probe_registry[probe_name] = probe
    _save_probe_registry(_probe_registry)
    return probe


def delete_probe(name: str) -> bool:
    probe_name = (name or "").strip()
    if not probe_name:
        return False
    if probe_name not in _probe_registry:
        return False
    del _probe_registry[probe_name]
    _save_probe_registry(_probe_registry)
    return True
