import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
SUPPORTED_PROBE_UNITS = {"dBuV", "dBuA", "V/m"}

@dataclass(frozen=True)
class Probe:
    name: str
    measured_units: str
    impedance_ohms: Optional[float] = None
    volts_to_v_per_m_gain: Optional[float] = None
    description: str = ""

    def can_convert(self, from_units: str, to_units: str) -> bool:
        if from_units == to_units:
            return True
        pair = {from_units, to_units}
        if pair == {"dBuV", "dBuA"}:
            return self.impedance_ohms is not None and self.impedance_ohms > 0
        if from_units == "dBuV" and to_units == "V/m":
            return self.volts_to_v_per_m_gain is not None and self.volts_to_v_per_m_gain > 0
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

        if from_units == "dBuV" and to_units == "V/m":
            if self.volts_to_v_per_m_gain is None or self.volts_to_v_per_m_gain <= 0:
                raise ValueError(
                    f"Probe '{self.name}' requires a positive V->V/m gain to convert "
                    f"{from_units} to {to_units}."
                )
            volts = 1e-6 * (10.0 ** (value_db / 20.0))
            return volts * self.volts_to_v_per_m_gain

        raise ValueError(f"Unsupported unit conversion: {from_units} -> {to_units}")


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
        "volts_to_v_per_m_gain": probe.volts_to_v_per_m_gain,
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
    gain_raw = raw.get("volts_to_v_per_m_gain", None)
    if gain_raw in ("", None):
        volts_to_v_per_m_gain = None
    else:
        try:
            volts_to_v_per_m_gain = float(gain_raw)
        except (TypeError, ValueError):
            return None
    if volts_to_v_per_m_gain is not None and volts_to_v_per_m_gain <= 0:
        return None
    if measured_units == "dBuA" and (impedance_ohms is None or impedance_ohms <= 0):
        return None
    if measured_units == "V/m" and (volts_to_v_per_m_gain is None or volts_to_v_per_m_gain <= 0):
        return None
    description = str(raw.get("description", "")).strip()
    return Probe(
        name=name,
        measured_units=measured_units,
        impedance_ohms=impedance_ohms,
        volts_to_v_per_m_gain=volts_to_v_per_m_gain,
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
    volts_to_v_per_m_gain: Optional[float] = None,
    description: str = "",
) -> Probe:
    probe_name = (name or "").strip()
    units = (measured_units or "").strip()
    if not probe_name:
        raise ValueError("Probe name is required.")
    if units not in SUPPORTED_PROBE_UNITS:
        raise ValueError("Units must be one of: dBuV, dBuA, V/m.")
    if units == "dBuA":
        if impedance_ohms is None:
            raise ValueError("dBuA probes require impedance (ohms).")
        if impedance_ohms <= 0:
            raise ValueError("Probe impedance must be > 0 ohms.")
    if units == "V/m":
        if volts_to_v_per_m_gain is None:
            raise ValueError("V/m probes require V->V/m gain.")
        if volts_to_v_per_m_gain <= 0:
            raise ValueError("V->V/m gain must be > 0.")
    elif impedance_ohms is not None and impedance_ohms <= 0:
        raise ValueError("Probe impedance must be > 0 ohms.")
    if impedance_ohms is not None and impedance_ohms <= 0:
        raise ValueError("Probe impedance must be > 0 ohms.")
    if volts_to_v_per_m_gain is not None and volts_to_v_per_m_gain <= 0:
        raise ValueError("V->V/m gain must be > 0.")

    probe = Probe(
        name=probe_name,
        measured_units=units,
        impedance_ohms=impedance_ohms,
        volts_to_v_per_m_gain=volts_to_v_per_m_gain,
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
