import numpy as np

from emicart.probes.registry import Probe


SCOPE_BASE_UNITS = "dBuV"


def convert_trace_db(values_db, from_units: str, to_units: str, probe: Probe):
    values = np.array(values_db, dtype=float, copy=True)
    if from_units == to_units:
        return values
    if not probe.can_convert(from_units, to_units):
        raise ValueError(
            f"Probe '{probe.name}' cannot convert {from_units} to {to_units}. "
            "Set probe impedance or choose compatible units."
        )
    converter = np.vectorize(lambda v: probe.convert_db_level(float(v), from_units, to_units))
    return converter(values).astype(float)
