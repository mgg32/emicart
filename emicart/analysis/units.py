import numpy as np

from emicart.probes.registry import Probe


SCOPE_BASE_UNITS = "dBuV"


def convert_trace_db(values_db, from_units: str, to_units: str, probe: Probe, frequencies_hz=None):
    values = np.array(values_db, dtype=float, copy=True)
    if not probe.can_convert(from_units, to_units):
        raise ValueError(
            f"Probe '{probe.name}' cannot convert {from_units} to {to_units}. "
            "Set probe impedance or choose compatible units."
        )
    if from_units == "dBuV" and to_units == "dBuV/m":
        if frequencies_hz is None:
            raise ValueError("Frequencies are required for dBuV/m correction.")
        frequencies = np.asarray(frequencies_hz, dtype=float)
        if frequencies.shape != values.shape:
            raise ValueError("Frequency and amplitude arrays must have the same shape.")
        corrections = np.array([probe.correction_db_at(frequency) for frequency in frequencies], dtype=float)
        return values + corrections
    if from_units == to_units:
        if frequencies_hz is None or not probe.frequency_correction_factors:
            return values
        frequencies = np.asarray(frequencies_hz, dtype=float)
        if frequencies.shape != values.shape:
            raise ValueError("Frequency and amplitude arrays must have the same shape.")
        return values + np.array([probe.correction_db_at(frequency) for frequency in frequencies], dtype=float)
    converter = np.vectorize(lambda v: probe.convert_db_level(float(v), from_units, to_units))
    converted = converter(values).astype(float)
    if frequencies_hz is None or not probe.frequency_correction_factors:
        return converted
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.shape != values.shape:
        raise ValueError("Frequency and amplitude arrays must have the same shape.")
    return converted + np.array([probe.correction_db_at(frequency) for frequency in frequencies], dtype=float)
