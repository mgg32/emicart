import importlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


class RegistryAndUnitsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old_home = os.environ.get("HOME")
        self.old_userprofile = os.environ.get("USERPROFILE")
        os.environ["HOME"] = self.tmp.name
        os.environ["USERPROFILE"] = self.tmp.name

    def tearDown(self):
        if self.old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self.old_home
        if self.old_userprofile is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = self.old_userprofile

    def test_probe_registry_seed_upsert_delete(self):
        probes = importlib.import_module("emicart.probes.registry")
        probes = importlib.reload(probes)

        names = probes.get_probe_names()
        self.assertTrue(len(names) >= 1)
        store = Path(self.tmp.name) / "Documents" / "EmiCart" / "probes.json"
        self.assertTrue(store.exists())

        created = probes.upsert_probe(
            name="Test Probe",
            measured_units="dBuA",
            impedance_ohms=10.0,
            description="test",
        )
        self.assertEqual(created.name, "Test Probe")
        self.assertIn("Test Probe", probes.get_probe_names())

        self.assertTrue(probes.delete_probe("Test Probe"))
        self.assertNotIn("Test Probe", probes.get_probe_names())

    def test_limit_registry_seed_upsert_delete(self):
        limits = importlib.import_module("emicart.limits.registry")
        limits = importlib.reload(limits)

        standards = limits.get_standards()
        self.assertTrue(len(standards) >= 1)
        store = Path(self.tmp.name) / "Documents" / "EmiCart" / "limits.json"
        self.assertTrue(store.exists())

        curve_obj = limits.upsert_curve(
            standard="TEST-STD",
            curve_name="Curve A",
            units="dBuV",
            breakpoints=[(1e4, 90.0), (1e6, 60.0)],
            resolution_bandwidth_hz=[1e4],
        )
        self.assertEqual(curve_obj.name, "Curve A")
        self.assertIsNotNone(limits.get_curve_by_name("Curve A", standard="TEST-STD"))

        self.assertTrue(limits.delete_curve("TEST-STD", "Curve A"))
        self.assertIsNone(limits.get_curve_by_name("Curve A", standard="TEST-STD"))

    def test_legacy_v_per_m_limit_is_migrated_to_dbuv_per_m(self):
        limits = importlib.import_module("emicart.limits.registry")
        legacy = limits._dict_to_curve(
            {"name": "Legacy E Field", "units": "V/m", "breakpoints": [[1e6, 40], [2e6, 42]]}
        )
        self.assertIsNotNone(legacy)
        self.assertEqual(legacy.units, "dBuV/m")

    def test_units_conversion(self):
        probes = importlib.import_module("emicart.probes.registry")
        probes = importlib.reload(probes)
        units = importlib.import_module("emicart.analysis.units")
        units = importlib.reload(units)

        probe = probes.upsert_probe(name="Conv Probe", measured_units="dBuA", impedance_ohms=50.0)
        data = np.array([100.0, 120.0], dtype=float)
        converted = units.convert_trace_db(data, "dBuV", "dBuA", probe)
        expected_offset = 20.0 * np.log10(50.0)
        np.testing.assert_allclose(converted, data - expected_offset, rtol=1e-10, atol=1e-10)

    def test_units_conversion_to_dbuv_per_m(self):
        probes = importlib.import_module("emicart.probes.registry")
        probes = importlib.reload(probes)
        units = importlib.import_module("emicart.analysis.units")
        units = importlib.reload(units)

        probe = probes.upsert_probe(
            name="EField Probe",
            measured_units="dBuV/m",
            frequency_correction_factors=[(1.0e6, 20.0)],
        )
        data = np.array([120.0], dtype=float)  # 120 dBuV == 1 V
        converted = units.convert_trace_db(data, "dBuV", "dBuV/m", probe, frequencies_hz=np.array([1.0e6]))
        np.testing.assert_allclose(converted, np.array([140.0], dtype=float), rtol=1e-10, atol=1e-10)

    def test_frequency_dependent_db_correction_factors(self):
        probes = importlib.import_module("emicart.probes.registry")
        probes = importlib.reload(probes)
        units = importlib.import_module("emicart.analysis.units")
        units = importlib.reload(units)
        probe = probes.upsert_probe(
            name="Calibrated E-Field Probe",
            measured_units="dBuV/m",
            frequency_correction_factors=[(1.0e6, 10.0), (3.0e6, 30.0)],
        )
        converted = units.convert_trace_db(
            np.array([120.0, 120.0, 120.0]),
            "dBuV",
            "dBuV/m",
            probe,
            frequencies_hz=np.array([0.5e6, 2.0e6, 4.0e6]),
        )
        # Factors interpolate linearly and hold the nearest endpoint beyond the calibration range.
        np.testing.assert_allclose(converted, np.array([130.0, 140.0, 150.0]), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
