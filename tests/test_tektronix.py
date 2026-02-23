import os
import unittest
from unittest import mock

import numpy as np

from emicart.instruments import tektronix


class FakeScope:
    def __init__(self, query_values=None, fail_writes=None, read_value="", fail_query_commands=None):
        self.query_values = dict(query_values or {})
        self.fail_writes = set(fail_writes or [])
        self.fail_query_commands = set(fail_query_commands or [])
        self.read_value = read_value
        self.writes = []
        self.binary_calls = []

    def query(self, command):
        if command in self.fail_query_commands:
            raise RuntimeError(f"failed query: {command}")
        if command not in self.query_values:
            raise RuntimeError(f"unsupported query: {command}")
        value = self.query_values[command]
        if isinstance(value, Exception):
            raise value
        return value

    def write(self, command):
        self.writes.append(command)
        if command in self.fail_writes:
            raise RuntimeError(f"failed write: {command}")

    def read(self):
        return self.read_value

    def query_binary_values(self, command, datatype, container):
        self.binary_calls.append((command, datatype, container))
        if datatype == "h":
            return np.array([-100, 0, 100], dtype=np.int16)
        return np.array([-10, 0, 10], dtype=np.int8)


class TektronixHelperTests(unittest.TestCase):
    def test_parse_idn_detects_mso4_family(self):
        info = tektronix._parse_idn("TEKTRONIX,MSO46,ABC123,1.2.3")
        self.assertEqual(info["model"], "MSO46")
        self.assertTrue(info["supports_wfmo_namespace"])
        self.assertTrue(info["prefers_16bit_waveform"])

    def test_configure_timebase_uses_fallback_commands(self):
        scope = FakeScope(
            fail_writes={
                "HORIZONTAL:SCALE 0.001",
                "HORIZONTAL:RECORDLENGTH 1000",
            }
        )
        tektronix.configure_timebase(scope, time_per_div=0.001, record_length=1000)
        self.assertIn("HOR:MAIN:SCALE 0.001", scope.writes)
        self.assertIn("HOR:RECO 1000", scope.writes)

    def test_get_record_length_fallback(self):
        scope = FakeScope(
            query_values={
                "HORIZONTAL:RECORDLENGTH?": RuntimeError("unsupported"),
                "HOR:RECO?": "2000",
            }
        )
        self.assertEqual(tektronix.get_record_length(scope), 2000)


    def test_is_tektronix_scope_falls_back_to_write_read(self):
        scope = FakeScope(
            query_values={},
            fail_query_commands={"*IDN?"},
            read_value="TEKTRONIX,MSO46,ABC123,1.2.3\n",
        )
        is_tek, idn = tektronix._is_tektronix_scope(scope)
        self.assertTrue(is_tek)
        self.assertIn("MSO46", idn)

    def test_resolve_backend_prefers_environment(self):
        with mock.patch.dict(os.environ, {"EMICART_VISA_BACKEND": "ni"}, clear=False):
            self.assertIsNone(tektronix._resolve_backend("@py"))
        with mock.patch.dict(os.environ, {"EMICART_VISA_BACKEND": "@py"}, clear=False):
            self.assertEqual(tektronix._resolve_backend(None), "@py")

    def test_get_scope_data_supports_16bit_waveforms(self):
        scope = FakeScope(
            query_values={
                "HORIZONTAL:RECORDLENGTH?": "3",
                "WFMO:BYT_N?": "2",
                "WFMO:YMULT?": "0.01",
                "WFMO:YZERO?": "0.0",
                "WFMO:YOFF?": "0",
                "WFMO:XINCR?": "1e-6",
            }
        )
        scope._emicart_idn_info = {"model": "MSO46"}

        volts, dt = tektronix.get_scope_data(scope, max_points=10)

        np.testing.assert_allclose(volts, np.array([-1.0, 0.0, 1.0]))
        self.assertEqual(dt, 1e-6)
        self.assertIn("DATA:WIDTH 2", scope.writes)
        self.assertEqual(scope.binary_calls[-1][1], "h")


if __name__ == "__main__":
    unittest.main()
