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


class TruncatingFakeScope(FakeScope):
    """Simulates a scope whose actual max sample rate is capped at
    max_rate_hz, silently reducing the delivered record length (while keeping
    the requested time/div) whenever the requested record length would need a
    higher rate -- confirmed real behavior on a Tektronix MSO24, verified via
    the HORIZONTAL:RECORDLENGTH? readback (ACQuire:SAMPLERATE? timed out
    entirely on real hardware and must not be used)."""

    def __init__(self, max_rate_hz, **kwargs):
        super().__init__(**kwargs)
        self.max_rate_hz = max_rate_hz
        self.time_per_div = None
        self.requested_record_length = None

    def write(self, command):
        super().write(command)
        if command.startswith("HORIZONTAL:SCALE ") or command.startswith("HOR:MAIN:SCALE "):
            self.time_per_div = float(command.split()[-1])
        elif command.startswith("HORIZONTAL:RECORDLENGTH ") or command.startswith("HOR:RECO "):
            self.requested_record_length = int(command.split()[-1])

    def query(self, command):
        if command in ("HORIZONTAL:RECORDLENGTH?", "HOR:RECO?"):
            if self.time_per_div is None or self.requested_record_length is None:
                return str(self.requested_record_length or 0)
            max_points_at_rate = int(self.time_per_div * tektronix.HORIZONTAL_DIVISIONS * self.max_rate_hz)
            achieved = min(self.requested_record_length, max_points_at_rate)
            return str(achieved)
        return super().query(command)


class MemoryCappedFakeScope(FakeScope):
    """Simulates a scope whose maximum record length is capped by a fixed
    device memory-depth ceiling, independent of time/div or sample rate --
    confirmed real behavior on a Tektronix MSO24 (max ~250,000 points
    regardless of how slow the time/div is set)."""

    def __init__(self, max_record_length, **kwargs):
        super().__init__(**kwargs)
        self.max_record_length = max_record_length
        self.requested_record_length = None

    def write(self, command):
        super().write(command)
        if command.startswith("HORIZONTAL:RECORDLENGTH ") or command.startswith("HOR:RECO "):
            self.requested_record_length = int(command.split()[-1])

    def query(self, command):
        if command in ("HORIZONTAL:RECORDLENGTH?", "HOR:RECO?"):
            if self.requested_record_length is None:
                return "0"
            return str(min(self.requested_record_length, self.max_record_length))
        return super().query(command)


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

    def test_time_per_div_steps_follow_1_2_4_sequence(self):
        steps = tektronix._time_per_div_steps(1e-3, 1.0)
        self.assertEqual(
            steps,
            [1e-3, 2e-3, 4e-3, 1e-2, 2e-2, 4e-2, 1e-1, 2e-1, 4e-1, 1.0],
        )

    def test_compute_time_per_div_for_max_frequency_rounds_down(self):
        # theoretical limit = 1_000_000 / (2 * 10 * 100_000) = 0.5 s/div exactly,
        # but 0.5 is not a valid 1-2-4 step -> round down to 0.4 s/div.
        time_per_div = tektronix.compute_time_per_div_for_max_frequency(
            max_frequency_hz=100_000, record_length=1_000_000
        )
        self.assertEqual(time_per_div, 0.4)

        # theoretical limit = 1_000_000 / (2*10*150_000) = 0.3333... -> round down to 0.2 s/div.
        time_per_div = tektronix.compute_time_per_div_for_max_frequency(
            max_frequency_hz=150_000, record_length=1_000_000
        )
        self.assertEqual(time_per_div, 0.2)

    def test_compute_time_per_div_for_max_frequency_falls_back_to_smallest_step(self):
        # An extremely high required frequency demands a time/div smaller than
        # any supported step; fall back to the smallest available step.
        time_per_div = tektronix.compute_time_per_div_for_max_frequency(
            max_frequency_hz=1e15, record_length=1_000_000, min_time_per_div=1e-9
        )
        self.assertEqual(time_per_div, 1e-9)

    def test_compute_time_per_div_for_max_points_rounds_up(self):
        # theoretical minimum = 1_000_000 / (10 * 1e9) = 1e-4 s/div exactly.
        time_per_div = tektronix.compute_time_per_div_for_max_points(
            record_length=1_000_000, max_sample_rate_hz=1e9
        )
        self.assertEqual(time_per_div, 1e-4)

        # theoretical minimum = 1_000_000 / (10 * 2.5e9) = 4e-5 s/div exactly.
        time_per_div = tektronix.compute_time_per_div_for_max_points(
            record_length=1_000_000, max_sample_rate_hz=2.5e9
        )
        self.assertEqual(time_per_div, 4e-5)

    def test_compute_time_per_div_for_max_points_falls_back_to_largest_step(self):
        # An extremely low max sample rate demands a time/div larger than any
        # supported step; fall back to the largest available step.
        time_per_div = tektronix.compute_time_per_div_for_max_points(
            record_length=1_000_000, max_sample_rate_hz=1e-3, max_time_per_div=100.0
        )
        self.assertEqual(time_per_div, 100.0)

    def test_disable_unused_channels_leaves_only_target_channel_on(self):
        scope = FakeScope()
        tektronix._disable_unused_channels(scope, channel="CH1", channel_count=4)
        self.assertIn("SELECT:CH1 ON", scope.writes)
        self.assertIn("SELECT:CH2 OFF", scope.writes)
        self.assertIn("SELECT:CH3 OFF", scope.writes)
        self.assertIn("SELECT:CH4 OFF", scope.writes)

    def test_disable_unused_channels_ignores_missing_channels(self):
        # A 2-channel model rejects SELECT:CH3/CH4; the helper should not raise.
        scope = FakeScope(fail_writes={"SELECT:CH3 OFF", "SELECT:CH4 OFF"})
        tektronix._disable_unused_channels(scope, channel="CH1", channel_count=4)
        self.assertIn("SELECT:CH1 ON", scope.writes)
        self.assertIn("SELECT:CH2 OFF", scope.writes)

    def test_apply_autoscale_disables_other_channels_before_autoset(self):
        scope = TruncatingFakeScope(max_rate_hz=2.5e9)
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            tektronix.apply_autoscale(
                scope,
                record_length=250_000,
                required_max_frequency_hz=None,
                min_time_per_div=1e-6,
                max_time_per_div=1.0,
            )
        autoset_index = scope.writes.index("AutoSet EXEC")
        self.assertLess(scope.writes.index("SELECT:CH2 OFF"), autoset_index)
        self.assertLess(scope.writes.index("SELECT:CH1 ON"), autoset_index)

    def test_apply_autoscale_uses_largest_time_per_div_for_limit_curve(self):
        scope = FakeScope()
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            info = tektronix.apply_autoscale(
                scope,
                record_length=1_000_000,
                required_max_frequency_hz=100_000,
            )
        self.assertEqual(info["scenario"], "limit_curve")
        self.assertEqual(info["time_per_div"], 0.4)
        self.assertIn("AutoSet EXEC", scope.writes)
        self.assertIn("HORIZONTAL:SCALE 0.4", scope.writes)
        self.assertIn("HORIZONTAL:RECORDLENGTH 1000000", scope.writes)

    def test_apply_autoscale_limit_curve_uses_true_achievable_record_length(self):
        # Regression test: confirmed real-world symptom on a Tektronix MSO24
        # -- for MIL-STD-461G CE102 (10 MHz top breakpoint), the scope's true
        # record-length ceiling is 250,000 points (not the requested
        # 1,000,000), so the Nyquist time/div ceiling must be computed from
        # 250,000, giving 1 ms/div. Computing it from the unachievable
        # 1,000,000 request instead gave a theoretical ceiling of 5 ms/div,
        # which rounds down (1-2-4 steps) to the wrong answer, 4 ms/div.
        scope = MemoryCappedFakeScope(max_record_length=250_000)
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            info = tektronix.apply_autoscale(
                scope,
                record_length=1_000_000,
                required_max_frequency_hz=10_000_000,
            )
        self.assertEqual(info["scenario"], "limit_curve")
        self.assertEqual(info["time_per_div"], 1e-3)
        self.assertIn("HORIZONTAL:SCALE 0.001", scope.writes)
        self.assertIn("HORIZONTAL:RECORDLENGTH 250000", scope.writes)

    def test_apply_autoscale_uses_smallest_time_per_div_verified_against_hardware(self):
        # Scope's true max rate is 2.5 GS/s -- matches confirmed real MSO24
        # behavior: full 250k points achievable at 10 us/div (the "optimal"
        # answer), but not at 1 us/div (would need 25 GS/s).
        scope = TruncatingFakeScope(max_rate_hz=2.5e9)
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            info = tektronix.apply_autoscale(
                scope,
                record_length=250_000,
                required_max_frequency_hz=None,
                min_time_per_div=1e-6,
                max_time_per_div=1.0,
            )
        self.assertEqual(info["scenario"], "max_points")
        self.assertEqual(info["time_per_div"], 1e-5)
        self.assertIn("HORIZONTAL:SCALE 1e-05", scope.writes)

    def test_apply_autoscale_reports_every_scpi_command_via_on_step(self):
        # Regression test: the UI status bar surfaces the literal SCPI
        # commands sent during autoscale via the on_step callback -- make
        # sure every scope.write() that apply_autoscale performs is also
        # reported as a "command_sent" on_step event with the exact text.
        scope = TruncatingFakeScope(max_rate_hz=2.5e9)
        reported_commands = []

        def on_step(event, info):
            if event == "command_sent":
                reported_commands.append(info["command"])

        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            tektronix.apply_autoscale(
                scope,
                record_length=250_000,
                required_max_frequency_hz=None,
                min_time_per_div=1e-6,
                max_time_per_div=1.0,
                termination_ohms=50.0,
                on_step=on_step,
            )

        # Every actual scope.write() call apply_autoscale makes must also
        # have been reported through on_step, in the same order.
        write_like_commands = [
            w
            for w in scope.writes
            if w.startswith(("SELECT:", "CH1:", "AutoSet", "HORIZONTAL:"))
        ]
        self.assertEqual(reported_commands, write_like_commands)
        self.assertIn("AutoSet EXEC", reported_commands)
        self.assertIn("CH1:TERMination 50", reported_commands)

    def test_find_min_time_per_div_for_full_record_matches_hardware_truncation(self):
        scope = TruncatingFakeScope(max_rate_hz=2.5e9)
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            time_per_div = tektronix._find_min_time_per_div_for_full_record(
                scope, record_length=250_000, min_time_per_div=1e-6, max_time_per_div=1.0
            )
        self.assertEqual(time_per_div, 1e-5)

    def test_find_min_time_per_div_ignores_stale_fast_timebase_left_by_autoset(self):
        # Regression test: if AutoSet (or a prior step) leaves the scope at a
        # fast time/div (e.g. 4 us/div) BEFORE the record-length ceiling is
        # resolved, the sample-rate-based truncation at that fast setting
        # must not be mistaken for the scope's true achievable record length.
        # Confirmed real-world symptom: scope converged on 4 us/div (only
        # ~100 kpts) instead of the correct 10 us/div (full 250 kpts).
        scope = TruncatingFakeScope(max_rate_hz=2.5e9)
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            scope.write("HORIZONTAL:SCALE 4e-06")  # simulate AutoSet's leftover fast timebase
            time_per_div = tektronix._find_min_time_per_div_for_full_record(
                scope, record_length=250_000, min_time_per_div=1e-6, max_time_per_div=1.0
            )
        self.assertEqual(time_per_div, 1e-5)

    def test_resolve_achievable_record_length_forces_safe_timebase_first(self):
        scope = TruncatingFakeScope(max_rate_hz=2.5e9)
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            scope.write("HORIZONTAL:SCALE 4e-06")  # simulate a stale fast timebase
            achievable = tektronix._resolve_achievable_record_length(
                scope, 250_000, safe_time_per_div=1.0
            )
        self.assertEqual(achievable, 250_000)
        self.assertIn("HORIZONTAL:SCALE 1.0", scope.writes)

    def test_resolve_achievable_record_length_returns_actual_if_clamped(self):
        # Confirmed real MSO24 behavior: requesting 1,000,000 points in AUTO
        # horizontal mode silently settles at a smaller value regardless of
        # time/div. The search must target that real value, not the
        # unreachable request.
        scope = FakeScope(query_values={"HORIZONTAL:RECORDLENGTH?": "250000"})
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            achievable = tektronix._resolve_achievable_record_length(scope, 1_000_000)
        self.assertEqual(achievable, 250000)
        self.assertIn("HORIZONTAL:RECORDLENGTH 1000000", scope.writes)

    def test_resolve_achievable_record_length_falls_back_to_desired_if_unqueryable(self):
        scope = FakeScope(fail_query_commands={"HORIZONTAL:RECORDLENGTH?", "HOR:RECO?"})
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            achievable = tektronix._resolve_achievable_record_length(scope, 1_000_000)
        self.assertEqual(achievable, 1_000_000)

    def test_set_channel_termination_writes_ohms_value(self):
        scope = FakeScope()
        tektronix.set_channel_termination(scope, 50.0, channel="CH1")
        self.assertIn("CH1:TERMination 50", scope.writes)

    def test_set_channel_termination_falls_back_to_impedance_command(self):
        scope = FakeScope(fail_writes={"CH1:TERMination 1e+06"})
        tektronix.set_channel_termination(scope, 1_000_000.0, channel="CH1")
        self.assertIn("CH1:IMPedance 1e+06", scope.writes)

    def test_set_channel_termination_rejects_unsupported_value(self):
        scope = FakeScope()
        with self.assertRaises(ValueError):
            tektronix.set_channel_termination(scope, 75.0, channel="CH1")
        self.assertEqual(scope.writes, [])

    def test_apply_autoscale_commands_termination_before_autoset(self):
        scope = FakeScope()
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            info = tektronix.apply_autoscale(
                scope,
                record_length=1_000_000,
                required_max_frequency_hz=100_000,
                termination_ohms=50.0,
            )
        autoset_index = scope.writes.index("AutoSet EXEC")
        self.assertLess(scope.writes.index("CH1:TERMination 50"), autoset_index)
        self.assertEqual(info["termination_ohms"], 50.0)

    def test_apply_autoscale_skips_termination_when_not_provided(self):
        scope = FakeScope()
        with mock.patch("emicart.instruments.tektronix.time.sleep"):
            info = tektronix.apply_autoscale(
                scope,
                record_length=1_000_000,
                required_max_frequency_hz=100_000,
            )
        self.assertFalse(any("TERMination" in w or "IMPedance" in w for w in scope.writes))
        self.assertIsNone(info["termination_ohms"])

    def test_get_max_sample_rate_returns_query_value(self):
        scope = FakeScope(query_values={"ACQuire:MAXSamplerate?": "1e9"})
        self.assertEqual(tektronix._get_max_sample_rate(scope), 1e9)

    def test_get_max_sample_rate_falls_back_to_default_if_unqueryable(self):
        scope = FakeScope(fail_query_commands={"ACQuire:MAXSamplerate?", "ACQ:MAXS?"})
        self.assertEqual(
            tektronix._get_max_sample_rate(scope), tektronix.DEFAULT_MAX_SAMPLE_RATE_HZ
        )

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

    def test_get_scope_data_does_not_cache_stale_waveform_metadata(self):
        # Regression test: confirmed real-world symptom -- after changing the
        # scope's timebase for a second capture (e.g. a different limit
        # curve requiring a different time/div), get_scope_data() returned
        # the FIRST capture's stale xincr (sample interval) instead of
        # re-querying the instrument, silently reporting a wrong frequency
        # axis while the reported time/div and record_length remained
        # correct. Verified by simulating a timebase change between two
        # get_scope_data() calls and checking dt is freshly re-read each time.
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

        _, first_dt = tektronix.get_scope_data(scope, max_points=10)
        self.assertEqual(first_dt, 1e-6)

        # Simulate the scope now being at a different (autoscaled) timebase.
        scope.query_values["WFMO:XINCR?"] = "4e-8"
        _, second_dt = tektronix.get_scope_data(scope, max_points=10)
        self.assertEqual(second_dt, 4e-8)


if __name__ == "__main__":
    unittest.main()
