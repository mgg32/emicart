import math
import time
import os
import threading

import numpy as np
import pyvisa


DEFAULT_RESOURCE = None
# On Windows, use the installed system VISA implementation by default.  USBTMC
# devices commonly use the vendor IVI driver there; pyvisa-py requires a
# separate libusb backend and otherwise lists no USB instruments.
DEFAULT_BACKEND = None if os.name == "nt" else "@py"
DEFAULT_BACKEND_ENV = "EMICART_VISA_BACKEND"
DEFAULT_OPEN_TIMEOUT_MS = 3000
DEFAULT_IO_TIMEOUT_MS = 5000
DEFAULT_MAX_POINTS = 1_000_000



def _resolve_backend(backend):
    env_backend = os.environ.get(DEFAULT_BACKEND_ENV)
    selected = backend
    if env_backend is not None:
        selected = env_backend.strip()

    normalized = (selected or "").strip().lower()
    if normalized in {"", "default", "system", "ni", "@ni", "ni-visa", "nivisa"}:
        return None
    return selected


def _open_resource_manager(backend):
    if backend is None:
        return pyvisa.ResourceManager()
    return pyvisa.ResourceManager(backend)

def _prioritized_candidates(resources):
    tcpip = [r for r in resources if "TCPIP" in r]
    usb_tek = [r for r in resources if "USB" in r and "0x0699" in r]
    usb_other = [r for r in resources if "USB" in r and r not in usb_tek]
    # Prefer network scopes first to avoid USB backend hangs on some Pi setups.
    tcpip_inst0 = [r for r in tcpip if "::inst0::INSTR" in r]
    tcpip_other = [r for r in tcpip if r not in tcpip_inst0]
    return tcpip_inst0 + tcpip_other + usb_tek + usb_other


def _open_resource(rm, resource_name, open_timeout_ms):
    result = {"scope": None, "error": None}

    def worker():
        try:
            try:
                scope = rm.open_resource(resource_name, open_timeout=open_timeout_ms)
            except TypeError:
                # Some backends/versions do not support open_timeout keyword.
                scope = rm.open_resource(resource_name)
            result["scope"] = scope
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    wait_s = max(float(open_timeout_ms) / 1000.0, 0.5) + 0.5
    t.join(wait_s)
    if t.is_alive():
        raise TimeoutError(f"open_resource timed out after ~{wait_s:.1f}s")
    if result["error"] is not None:
        raise result["error"]
    if result["scope"] is None:
        raise RuntimeError("open_resource returned no resource and no error.")
    return result["scope"]


def _is_tektronix_scope(scope):
    try:
        idn = scope.query("*IDN?").strip()
    except Exception:
        try:
            scope.write("*IDN?")
            idn = scope.read().strip()
        except Exception:
            return False, ""
    return ("TEKTRONIX" in idn.upper()), idn


def _parse_idn(idn):
    text = (idn or "").strip()
    parts = [p.strip() for p in text.split(",")]
    vendor = parts[0] if len(parts) >= 1 else ""
    model = parts[1] if len(parts) >= 2 else ""
    serial = parts[2] if len(parts) >= 3 else ""
    firmware = parts[3] if len(parts) >= 4 else ""

    model_upper = model.upper()
    is_mso4_family = model_upper.startswith("MSO4")
    return {
        "raw": text,
        "vendor": vendor,
        "model": model,
        "serial": serial,
        "firmware": firmware,
        "supports_wfmo_namespace": is_mso4_family,
        "prefers_16bit_waveform": is_mso4_family,
    }


def _query_first(scope, commands, parser=lambda x: x):
    last_error = None
    for command in commands:
        try:
            return parser(scope.query(command)), command
        except Exception as e:
            last_error = e
    if last_error is None:
        raise RuntimeError("No query commands provided.")
    raise last_error


def _write_first(scope, commands):
    last_error = None
    for command in commands:
        try:
            scope.write(command)
            return command
        except Exception as e:
            last_error = e
    if last_error is None:
        raise RuntimeError("No write commands provided.")
    raise last_error


def _read_waveform_metadata(scope):
    # NOTE: this must NEVER be cached across calls. It reflects the CURRENT
    # timebase/vertical-scale state (xincr, ymult, yzero, yoff), which
    # apply_autoscale() changes before every capture. A previous version of
    # this function cached the result on the scope object after the first
    # read -- on any capture after the first, that reused the PRIOR
    # capture's stale xincr (sample interval), silently computing the wrong
    # frequency axis (e.g. reporting a much lower Nyquist frequency than the
    # freshly-autoscaled timebase actually delivers) while still reporting
    # the correct (fresh) time/div and record_length elsewhere.
    try:
        model_upper = str(getattr(scope, "_emicart_idn_info", {}).get("model", "")).upper()
    except Exception:
        model_upper = ""
    prefers_wfmo = model_upper.startswith("MSO4")

    bytenr_order = ["WFMO:BYT_N?", "WFMPRE:BYT_NR?"]
    if not prefers_wfmo:
        bytenr_order = ["WFMPRE:BYT_NR?", "WFMO:BYT_N?"]
    byte_width, _ = _query_first(scope, bytenr_order, parser=lambda v: int(float(v)))
    byte_width = 2 if int(byte_width) >= 2 else 1

    float_parser = lambda v: float(v)
    ymult, ymult_cmd = _query_first(scope, ["WFMO:YMULT?", "WFMPRE:YMULT?"] if prefers_wfmo else ["WFMPRE:YMULT?", "WFMO:YMULT?"], parser=float_parser)
    yzero, yzero_cmd = _query_first(scope, ["WFMO:YZERO?", "WFMPRE:YZERO?"] if prefers_wfmo else ["WFMPRE:YZERO?", "WFMO:YZERO?"], parser=float_parser)
    yoff, yoff_cmd = _query_first(scope, ["WFMO:YOFF?", "WFMPRE:YOFF?"] if prefers_wfmo else ["WFMPRE:YOFF?", "WFMO:YOFF?"], parser=float_parser)
    xincr, xincr_cmd = _query_first(scope, ["WFMO:XINCR?", "WFMPRE:XINCR?"] if prefers_wfmo else ["WFMPRE:XINCR?", "WFMO:XINCR?"], parser=float_parser)

    return {
        "byte_width": byte_width,
        "datatype": "h" if byte_width == 2 else "b",
        "ymult": ymult,
        "yzero": yzero,
        "yoff": yoff,
        "xincr": xincr,
        "commands": {
            "ymult": ymult_cmd,
            "yzero": yzero_cmd,
            "yoff": yoff_cmd,
            "xincr": xincr_cmd,
        },
    }


def _get_record_length(scope, default_points):
    try:
        value, _ = _query_first(
            scope,
            ["HORIZONTAL:RECORDLENGTH?", "HOR:RECO?"],
            parser=lambda v: int(float(v)),
        )
        return int(value)
    except Exception:
        try:
            value, _ = _query_first(
                scope,
                ["WFMO:NR_PT?", "WFMPRE:NR_PT?"],
                parser=lambda v: int(float(v)),
            )
            return int(value)
        except Exception:
            return int(default_points)


def get_record_length(scope):
    return _get_record_length(scope, DEFAULT_MAX_POINTS)


def get_time_per_div(scope):
    """Read back the scope's currently configured horizontal time/div, for
    diagnostics/status reporting."""
    value, _ = _query_first(
        scope,
        ["HORizontal:SCAle?", "HOR:MAIN:SCALE?"],
        parser=lambda v: float(v),
    )
    return value


def connect_to_scope(
    resource_str=None,
    backend=DEFAULT_BACKEND,
    open_timeout_ms=DEFAULT_OPEN_TIMEOUT_MS,
    io_timeout_ms=DEFAULT_IO_TIMEOUT_MS,
):
    env_resource = os.environ.get("EMICART_SCOPE_RESOURCE", "").strip()
    preferred = resource_str or env_resource
    selected_backend = _resolve_backend(backend)
    backend_desc = selected_backend if selected_backend is not None else "system-default"
    print(f"Using VISA backend: {backend_desc}")
    rm = _open_resource_manager(selected_backend)
    resources = tuple(rm.list_resources())
    if not resources:
        raise RuntimeError("No VISA instruments detected.")

    if preferred:
        if preferred not in resources:
            raise RuntimeError(
                "Configured scope resource was not found.\n"
                f"Configured: {preferred}\n"
                f"Detected: {', '.join(resources)}"
            )
        print(f"Connecting to {preferred}...")
        scope = _open_resource(rm, preferred, open_timeout_ms)
        scope.timeout = io_timeout_ms
        scope.read_termination = "\n"
        scope.write_termination = "\n"
        is_tek, idn = _is_tektronix_scope(scope)
        if not is_tek:
            try:
                scope.close()
            except Exception:
                pass
            raise RuntimeError(
                f"Connected to non-Tek instrument at {preferred}: {idn or 'No IDN response'}"
            )
        info = _parse_idn(idn)
        scope._emicart_idn_info = info
        print(f"Connected to: {idn}")
        print(f"Detected Tek model: {info['model'] or 'Unknown'} (firmware {info['firmware'] or 'unknown'})")
        return scope

    candidates = _prioritized_candidates(resources)
    if not candidates:
        raise RuntimeError(
            "No USB/TCPIP VISA resources found.\n"
            f"Detected resources: {', '.join(resources)}"
        )

    probe_timeout_ms = min(max(int(io_timeout_ms), 500), 1500)
    errors = []
    for candidate in candidates:
        print(f"Connecting to {candidate}...")
        scope = None
        try:
            scope = _open_resource(rm, candidate, open_timeout_ms)
            scope.timeout = probe_timeout_ms
            scope.read_termination = "\n"
            scope.write_termination = "\n"
            is_tek, idn = _is_tektronix_scope(scope)
            if not is_tek:
                errors.append(f"{candidate}: not Tektronix ({idn or 'no IDN'})")
                scope.close()
                continue
            info = _parse_idn(idn)
            scope._emicart_idn_info = info
            scope.timeout = io_timeout_ms
            print(f"Connected to: {idn}")
            print(
                f"Detected Tek model: {info['model'] or 'Unknown'} "
                f"(firmware {info['firmware'] or 'unknown'})"
            )
            return scope
        except Exception as e:
            errors.append(f"{candidate}: {e}")
            if scope is not None:
                try:
                    scope.close()
                except Exception:
                    pass

    raise RuntimeError(
        "Unable to connect to a Tek scope over USB/TCPIP.\n"
        f"Detected resources: {', '.join(resources)}\n"
        f"Attempts: {' | '.join(errors)}\n"
        "Set EMICART_SCOPE_RESOURCE to the correct resource if needed."
    )


def setup_scope(scope):
    scope.write("AutoSet EXEC")
    time.sleep(5)
    scope.write("HOR:MAIN:SCALE .0004")
    time.sleep(1)


def configure_timebase(scope, time_per_div, record_length, on_step=None):
    scale_cmd = _write_first(
        scope,
        [
            f"HORIZONTAL:SCALE {time_per_div}",
            f"HOR:MAIN:SCALE {time_per_div}",
        ],
    )
    if on_step:
        try:
            on_step("command_sent", {"command": scale_cmd})
        except Exception:
            pass
    reco_cmd = _write_first(
        scope,
        [
            f"HORIZONTAL:RECORDLENGTH {record_length}",
            f"HOR:RECO {record_length}",
        ],
    )
    if on_step:
        try:
            on_step("command_sent", {"command": reco_cmd})
        except Exception:
            pass
    time.sleep(0.5)


# Tektronix scopes display 10 horizontal divisions across the full record.
HORIZONTAL_DIVISIONS = 10
# Practical bounds for the requested time/div; the scope will clamp to its own
# supported range if a value outside this falls outside what it can accept.
# 100ps/div covers even the fastest current Tektronix models (e.g. 6 Series B
# scopes reach tens of GS/s); the scope itself will clamp further if a given
# model's true fastest timebase is slower than this.
MIN_TIME_PER_DIV_S = 1e-10
MAX_TIME_PER_DIV_S = 100.0
# Confirmed on real Tektronix MSO24 hardware: the 2/4/5/6-Series MSO family
# steps its horizontal scale through a 1-2-4-10 decade sequence (1, 2, 4, 10,
# 20, 40, 100, ...), NOT the 1-2-5-10 sequence used by older TDS/DPO scopes.
# Requesting an invalid intermediate value (e.g. 5us/div) causes the scope to
# silently snap to the nearest actual step (e.g. 4us/div) -- this was the
# root cause of the autoscale landing on unexpected values like 4us/div or
# 40us/div instead of the intended 10us/div.
_TIME_PER_DIV_MULTIPLIERS = (1.0, 2.0, 4.0)
# Conservative fallback if the scope cannot report its true maximum real-time
# sample rate (older models / unsupported query).
DEFAULT_MAX_SAMPLE_RATE_HZ = 1.0e9


def _time_per_div_steps(min_step=MIN_TIME_PER_DIV_S, max_step=MAX_TIME_PER_DIV_S):
    """Return the sorted 1-2-4 (per-decade) sequence of time/div values between
    min_step and max_step, matching the real Tektronix MSO2/4/5/6-Series
    horizontal scale steps."""
    if min_step <= 0 or max_step <= 0 or max_step < min_step:
        raise ValueError("min_step and max_step must be positive with max_step >= min_step.")
    start_exp = int(math.floor(math.log10(min_step)))
    end_exp = int(math.ceil(math.log10(max_step)))
    steps = set()
    for exp in range(start_exp, end_exp + 1):
        decade = 10.0 ** exp
        for mult in _TIME_PER_DIV_MULTIPLIERS:
            value = mult * decade
            if min_step * (1 - 1e-9) <= value <= max_step * (1 + 1e-9):
                steps.add(round(value, 15))
    return sorted(steps)


def _largest_step_at_most(value, steps):
    """Return the largest step <= value, falling back to the smallest step if
    every step exceeds value (e.g. a very high required frequency)."""
    eligible = [s for s in steps if s <= value * (1 + 1e-9)]
    if eligible:
        return max(eligible)
    return min(steps)


def _smallest_step_at_least(value, steps):
    """Return the smallest step >= value, falling back to the largest step if
    every step is smaller (e.g. an unusually low max sample rate)."""
    eligible = [s for s in steps if s >= value * (1 - 1e-9)]
    if eligible:
        return min(eligible)
    return max(steps)


def compute_time_per_div_for_max_frequency(
    max_frequency_hz,
    record_length,
    min_time_per_div=MIN_TIME_PER_DIV_S,
    max_time_per_div=MAX_TIME_PER_DIV_S,
):
    """Largest supported time/div whose Nyquist frequency still covers max_frequency_hz.

    sample_rate = record_length / (time_per_div * HORIZONTAL_DIVISIONS)
    Nyquist = sample_rate / 2 must be >= max_frequency_hz, so:
        time_per_div <= record_length / (2 * HORIZONTAL_DIVISIONS * max_frequency_hz)
    The result is rounded down to the nearest standard 1-2-4 step so the
    requirement is never violated due to instrument step granularity.
    """
    if max_frequency_hz is None or max_frequency_hz <= 0:
        raise ValueError("max_frequency_hz must be > 0.")
    if record_length is None or record_length <= 0:
        raise ValueError("record_length must be > 0.")
    theoretical_limit = record_length / (2 * HORIZONTAL_DIVISIONS * max_frequency_hz)
    steps = _time_per_div_steps(min_time_per_div, max_time_per_div)
    return _largest_step_at_most(theoretical_limit, steps)


def compute_time_per_div_for_max_points(
    record_length,
    max_sample_rate_hz,
    min_time_per_div=MIN_TIME_PER_DIV_S,
    max_time_per_div=MAX_TIME_PER_DIV_S,
):
    """Smallest supported time/div whose required sample rate does not exceed
    the scope's true maximum sample rate.

    Requesting a time/div faster than the scope can actually sustain for
    record_length does not increase the number of captured points -- the
    scope instead truncates the delivered record to whatever it can sample
    at its real maximum rate. To actually maximize sampled points we pick the
    smallest time/div whose implied sample rate (record_length / (time_per_div
    * HORIZONTAL_DIVISIONS)) still fits within max_sample_rate_hz:
        time_per_div >= record_length / (HORIZONTAL_DIVISIONS * max_sample_rate_hz)
    The result is rounded up to the nearest standard 1-2-4 step so the
    scope's real sample rate ceiling is never exceeded.
    """
    if max_sample_rate_hz is None or max_sample_rate_hz <= 0:
        raise ValueError("max_sample_rate_hz must be > 0.")
    if record_length is None or record_length <= 0:
        raise ValueError("record_length must be > 0.")
    theoretical_min = record_length / (HORIZONTAL_DIVISIONS * max_sample_rate_hz)
    steps = _time_per_div_steps(min_time_per_div, max_time_per_div)
    return _smallest_step_at_least(theoretical_min, steps)


def _get_max_sample_rate(scope, default_sample_rate_hz=DEFAULT_MAX_SAMPLE_RATE_HZ):
    """Query the scope's maximum real-time sample rate (samples/sec) for the
    currently selected channel configuration. Falls back to a conservative
    default if the instrument does not support the query.

    NOTE: on some instruments/firmware this query does not accurately reflect
    what is actually achievable (it can be stale or model-independent), so
    `_find_min_time_per_div_for_full_record` verifies against real hardware
    readback instead of trusting this value outright.
    """
    try:
        value, _ = _query_first(
            scope,
            ["ACQuire:MAXSamplerate?", "ACQ:MAXS?"],
            parser=lambda v: float(v),
        )
        if value and value > 0:
            return float(value)
    except Exception:
        pass
    return float(default_sample_rate_hz)


# Many Tektronix scopes divide their maximum real-time sample rate by the
# number of enabled channels (e.g. full rate with 1 channel active, a quarter
# rate with 4 active). Since captures only ever read one channel, leaving
# unused channels enabled needlessly throttles the achievable sample rate.
CAPTURE_CHANNEL = "CH1"
MAX_ANALOG_CHANNELS = 4
# A record length is considered "achieved" if it comes within this fraction of
# the requested length (scopes sometimes deliver a handful fewer points).
RECORD_LENGTH_TOLERANCE = 0.98


def _disable_unused_channels(scope, channel=CAPTURE_CHANNEL, channel_count=MAX_ANALOG_CHANNELS, on_step=None):
    """Ensure only `channel` is enabled so it gets the scope's full per-channel
    sample rate. Channels the model does not have are silently ignored."""
    for i in range(1, channel_count + 1):
        ch_name = f"CH{i}"
        state = "ON" if ch_name == channel else "OFF"
        cmd = f"SELECT:{ch_name} {state}"
        try:
            scope.write(cmd)
            if on_step:
                try:
                    on_step("command_sent", {"command": cmd})
                except Exception:
                    pass
        except Exception:
            pass


# Scope input termination options this app can command ahead of a capture.
# 50 ohm matches devices with a 50 ohm coax output (LISNs, current probes);
# 1 Mohm is the high-impedance option most passive/active voltage probes and
# antennas expect.
TERMINATION_OPTIONS_OHMS = (50.0, 1_000_000.0)


def set_channel_termination(scope, termination_ohms, channel=CAPTURE_CHANNEL, on_step=None):
    """Command the scope's input termination for `channel` to match the
    active probe (50 ohm or 1 Mohm). Raises ValueError for any other value so
    a typo/unsupported value fails loudly instead of silently mis-terminating
    the input."""
    if not any(math.isclose(termination_ohms, option) for option in TERMINATION_OPTIONS_OHMS):
        raise ValueError(
            f"termination_ohms must be one of {TERMINATION_OPTIONS_OHMS}, got {termination_ohms!r}."
        )
    cmd = _write_first(
        scope,
        [
            f"{channel}:TERMination {termination_ohms:g}",
            f"{channel}:IMPedance {termination_ohms:g}",
        ],
    )
    if on_step:
        try:
            on_step("command_sent", {"command": cmd})
        except Exception:
            pass



def get_channel_termination(scope, channel=CAPTURE_CHANNEL):
    """Read back the scope's actual configured input termination (ohms) for
    `channel`, for status reporting/verification."""
    value, _ = _query_first(
        scope,
        [f"{channel}:TERMination?", f"{channel}:IMPedance?"],
        parser=lambda v: float(v),
    )
    return value


def _resolve_achievable_record_length(
    scope, desired_record_length, safe_time_per_div=MAX_TIME_PER_DIV_S, on_step=None
):
    """Some scopes silently cap the maximum usable record length depending on
    the CURRENT time/div (confirmed on a Tektronix MSO24: at fast time/div
    settings the achievable record length shrinks because it is bounded by
    sample rate -- e.g. only ~100k points at 4us/div vs the true ~250k device
    ceiling at 10us/div or slower). Whatever left the scope at a fast
    time/div before this call (AutoSet, a previous capture, ...) would
    otherwise cause us to read back that smaller, sample-rate-limited value
    and mistake it for the true memory-depth ceiling. To avoid that, force a
    very slow, safe time/div first (where sample rate is never the
    bottleneck) before writing the desired record length and reading back
    what the instrument actually accepts -- that reveals the real ceiling,
    independent of whatever time/div happens to be active.

    `on_step`, if provided, is called with (event_name, info_dict) after each
    notable action, for diagnostics."""
    scale_cmd = _write_first(
        scope,
        [
            f"HORIZONTAL:SCALE {safe_time_per_div}",
            f"HOR:MAIN:SCALE {safe_time_per_div}",
        ],
    )
    if on_step:
        try:
            on_step("command_sent", {"command": scale_cmd})
        except Exception:
            pass
    time.sleep(1.0)
    if on_step:
        try:
            on_step("safe_timebase_set", {"safe_time_per_div": safe_time_per_div, "readback": get_record_length(scope)})
        except Exception:
            pass
    reco_cmd = _write_first(
        scope,
        [
            f"HORIZONTAL:RECORDLENGTH {desired_record_length}",
            f"HOR:RECO {desired_record_length}",
        ],
    )
    if on_step:
        try:
            on_step("command_sent", {"command": reco_cmd})
        except Exception:
            pass
    time.sleep(1.0)
    try:
        achievable = get_record_length(scope)
        if on_step:
            on_step(
                "record_length_ceiling_resolved",
                {"desired_record_length": desired_record_length, "achievable_record_length": achievable},
            )
        if achievable and achievable > 0:
            return achievable
    except Exception as e:
        if on_step:
            on_step("record_length_ceiling_query_failed", {"error": repr(e)})
    return desired_record_length


def _find_min_time_per_div_for_full_record(
    scope,
    record_length,
    min_time_per_div=MIN_TIME_PER_DIV_S,
    max_time_per_div=MAX_TIME_PER_DIV_S,
    on_step=None,
):
    """Binary-search the standard 1-2-4 time/div steps for the SMALLEST value
    the scope can actually deliver the full record length at, verified
    against real hardware readback.

    Requesting a time/div faster than the scope can sustain does not increase
    the number of captured points -- the scope instead keeps the time/div we
    asked for but silently reduces the record length to whatever it can
    sample at its true max rate (confirmed on a Tektronix MSO24: at 1 us/div
    with a 250,000-point target and a 2.5 GS/s hardware limit, the scope
    reported only 25,000 points -- exactly 250,000 * (1us/10us) -- rather than
    slowing down or erroring). We verify each candidate by writing it and
    reading back HORIZONTAL:RECORDLENGTH? (confirmed fast and reliable --
    unlike ACQuire:SAMPLERATE?, which timed out entirely on this instrument
    and must not be used). Achieved record length is non-decreasing as
    time/div increases, so binary search finds the smallest sufficient step.

    `on_step`, if provided, is called with (event_name, info_dict) after each
    notable action, for diagnostics."""
    achievable_record_length = _resolve_achievable_record_length(
        scope, record_length, safe_time_per_div=max_time_per_div, on_step=on_step
    )
    steps = _time_per_div_steps(min_time_per_div, max_time_per_div)

    def achieves_full_record(candidate):
        configure_timebase(scope, time_per_div=candidate, record_length=achievable_record_length, on_step=on_step)
        time.sleep(0.5)
        try:
            achieved = get_record_length(scope)
        except Exception as e:
            if on_step:
                on_step("candidate_query_failed", {"candidate": candidate, "error": repr(e)})
            return False
        result = achieved >= achievable_record_length * RECORD_LENGTH_TOLERANCE
        if on_step:
            on_step(
                "candidate_tested",
                {
                    "candidate": candidate,
                    "achieved_record_length": achieved,
                    "target_record_length": achievable_record_length,
                    "success": result,
                },
            )
        return result

    lo, hi = 0, len(steps) - 1
    best = steps[-1]  # The slowest step is always assumed achievable as a fallback.
    while lo <= hi:
        mid = (lo + hi) // 2
        if achieves_full_record(steps[mid]):
            best = steps[mid]
            hi = mid - 1  # Search for an even smaller (faster) step.
        else:
            lo = mid + 1  # Too fast; need a larger (slower) step.

    # The binary search loop leaves the scope at whatever candidate was tested
    # LAST, which is not necessarily `best` (e.g. the search may finish on a
    # failing candidate that was slower than `best`, or an earlier failing
    # candidate faster than it). Explicitly re-apply the winning value so the
    # scope's actual state always matches what this function returns.
    configure_timebase(scope, time_per_div=best, record_length=achievable_record_length, on_step=on_step)
    time.sleep(0.5)
    if on_step:
        try:
            on_step(
                "best_reapplied",
                {"time_per_div": best, "record_length": get_record_length(scope)},
            )
        except Exception:
            pass
    return best


def apply_autoscale(
    scope,
    *,
    record_length=DEFAULT_MAX_POINTS,
    required_max_frequency_hz=None,
    channel=CAPTURE_CHANNEL,
    min_time_per_div=MIN_TIME_PER_DIV_S,
    max_time_per_div=MAX_TIME_PER_DIV_S,
    termination_ohms=None,
    on_step=None,
):
    """Auto-scale the scope ahead of a capture.

    1. Disables all channels other than `channel` so it gets the scope's full
       per-channel sample rate (some models divide max sample rate by the
       number of enabled channels).
    2. If termination_ohms is provided, commands the channel's input
       termination (50 ohm or 1 Mohm) to match the active probe before
       AutoSet runs, since termination changes the effective attenuation
       AutoSet's vertical-scale calculation must account for.
    3. Runs the scope's AutoSet routine, which sets the vertical scale (and an
       initial horizontal scale) automatically.
    4. Overrides the horizontal scale based on the capture scenario:
       - If required_max_frequency_hz is provided (a limit curve is applied),
         use the LARGEST time/div that still gives Nyquist coverage of that
         frequency.
       - Otherwise, use the SMALLEST time/div that the scope can actually
         deliver the full record_length at (verified via hardware readback),
         maximizing the number of points actually captured.

    `on_step`, if provided, is called with (event_name, info_dict) after each
    notable action, for diagnostics.

    Returns a dict describing the timebase that was requested.
    """
    _disable_unused_channels(scope, channel, on_step=on_step)

    if termination_ohms is not None:
        set_channel_termination(scope, termination_ohms, channel, on_step=on_step)

    scope.write("AutoSet EXEC")
    if on_step:
        try:
            on_step("command_sent", {"command": "AutoSet EXEC"})
        except Exception:
            pass
    time.sleep(5)
    if on_step:
        try:
            on_step("post_autoset", {"time_per_div": get_time_per_div(scope), "record_length": get_record_length(scope)})
        except Exception as e:
            on_step("post_autoset_query_failed", {"error": repr(e)})

    if required_max_frequency_hz:
        # Use the scope's true achievable record-length ceiling (which can be
        # well below `record_length`, e.g. 250,000 vs. a requested 1,000,000
        # on a Tektronix MSO24) in the Nyquist calculation below -- otherwise
        # the theoretical time/div ceiling is computed from a record length
        # the scope can never actually deliver, landing on a smaller time/div
        # than truly optimal (confirmed real-world symptom: computed 4 ms/div
        # instead of the correct 1 ms/div for a 10 MHz limit curve once the
        # real 250k-point ceiling is accounted for).
        achievable_record_length = _resolve_achievable_record_length(
            scope, record_length, safe_time_per_div=max_time_per_div, on_step=on_step
        )
        time_per_div = compute_time_per_div_for_max_frequency(
            required_max_frequency_hz, achievable_record_length, min_time_per_div, max_time_per_div
        )
        scenario = "limit_curve"
        configure_timebase(
            scope, time_per_div=time_per_div, record_length=achievable_record_length, on_step=on_step
        )
    else:
        # Verified against real hardware readback (HORIZONTAL:RECORDLENGTH?)
        # rather than a spec query, since ACQuire:SAMPLERATE? and
        # ACQuire:MAXSamplerate? have both proven unreliable on real
        # instruments (the former times out entirely on a Tektronix MSO24;
        # the latter can be stale/not reflect the current channel config).
        # This also leaves the scope already configured with the achievable
        # record length by the time it returns.
        time_per_div = _find_min_time_per_div_for_full_record(
            scope, record_length, min_time_per_div, max_time_per_div, on_step=on_step
        )
        scenario = "max_points"

    try:
        achieved_record_length = get_record_length(scope)
    except Exception:
        achieved_record_length = record_length

    return {
        "scenario": scenario,
        "time_per_div": time_per_div,
        "record_length": achieved_record_length,
        "required_max_frequency_hz": required_max_frequency_hz,
        "termination_ohms": termination_ohms,
    }


def download_waveform(scope, channel="CH1", num_points=10000):
    scope.write(f"DATA:SOURCE {channel}")
    scope.write("DATA:ENCdg RIBinary")
    scope.write("DATA:START 1")
    scope.write(f"DATA:STOP {num_points}")

    metadata = _read_waveform_metadata(scope)
    scope.write(f"DATA:WIDTH {metadata['byte_width']}")

    raw = scope.query_binary_values("CURVE?", datatype=metadata["datatype"], container=np.array)
    volts = (raw - metadata["yoff"]) * metadata["ymult"] + metadata["yzero"]
    return volts


def get_scope_data(scope, max_points=DEFAULT_MAX_POINTS):
    scope.write("DATA:SOURCE CH1")
    scope.write("DATA:ENCdg RIBinary")
    scope.write("DATA:START 1")
    n_points = _get_record_length(scope, max_points)
    n_points = max(1, min(int(n_points), int(max_points)))
    scope.write(f"DATA:STOP {n_points}")

    metadata = _read_waveform_metadata(scope)
    scope.write(f"DATA:WIDTH {metadata['byte_width']}")

    raw = scope.query_binary_values("CURVE?", datatype=metadata["datatype"], container=np.array)
    volts = (raw - metadata["yoff"]) * metadata["ymult"] + metadata["yzero"]
    dt = metadata["xincr"]
    return volts, dt
