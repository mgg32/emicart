import time
import os
import threading

import numpy as np
import pyvisa


DEFAULT_RESOURCE = "USB0::0x0699::0x0105::SGVJ011062::INSTR"
DEFAULT_BACKEND = "@py"
DEFAULT_OPEN_TIMEOUT_MS = 3000
DEFAULT_IO_TIMEOUT_MS = 5000
DEFAULT_MAX_POINTS = 1_000_000


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
        return False, ""
    return ("TEKTRONIX" in idn.upper()), idn


def connect_to_scope(
    resource_str=None,
    backend=DEFAULT_BACKEND,
    open_timeout_ms=DEFAULT_OPEN_TIMEOUT_MS,
    io_timeout_ms=DEFAULT_IO_TIMEOUT_MS,
):
    env_resource = os.environ.get("EMICART_SCOPE_RESOURCE", "").strip()
    preferred = resource_str or env_resource
    rm = pyvisa.ResourceManager(backend)
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
        print(f"Connected to: {idn}")
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
            scope.timeout = io_timeout_ms
            print(f"Connected to: {idn}")
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


def configure_timebase(scope, time_per_div, record_length):
    scope.write(f"HOR:MAIN:SCALE {time_per_div}")
    scope.write(f"HOR:RECO {record_length}")
    time.sleep(0.5)


def download_waveform(scope, channel="CH1", num_points=10000):
    scope.write(f"DATA:SOURCE {channel}")
    scope.write("DATA:ENCdg RIBinary")
    scope.write("DATA:WIDTH 1")
    scope.write("DATA:START 1")
    scope.write(f"DATA:STOP {num_points}")

    ymult = float(scope.query("WFMPRE:YMULT?"))
    yzero = float(scope.query("WFMPRE:YZERO?"))
    yoff = float(scope.query("WFMPRE:YOFF?"))

    raw = scope.query_binary_values("CURVE?", datatype="b", container=np.array)
    volts = (raw - yoff) * ymult + yzero
    return volts


def get_scope_data(scope, max_points=DEFAULT_MAX_POINTS):
    scope.write("DATA:SOURCE CH1")
    scope.write("DATA:ENCdg RIBinary")
    scope.write("DATA:WIDTH 1")
    scope.write("DATA:START 1")
    try:
        n_points = int(float(scope.query("WFMPRE:NR_PT?")))
    except Exception:
        n_points = max_points
    n_points = max(1, min(int(n_points), int(max_points)))
    scope.write(f"DATA:STOP {n_points}")

    ymult = float(scope.query("WFMPRE:YMULT?"))
    yzero = float(scope.query("WFMPRE:YZERO?"))
    yoff = float(scope.query("WFMPRE:YOFF?"))
    dt = float(scope.query("WFMPRE:XINCR?"))

    raw = scope.query_binary_values("CURVE?", datatype="b", container=np.array)
    volts = (raw - yoff) * ymult + yzero
    return volts, dt
