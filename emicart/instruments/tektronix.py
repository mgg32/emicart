import time
import os
import threading

import numpy as np
import pyvisa


DEFAULT_RESOURCE = None
DEFAULT_BACKEND = "@py"
DEFAULT_BACKEND_ENV = "EMICART_VISA_BACKEND"
DEFAULT_OPEN_TIMEOUT_MS = 3000
DEFAULT_IO_TIMEOUT_MS = 5000
DEFAULT_MAX_POINTS = 1_000_000

WAVEFORM_METADATA_ATTR = "_emicart_waveform_metadata"


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
    metadata = getattr(scope, WAVEFORM_METADATA_ATTR, None)
    if metadata is not None:
        return metadata

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

    metadata = {
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
    setattr(scope, WAVEFORM_METADATA_ATTR, metadata)
    return metadata


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


def configure_timebase(scope, time_per_div, record_length):
    _write_first(
        scope,
        [
            f"HORIZONTAL:SCALE {time_per_div}",
            f"HOR:MAIN:SCALE {time_per_div}",
        ],
    )
    _write_first(
        scope,
        [
            f"HORIZONTAL:RECORDLENGTH {record_length}",
            f"HOR:RECO {record_length}",
        ],
    )
    time.sleep(0.5)


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
