import bisect
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class Curve:
    def __init__(
        self,
        name,
        breakpoints: List[Tuple[float, float]],
        slopes: List[float],
        units: str,
        resolution_bandwidth_hz: Optional[List[float]] = None,
    ):
        assert len(slopes) == len(breakpoints) - 1, "len(slopes) must be len(breakpoints)-1"
        assert all(
            breakpoints[i][0] < breakpoints[i + 1][0] for i in range(len(breakpoints) - 1)
        ), "breakpoints must be sorted by x (strictly increasing)"
        if resolution_bandwidth_hz is None:
            resolution_bandwidth_hz = [None] * len(slopes)
        assert len(resolution_bandwidth_hz) == len(slopes), (
            "len(resolution_bandwidth_hz) must match len(slopes)"
        )
        self.name = name
        self.breakpoints = list(breakpoints)
        self.slopes = list(slopes)
        self.units = units
        self.resolution_bandwidth_hz = list(resolution_bandwidth_hz)

    def get_curve(self, x_values: Union[float, Iterable[float]]) -> Union[Optional[float], List[Optional[float]]]:
        xs, ys = zip(*self.breakpoints)

        def eval_one(x: float) -> Optional[float]:
            if x < xs[0] or x > xs[-1]:
                return None

            idx = bisect.bisect_left(xs, x)
            if idx < len(xs) and xs[idx] == x:
                return ys[idx]

            i = idx - 1
            if i < 0 or i >= len(self.slopes):
                return None

            x1, y1 = xs[i], ys[i]
            slope = self.slopes[i]
            y = y1 + slope * math.log10(x / x1)
            return y

        if isinstance(x_values, (int, float)):
            return eval_one(float(x_values))
        return [eval_one(float(x)) for x in x_values]

    def get_resolution_bandwidth(
        self, x_values: Union[float, Iterable[float]]
    ) -> Union[Optional[float], List[Optional[float]]]:
        xs, _ = zip(*self.breakpoints)

        def eval_one(x: float) -> Optional[float]:
            if x < xs[0]:
                seg = 0
            elif x > xs[-1]:
                seg = len(self.slopes) - 1
            else:
                idx = bisect.bisect_left(xs, x)
                if idx == 0:
                    seg = 0
                elif idx >= len(xs):
                    seg = len(self.slopes) - 1
                elif xs[idx] == x:
                    seg = min(max(idx - 1, 0), len(self.slopes) - 1)
                else:
                    seg = idx - 1

            if seg < 0 or seg >= len(self.resolution_bandwidth_hz):
                return None
            return self.resolution_bandwidth_hz[seg]

        if isinstance(x_values, (int, float)):
            return eval_one(float(x_values))
        return [eval_one(float(x)) for x in x_values]


curve = Curve


def _calc_slopes_from_breakpoints(breakpoints: List[Tuple[float, float]]) -> List[float]:
    slopes: List[float] = []
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]
        if x1 <= 0 or x2 <= 0 or x2 <= x1:
            raise ValueError("Breakpoint frequencies must be positive and strictly increasing.")
        decades = math.log10(x2 / x1)
        if decades == 0:
            raise ValueError("Adjacent breakpoints cannot have identical frequencies.")
        slopes.append((y2 - y1) / decades)
    return slopes


def _get_limit_store_path() -> Path:
    base = Path.home() / "Documents" / "EmiCart"
    base.mkdir(parents=True, exist_ok=True)
    return base / "limits.json"


def _get_default_limit_template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "default_limits.json"


def _build_default_standard_registry() -> Dict[str, List[Curve]]:
    template_path = _get_default_limit_template_path()
    try:
        payload = json.loads(template_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load limit defaults from %s: %s", template_path, e)
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    registry: Dict[str, List[Curve]] = {}
    for standard_name, curves_raw in payload.items():
        std = str(standard_name).strip()
        if not std or not isinstance(curves_raw, list):
            logger.warning("Skipped invalid standard entry in default template: %r", standard_name)
            continue
        curves_for_std: List[Curve] = []
        invalid_curves = 0
        for raw_curve in curves_raw:
            c = _dict_to_curve(raw_curve)
            if c is not None:
                curves_for_std.append(c)
            else:
                invalid_curves += 1
        if invalid_curves:
            logger.warning(
                "Dropped %d invalid default curve entries for standard %s.",
                invalid_curves,
                std,
            )
        if curves_for_std:
            registry[std] = curves_for_std
    if not registry:
        logger.warning("Limit default template had no valid standards: %s", template_path)
    return registry


def _curve_to_dict(c: Curve) -> dict:
    return {
        "name": c.name,
        "units": c.units,
        "breakpoints": [[float(x), float(y)] for x, y in c.breakpoints],
        "resolution_bandwidth_hz": [
            None if rbw is None else float(rbw) for rbw in c.resolution_bandwidth_hz
        ],
    }


def _dict_to_curve(raw: dict) -> Optional[Curve]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    units = str(raw.get("units", "")).strip()
    breakpoints_raw = raw.get("breakpoints")
    rbw_raw = raw.get("resolution_bandwidth_hz")
    if not name or not units or not isinstance(breakpoints_raw, list) or len(breakpoints_raw) < 2:
        return None

    breakpoints: List[Tuple[float, float]] = []
    for row in breakpoints_raw:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None
        try:
            x = float(row[0])
            y = float(row[1])
        except (TypeError, ValueError):
            return None
        breakpoints.append((x, y))

    try:
        slopes = _calc_slopes_from_breakpoints(breakpoints)
    except ValueError:
        return None

    if rbw_raw is None:
        rbw = [None] * len(slopes)
    else:
        if not isinstance(rbw_raw, list) or len(rbw_raw) != len(slopes):
            return None
        rbw = []
        for v in rbw_raw:
            if v in ("", None):
                rbw.append(None)
            else:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return None
                rbw.append(fv if fv > 0 else None)

    try:
        return Curve(
            name=name,
            breakpoints=breakpoints,
            slopes=slopes,
            units=units,
            resolution_bandwidth_hz=rbw,
        )
    except AssertionError:
        return None


def _save_standard_registry(registry: Dict[str, List[Curve]]) -> None:
    path = _get_limit_store_path()
    payload = {
        standard: [_curve_to_dict(c) for c in curves]
        for standard, curves in sorted(registry.items(), key=lambda kv: kv[0])
        if curves
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_standard_registry() -> Dict[str, List[Curve]]:
    path = _get_limit_store_path()
    if not path.exists():
        seeded = _build_default_standard_registry()
        _save_standard_registry(seeded)
        return seeded
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse limit registry %s: %s", path, e)
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    registry: Dict[str, List[Curve]] = {}
    for standard_name, curves_raw in payload.items():
        std = str(standard_name).strip()
        if not std or not isinstance(curves_raw, list):
            logger.warning("Skipped invalid standard entry in %s: %r", path, standard_name)
            continue
        curves_for_std: List[Curve] = []
        invalid_curves = 0
        for raw_curve in curves_raw:
            c = _dict_to_curve(raw_curve)
            if c is not None:
                curves_for_std.append(c)
            else:
                invalid_curves += 1
        if invalid_curves:
            logger.warning(
                "Dropped %d invalid curve entries for standard %s in %s.",
                invalid_curves,
                std,
                path,
            )
        if curves_for_std:
            registry[std] = curves_for_std

    if not registry:
        registry = _build_default_standard_registry()
        _save_standard_registry(registry)
        return registry

    # Add newly introduced default standards/curves without overwriting user edits.
    defaults = _build_default_standard_registry()
    added = False
    for std, default_curves in defaults.items():
        if std not in registry:
            registry[std] = list(default_curves)
            added = True
            continue
        existing_names = {c.name for c in registry[std]}
        for c in default_curves:
            if c.name not in existing_names:
                registry[std].append(c)
                added = True
    if added:
        _save_standard_registry(registry)
    return registry


_standard_curves = _load_standard_registry()


def _refresh_runtime_cache() -> None:
    global standard_curves
    global curves

    standard_curves = {std: list(cs) for std, cs in _standard_curves.items()}
    curves = [c for curves_for_std in standard_curves.values() for c in curves_for_std]


def reload_standards() -> None:
    global _standard_curves
    _standard_curves = _load_standard_registry()
    _refresh_runtime_cache()


def get_standards():
    return list(standard_curves.keys())


def get_curves_for_standard(standard: str):
    return standard_curves.get(standard, [])


def get_curve_by_name(name: str, standard: Optional[str] = None) -> Optional[Curve]:
    search_pool = curves if standard is None else get_curves_for_standard(standard)
    return next((c for c in search_pool if c.name == name), None)


def upsert_curve(
    standard: str,
    curve_name: str,
    units: str,
    breakpoints: List[Tuple[float, float]],
    resolution_bandwidth_hz: Optional[List[Optional[float]]] = None,
) -> Curve:
    std = (standard or "").strip()
    name = (curve_name or "").strip()
    units_str = (units or "").strip()
    if not std:
        raise ValueError("Standard name is required.")
    if not name:
        raise ValueError("Curve name is required.")
    if not units_str:
        raise ValueError("Units are required.")
    if len(breakpoints) < 2:
        raise ValueError("At least two breakpoints are required.")

    slopes = _calc_slopes_from_breakpoints(breakpoints)
    if resolution_bandwidth_hz is None:
        rbw = [None] * len(slopes)
    else:
        if len(resolution_bandwidth_hz) != len(slopes):
            raise ValueError("RBW count must be exactly one less than breakpoint count.")
        rbw = []
        for v in resolution_bandwidth_hz:
            if v in ("", None):
                rbw.append(None)
            else:
                fv = float(v)
                rbw.append(fv if fv > 0 else None)

    updated_curve = Curve(
        name=name,
        breakpoints=breakpoints,
        slopes=slopes,
        units=units_str,
        resolution_bandwidth_hz=rbw,
    )

    std_curves = _standard_curves.setdefault(std, [])
    idx = next((i for i, c in enumerate(std_curves) if c.name == name), None)
    if idx is None:
        std_curves.append(updated_curve)
    else:
        std_curves[idx] = updated_curve

    _save_standard_registry(_standard_curves)
    _refresh_runtime_cache()
    return updated_curve


def delete_curve(standard: str, curve_name: str) -> bool:
    std = (standard or "").strip()
    name = (curve_name or "").strip()
    if not std or not name:
        return False
    curves_for_std = _standard_curves.get(std, [])
    new_curves = [c for c in curves_for_std if c.name != name]
    if len(new_curves) == len(curves_for_std):
        return False
    if new_curves:
        _standard_curves[std] = new_curves
    else:
        _standard_curves.pop(std, None)
    _save_standard_registry(_standard_curves)
    _refresh_runtime_cache()
    return True


def delete_standard(standard: str) -> bool:
    std = (standard or "").strip()
    if not std:
        return False
    if std not in _standard_curves:
        return False
    del _standard_curves[std]
    _save_standard_registry(_standard_curves)
    _refresh_runtime_cache()
    return True


_refresh_runtime_cache()
