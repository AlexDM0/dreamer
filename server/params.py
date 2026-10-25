"""Live generation params (SPEC-addendum-params): GenParams, validation, persistence.

``GenParams`` holds every knob the admin page can turn without a restart. The store
persists it in ``data/params.json`` (atomic write) and validates partial updates:
wrong types raise ``ValueError`` (HTTP 400), numbers outside their range are clamped.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

log = logging.getLogger("dreamer.params")

PARAMS_FILE = "params.json"
SEED_MAX = 2**31 - 1
T_INDEX_MAX = 49
T_INDEX_MAX_STEPS = 8
POLARITIES = ("white_on_black", "black_on_white")
INIT_MODES = ("colors", "sketch", "blank")
UPSCALERS = ("off", "x2")
RESOLUTIONS = (512, 640, 768, 896, 1024)  # long sides offered on the admin page
REFINE_MAX_STEPS = 20

# Numeric ranges (min, max, step) per field, also served to the admin UI.
LIMITS: dict[str, dict[str, Any]] = {
    "t_index_list": {"min": 0, "max": T_INDEX_MAX, "step": 1,
                     "min_len": 1, "max_len": T_INDEX_MAX_STEPS},
    "seed": {"min": 0, "max": SEED_MAX, "step": 1},
    "controlnet_scale": {"min": 0.0, "max": 2.0, "step": 0.05},
    "safety_threshold": {"min": 0.05, "max": 0.95, "step": 0.05},
    "line_threshold": {"min": 10, "max": 120, "step": 1},
    "line_dilate": {"min": 0, "max": 3, "step": 1},
    "jpeg_quality": {"min": 50, "max": 95, "step": 1},
    "line_polarity": {"options": list(POLARITIES)},
    "init_mode": {"options": list(INIT_MODES)},
    "init_blur": {"min": 4, "max": 64, "step": 2},
    "refine_steps": {"min": 0, "max": REFINE_MAX_STEPS, "step": 1},
    "upscale": {"options": list(UPSCALERS)},
    # Long side; a multiple of 64 (UNet). The TensorRT engines handle 384..1024.
    "resolution": {"min": 64, "max": 1024, "step": 64, "options": list(RESOLUTIONS)},
}


@dataclass(frozen=True)
class GenParams:
    t_index_list: tuple[int, ...] = (6, 26)
    seed: int = 2
    controlnet_enabled: bool = True
    controlnet_scale: float = 0.55
    safety_checker: bool = True
    safety_threshold: float = 0.5
    line_threshold: int = 40
    line_polarity: str = "white_on_black"
    line_dilate: int = 1
    # What img2img starts from: "colors" (soft colour clouds, no strokes), "sketch", "blank".
    init_mode: str = "colors"
    init_blur: int = 24
    jpeg_quality: int = 85
    # After drawing pauses: refine up to this many steps (0 = off), upscaling if set.
    refine_steps: int = 8
    upscale: str = "off"
    # Long side of pictures in pixels (the short side follows the iPad's shape).
    resolution: int = 640

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["t_index_list"] = list(self.t_index_list)
        return out


_FIELDS = {f.name for f in fields(GenParams)}


def _clamp(value: float, name: str) -> float:
    lim = LIMITS[name]
    return min(max(value, lim["min"]), lim["max"])


def _as_int(value: Any, name: str) -> int:
    # bool is an int subclass; reject it explicitly. Integral floats (e.g. 3.0) are fine.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _t_index_list(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("t_index_list must be a list of numbers")
    lim = LIMITS["t_index_list"]
    if not lim["min_len"] <= len(value) <= lim["max_len"]:
        raise ValueError(f"t_index_list needs {lim['min_len']} to {lim['max_len']} entries")
    out = tuple(int(_clamp(_as_int(v, "t_index_list"), "t_index_list")) for v in value)
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError("t_index_list must be strictly increasing")
    return out


def apply_changes(base: GenParams, changes: Any) -> GenParams:
    """Return ``base`` with a partial update applied (validated and clamped).

    Raises ValueError for unknown fields or wrong types.
    """
    if not isinstance(changes, dict):
        raise ValueError("expected a JSON object")
    unknown = sorted(set(changes) - _FIELDS)
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
    out: dict[str, Any] = {}
    for name, value in changes.items():
        if name == "t_index_list":
            out[name] = _t_index_list(value)
        elif name in ("seed", "line_threshold", "line_dilate", "init_blur", "jpeg_quality",
                      "refine_steps"):
            out[name] = int(_clamp(_as_int(value, name), name))
        elif name in ("controlnet_scale", "safety_threshold"):
            out[name] = round(_clamp(_as_float(value, name), name), 4)
        elif name in ("controlnet_enabled", "safety_checker"):
            out[name] = _as_bool(value, name)
        elif name == "line_polarity":
            if value not in POLARITIES:
                raise ValueError(f"line_polarity must be one of {POLARITIES}")
            out[name] = value
        elif name == "init_mode":
            if value not in INIT_MODES:
                raise ValueError(f"init_mode must be one of {INIT_MODES}")
            out[name] = value
        elif name == "upscale":
            if value not in UPSCALERS:
                raise ValueError(f"upscale must be one of {UPSCALERS}")
            out[name] = value
        elif name == "resolution":
            side = _as_int(value, name)
            if side % 64 or not LIMITS[name]["min"] <= side <= LIMITS[name]["max"]:
                raise ValueError("resolution must be a multiple of 64 from 64 to 1024")
            out[name] = side
    return replace(base, **out)


def defaults_from_config(config: Any) -> GenParams:
    """Defaults come from config.model (and config.jpeg_quality), clamped to the limits."""
    model = config.model
    base = GenParams()
    try:
        t_index = _t_index_list(list(model.t_index_list))
    except ValueError:
        log.warning("config model.t_index_list %r is outside the live-control limits; "
                    "using %r", model.t_index_list, list(base.t_index_list))
        t_index = base.t_index_list
    return replace(
        base,
        t_index_list=t_index,
        controlnet_enabled=model.controlnet is not None,
        controlnet_scale=round(_clamp(float(model.controlnet_scale), "controlnet_scale"), 4),
        safety_checker=bool(model.safety_checker),
        safety_threshold=round(_clamp(float(model.safety_threshold), "safety_threshold"), 4),
        line_polarity=model.line_polarity if model.line_polarity in POLARITIES else base.line_polarity,
        jpeg_quality=int(_clamp(int(config.jpeg_quality), "jpeg_quality")),
        refine_steps=int(_clamp(int(getattr(model, "refine_steps", base.refine_steps)), "refine_steps")),
        upscale=getattr(model, "upscaler", "off") if getattr(model, "upscaler", "off") in UPSCALERS else "off",
        resolution=int(config.size),
    )


class ParamsStore:
    """Thread-safe current params, persisted to ``data_dir/params.json``."""

    def __init__(self, data_dir: str | os.PathLike, defaults: GenParams) -> None:
        self._path = Path(data_dir) / PARAMS_FILE
        self._lock = threading.Lock()
        self.defaults = defaults
        self._params = self._load()

    def _load(self) -> GenParams:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self.defaults
        except (OSError, ValueError) as exc:
            log.error("could not read %s (%s); using defaults", self._path, exc)
            return self.defaults
        if not isinstance(data, dict):
            return self.defaults
        params = self.defaults
        for name, value in data.items():  # field by field, so one bad value costs one field
            try:
                params = apply_changes(params, {name: value})
            except ValueError as exc:
                log.warning("params.json: ignoring %s (%s)", name, exc)
        return params

    def _save(self, params: GenParams) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(params.to_dict(), fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path)

    def get(self) -> GenParams:
        with self._lock:
            return self._params

    def set(self, params: GenParams) -> GenParams:
        """Persist and adopt ``params`` (OSError propagates; nothing changes then)."""
        with self._lock:
            self._save(params)
            self._params = params
            return params

    def patch(self, changes: Any) -> GenParams:
        with self._lock:
            params = apply_changes(self._params, changes)
            self._save(params)
            self._params = params
            return params
