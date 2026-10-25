"""Configuration: a YAML file mapped onto dataclasses. Every key is optional."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("dreamer.config")

DEFAULT_PIN = "1234"
VIEW_MODES = ("auto", "sketch", "picture", "split")
KID_THEMES = ("dark", "light")
LINE_POLARITIES = ("white_on_black", "black_on_white")
BACKENDS = ("streamdiffusion", "mock")
CONFIG_ENV = "SKETCH_CONFIG"
PIN_RE = re.compile(r"\d{4,12}")  # what the admin PIN pad can type
ORIENTATIONS = ("landscape", "portrait")
# Drawings have the iPad's shape: the short side is this fraction of ``size`` (rounded to
# a multiple of 64, which the UNet needs), so 640 gives 640x448 (iPad: about 1.43:1).
SHORT_SIDE_RATIO = 0.7


@dataclass
class ModelConfig:
    base: str = "stabilityai/sd-turbo"
    t_index_list: list[int] = field(default_factory=lambda: [6, 26])
    # Most denoising steps the admin page may pick.
    max_steps: int = 8
    # Up to this many steps run together as one UNet batch (fast, one frame of lag per extra
    # step); more steps run one after another at batch 1. The TensorRT UNet is built for
    # batch 1..max_batch_steps, so this sets its memory; changing it rebuilds it once.
    max_batch_steps: int = 4
    # Once the child stops drawing, the picture keeps refining with more steps (run one
    # after another) up to this many; 0 turns refinement off. Live-changeable (1..20).
    refine_steps: int = 8
    # AI upscaler for refined pictures (never while drawing): "off" or "x2".
    upscaler: str = "off"
    upscaler_model: str = "models/realesr-animevideov3.pth"
    controlnet: str | None = "thibaud/controlnet-sd21-scribble-diffusers"  # None disables
    controlnet_scale: float = 0.55
    line_polarity: str = "white_on_black"
    engine_dir: str = "engines"
    safety_checker: bool = True
    safety_threshold: float = 0.5


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 7860
    backend: str = "streamdiffusion"
    admin_pin: str = DEFAULT_PIN
    secret_key: str = ""
    data_dir: str = "data"
    jpeg_quality: int = 85
    size: int = 640  # long side; see canvas_dims()
    default_view_mode: str = "auto"
    kid_theme: str = "dark"  # the drawing page's look: dark | light (switchable on /admin)
    mock_delay_ms: int = 30  # simulated GPU time for the mock backend
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


def _apply(target: Any, values: dict[str, Any], where: str) -> None:
    """Copy known keys from ``values`` onto a dataclass instance, coercing scalars."""
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            log.warning("config: ignoring unknown key %s%s", where, key)
            continue
        current = getattr(target, key)
        if key == "model":
            if not isinstance(value, dict):
                raise ValueError("config: 'model' must be a mapping")
            _apply(current, value, "model.")
            continue
        if value is None:
            if key == "controlnet":
                setattr(target, key, None)
            continue
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError(f"config: {where}{key} must be true/false")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list):
            if not isinstance(value, list):
                raise ValueError(f"config: {where}{key} must be a list")
            value = [int(v) for v in value]
        elif isinstance(current, str) or current is None:
            value = str(value)
        setattr(target, key, value)


def _validate(cfg: Config) -> None:
    cfg.backend = cfg.backend.lower()
    if cfg.backend not in BACKENDS:
        raise ValueError(f"config: backend must be one of {BACKENDS}")
    if cfg.default_view_mode not in VIEW_MODES:
        raise ValueError(f"config: default_view_mode must be one of {VIEW_MODES}")
    if cfg.kid_theme not in KID_THEMES:
        raise ValueError(f"config: kid_theme must be one of {KID_THEMES}")
    if cfg.model.line_polarity not in LINE_POLARITIES:
        raise ValueError(f"config: model.line_polarity must be one of {LINE_POLARITIES}")
    if not 1 <= cfg.jpeg_quality <= 100:
        raise ValueError("config: jpeg_quality must be 1..100")
    if not 64 <= cfg.size <= 2048 or cfg.size % 64:
        raise ValueError("config: size must be a multiple of 64 in 64..2048")
    if not PIN_RE.fullmatch(cfg.admin_pin):
        raise ValueError("config: admin_pin must be 4 to 12 digits (the PIN pad only has "
                         'digits), written in quotes, e.g. admin_pin: "0420"')


def _raw_pin(text: str, parsed: Any) -> Any:
    """Recover an unquoted numeric PIN exactly as typed.

    YAML turns ``admin_pin: 0420`` into the octal number 272 and drops leading zeros, so
    when the parsed value is an int, take the digits from the file text instead.
    """
    if not isinstance(parsed, int) or isinstance(parsed, bool):
        return parsed
    match = re.search(r"^admin_pin:[ \t]*(\d+)[ \t]*(?:#.*)?$", text, re.MULTILINE)
    raw = match.group(1) if match else str(parsed)
    log.warning("config: admin_pin should be in quotes (admin_pin: \"%s\"); "
                "unquoted numbers can lose leading zeros", raw)
    return raw


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config from ``path``, else $SKETCH_CONFIG, else ./config.yaml.

    A missing file yields defaults. Invalid values raise ValueError.
    """
    if path is None:
        path = os.environ.get(CONFIG_ENV) or "config.yaml"
    cfg = Config()
    p = Path(path)
    if p.is_file():
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config: {p} must contain a mapping")
        if "admin_pin" in data:
            data["admin_pin"] = _raw_pin(text, data["admin_pin"])
        _apply(cfg, data, "")
        log.info("loaded config from %s", p)
    else:
        log.info("no config file at %s; using defaults", p)
    _validate(cfg)
    return cfg


def canvas_dims(cfg: Any, long_side: int | None = None) -> dict[str, tuple[int, int]]:
    """(width, height) of a drawing and its picture per orientation; ``long_side``
    overrides ``cfg.size`` (the live "resolution" param)."""
    long_side = int(long_side or cfg.size)
    short_side = max(64, round(long_side * SHORT_SIDE_RATIO / 64) * 64)
    return {"landscape": (long_side, short_side), "portrait": (short_side, long_side)}
