"""Run without the internet once the picture models are cached.

After the first successful start of the GPU backend, ``mark_ready`` writes a small marker
next to the engines listing the models in use. On later starts ``apply`` sees the marker and
sets ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``, so the model loaders read only the local
cache and never contact Hugging Face. Changing a model in config.yaml (or switching the safety
checker on) no longer matches the marker, so that start may download again.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("dreamer.offline")
MARKER = "models-ready.json"
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _marker(config) -> Path:
    return Path(config.model.engine_dir) / MARKER


def _models(config) -> dict:
    return {"base": config.model.base, "controlnet": config.model.controlnet,
            "safety_checker": bool(config.model.safety_checker)}


def apply(config) -> bool:
    """Switch the model loaders offline when this config's models are known to be cached.
    Must run before huggingface_hub is imported (it reads the setting once). Returns True
    when offline."""
    if any(os.environ.get(name) for name in OFFLINE_ENV):
        return True  # the user decided
    try:
        ready = json.loads(_marker(config).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if ready != _models(config):
        return False
    if "huggingface_hub" in sys.modules:
        log.warning("huggingface_hub was imported before the offline switch; it may go online")
    for name in OFFLINE_ENV:
        os.environ[name] = "1"
    log.info("Dreamer: models are cached, running offline (delete %s to allow downloads)",
             _marker(config))
    return True


def mark_ready(config) -> None:
    """Record that this config's models loaded successfully (called after GPU warmup)."""
    marker = _marker(config)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(_models(config), indent=1), encoding="utf-8")
    except OSError:
        log.exception("could not write %s", marker)
