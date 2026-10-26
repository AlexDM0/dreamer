"""Run without the internet once the picture models are cached.

After the first successful start of the GPU backend, ``mark_ready`` writes a small marker
next to the engines listing the models in use. On later starts ``apply`` sees the marker and
sets ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``, so the model loaders read only the local
cache and never contact Hugging Face. Changing a model in config.yaml (or switching the safety
checker on) no longer matches the marker, so that start may download again.

The SD 1.5 + TCD picture model is recorded in the same marker under "tcd" once it has
loaded. While Dreamer runs offline and TCD was never loaded, ``fetch_online`` downloads its
models in a separate process that is allowed online (the offline switch is read once per
process, so this one cannot turn it off again).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("dreamer.offline")
MARKER = "models-ready.json"
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _marker(config) -> Path:
    return Path(config.model.engine_dir) / MARKER


def _models(config) -> dict:
    return {"base": config.model.base, "controlnet": config.model.controlnet,
            "safety_checker": bool(config.model.safety_checker)}


def tcd_models(config) -> dict:
    """What the TCD picture model downloads (server/backends/tcd.py)."""
    model = config.model
    return {"base": model.tcd_base, "lora": model.tcd_lora, "lora_file": model.tcd_lora_file,
            "controlnet": model.tcd_controlnet}


def _read(config) -> dict[str, Any]:
    try:
        data = json.loads(_marker(config).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def is_offline() -> bool:
    return any(os.environ.get(name) for name in OFFLINE_ENV)


def tcd_ready(config) -> bool:
    """Whether the TCD models have loaded on this PC before (so they are in the cache)."""
    return _read(config).get("tcd") == tcd_models(config)


def apply(config) -> bool:
    """Switch the model loaders offline when this config's models are known to be cached.
    Must run before huggingface_hub is imported (it reads the setting once). Returns True
    when offline."""
    if any(os.environ.get(name) for name in OFFLINE_ENV):
        return True  # the user decided
    ready = _read(config)
    if {key: ready.get(key) for key in _models(config)} != _models(config):
        return False
    if "huggingface_hub" in sys.modules:
        log.warning("huggingface_hub was imported before the offline switch; it may go online")
    for name in OFFLINE_ENV:
        os.environ[name] = "1"
    log.info("Dreamer: models are cached, running offline (delete %s to allow downloads)",
             _marker(config))
    return True


def mark_ready(config, kind: str = "turbo") -> None:
    """Record that this config's models for picture model ``kind`` ("turbo" or "tcd") loaded
    successfully (called after GPU warmup). The other model's entry is kept."""
    marker = _marker(config)
    data = _read(config)
    if kind == "tcd":
        data["tcd"] = tcd_models(config)
    else:
        tcd = data.get("tcd")
        data = {**_models(config), **({"tcd": tcd} if tcd is not None else {})}
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        log.exception("could not write %s", marker)


# Run in a child process with the offline switch off: loads (and so downloads) exactly what
# TcdBackend.warmup() loads, on the CPU, then exits. argv: base, lora, lora_file, controlnet,
# safety model.
_FETCH_SCRIPT = """
import sys, torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification
base, lora, lora_file, controlnet, safety = sys.argv[1:6]
cn = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base, controlnet=cn, torch_dtype=torch.float16, safety_checker=None,
    requires_safety_checker=False)
hf_hub_download(lora, lora_file)
AutoModelForImageClassification.from_pretrained(safety, torch_dtype=torch.float16)
"""


def fetch_online(config, safety_model: str, timeout: float = 3600.0) -> None:
    """Download the TCD models while this process runs offline (see the module doc).
    Raises RuntimeError with a readable message if that fails (e.g. no internet)."""
    models = tcd_models(config)
    env = {key: value for key, value in os.environ.items() if key not in OFFLINE_ENV}
    log.info("Dreamer: downloading SD 1.5 + TCD once (several GB; this can take a while)")
    try:
        result = subprocess.run(
            [sys.executable, "-c", _FETCH_SCRIPT, models["base"], models["lora"],
             models["lora_file"], models["controlnet"], safety_model],
            env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not download SD 1.5 + TCD: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        raise RuntimeError("SD 1.5 + TCD needs the internet once to download its models, and "
                           f"the download failed ({tail[0]})")
