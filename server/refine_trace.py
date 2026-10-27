"""Refinement traces: every picture of one refinement, kept in memory for the admin page.

When the drawing pauses, the worker refines the picture step by step (server/refine.py).
A trace records that run so the owner can see which step changes the picture too much:

- step 0 is the picture on screen when the child lifted the brush (the drawing picture);
- steps 1..N are the refined pictures, as sent (after upscaling, if on).

Each step gets two change scores, the metrics of tools/refine_lab.py measured on 1/16-size
copies (cheap, on the CPU): ``step`` is the mean absolute difference (0-255) to the previous
step, ``drift`` the one to step 0. A step over JUMP_STEP / JUMP_DRIFT is flagged as a jump.

Traces live only in memory (the latest TRACE_RING). ``RefineTraces.save`` writes one to
``data/refine-batches/<timestamp>-<model>/`` on demand: the sketch as received, every step's
JPEG and a settings.json with what tools/replay_batch.py needs to run it again.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .fsutil import replace_with_retry

log = logging.getLogger("dreamer.refine_trace")

TRACE_RING = 5
BATCHES_DIR = "refine-batches"
SKETCH_FILE = "sketch.png"
SETTINGS_FILE = "settings.json"
SETTINGS_VERSION = 1
# A step whose change jumps over these (0-255, at 1/16 size) is highlighted on the admin page.
JUMP_STEP = 6.0
JUMP_DRIFT = 6.0
SMALL_FACTOR = 16
TRACE_ID_RE = re.compile(r"[0-9a-f]{12}")
REPO_ROOT = Path(__file__).resolve().parent.parent
T_INDEX_MAX = 49


def valid_trace_id(value: Any) -> bool:
    return isinstance(value, str) and TRACE_ID_RE.fullmatch(value) is not None


def step_file(index: int, prefix: str = "step") -> str:
    return f"{prefix}-{index:02d}.jpg"


def small(picture: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    """``picture`` at 1/16 size (box filter), as tools/refine_lab.py measures it. ``size``:
    the small size to use instead, so a bigger (upscaled) step compares with step 0."""
    if size is None:
        size = (max(1, picture.width // SMALL_FACTOR), max(1, picture.height // SMALL_FACTOR))
    return np.asarray(picture.convert("RGB").resize(size, Image.Resampling.BOX), dtype=np.float32)


def mean_change(first: np.ndarray, second: np.ndarray) -> float:
    return round(float(np.abs(first - second).mean()), 1)


def is_jump(step_change: float | None, drift: float) -> bool:
    return (step_change is not None and step_change > JUMP_STEP) or drift > JUMP_DRIFT


def creativity_pct(t_index_list: list[int]) -> int:
    """The admin page's Creativity slider for a t_index_list (static/admin.js creativityOf)."""
    return round((1 - t_index_list[0] / T_INDEX_MAX) * 100) if t_index_list else 0


@functools.lru_cache(maxsize=1)
def git_commit() -> str | None:
    """The Dreamer checkout's commit (once per process), or None without git."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


@dataclass
class TraceStep:
    index: int
    jpeg: bytes
    ms: float            # after the refinement started (the drawing had paused)
    step_change: float | None  # vs the previous step; None for step 0
    drift: float         # vs step 0
    width: int
    height: int

    def meta(self) -> dict[str, Any]:
        return {"index": self.index, "ms": round(self.ms, 1), "step": self.step_change,
                "drift": self.drift, "jump": is_jump(self.step_change, self.drift),
                "width": self.width, "height": self.height}


@dataclass
class RefineTrace:
    """One refinement run. Made and filled by the worker thread through RefineTraces."""

    seq: int
    model: str           # the refinement model (server/refine.py)
    sketch_png: bytes
    prompt: str
    background: str | None
    orientation: str | None
    dims: tuple[int, int]
    backend: str
    params: dict[str, Any]
    upscales_itself: bool = False
    id: str = field(default_factory=lambda: secrets.token_hex(6))
    started: float = field(default_factory=time.perf_counter)
    created: datetime = field(default_factory=datetime.now)
    steps: list[TraceStep] = field(default_factory=list)
    done: bool = False
    stopped: bool = False  # cut short (a new stroke, a reset) before the last step
    error: str | None = None
    saved_as: str | None = None
    saved_steps: int = 0
    # Small copies for the scores (worker thread only).
    _base_small: np.ndarray | None = None
    _last_small: np.ndarray | None = None

    def meta(self) -> dict[str, Any]:
        """Everything about the trace but the image bytes (JSON for the admin page)."""
        steps = [step.meta() for step in self.steps]
        return {"id": self.id, "seq": self.seq, "model": self.model,
                "created_at": self.created.isoformat(timespec="seconds"),
                "backend": self.backend, "width": self.dims[0], "height": self.dims[1],
                "orientation": self.orientation or "landscape", "background": self.background,
                "refine_steps": self.params.get("refine_steps"),
                "upscale": self.params.get("upscale"),
                "done": self.done, "stopped": self.stopped, "error": self.error,
                "saved_as": self.saved_as, "steps": steps,
                "max_step": max((step["step"] for step in steps if step["step"] is not None), default=None),
                "max_drift": max((step["drift"] for step in steps), default=None),
                "jumps": sum(step["jump"] for step in steps)}

    def settings(self) -> dict[str, Any]:
        """settings.json: the exact inputs to replay this run (tools/replay_batch.py)."""
        meta = self.meta()
        return {
            "version": SETTINGS_VERSION,
            "trace_id": self.id,
            "created_at": meta["created_at"],
            "seq": self.seq,
            "refine_model": self.model,
            "params": self.params,
            "creativity_pct": creativity_pct(self.params.get("t_index_list") or []),
            "prompt": self.prompt,
            "background": self.background,
            "orientation": self.orientation or "landscape",
            "layered": self.background is not None,
            "picture_size": list(self.dims),
            "backend": self.backend,
            "refiner_upscales_itself": self.upscales_itself,
            "git_commit": git_commit(),
            "sketch": SKETCH_FILE,
            "complete": self.done and not self.stopped and self.error is None,
            "stopped": self.stopped,
            "error": self.error,
            "score_note": ("step = mean absolute difference (0-255) to the previous step, drift = "
                           "to step 0, both at 1/16 size (tools/refine_lab.py)"),
            "jump_thresholds": {"step": JUMP_STEP, "drift": JUMP_DRIFT},
            "steps": [{**step, "file": step_file(step["index"])} for step in meta["steps"]],
        }


class RefineTraces:
    """The latest traces (a ring of TRACE_RING), shared by the worker and the app."""

    def __init__(self, size: int = TRACE_RING) -> None:
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()  # one batch folder written at a time
        self._ring: deque[RefineTrace] = deque(maxlen=max(1, size))

    # ----- worker thread -------------------------------------------------------------

    def begin(self, trace: RefineTrace, first: Image.Image, first_jpeg: bytes) -> RefineTrace:
        """Start recording ``trace`` with step 0 (the picture on screen)."""
        trace._base_small = small(first)
        trace._last_small = trace._base_small
        step = TraceStep(0, first_jpeg, 0.0, None, 0.0, first.width, first.height)
        with self._lock:
            trace.steps.append(step)
            self._ring.append(trace)
        return trace

    def add_step(self, trace: RefineTrace, picture: Image.Image, jpeg: bytes) -> TraceStep:
        """Record a refined picture as sent; scores are computed here, outside the lock."""
        ms = (time.perf_counter() - trace.started) * 1000.0
        base = trace._base_small
        current = small(picture, (base.shape[1], base.shape[0])) if base is not None else small(picture)
        step_change = mean_change(current, trace._last_small) if trace._last_small is not None else None
        drift = mean_change(current, base) if base is not None else 0.0
        trace._last_small = current
        with self._lock:
            step = TraceStep(len(trace.steps), jpeg, ms, step_change, drift,
                             picture.width, picture.height)
            trace.steps.append(step)
        return step

    def finish(self, trace: RefineTrace, stopped: bool = False, error: str | None = None) -> None:
        with self._lock:
            trace.done, trace.stopped, trace.error = True, stopped, error
        trace._base_small = trace._last_small = None

    # ----- app (any thread) ----------------------------------------------------------

    def meta(self, trace: RefineTrace) -> dict[str, Any]:
        with self._lock:
            return trace.meta()

    def list(self) -> list[dict[str, Any]]:
        """Metadata of the traces in memory, newest first."""
        with self._lock:
            return [trace.meta() for trace in reversed(self._ring)]

    def get(self, trace_id: str) -> RefineTrace | None:
        if not valid_trace_id(trace_id):
            return None
        with self._lock:
            return next((trace for trace in self._ring if trace.id == trace_id), None)

    def step_jpeg(self, trace_id: str, index: int) -> bytes | None:
        trace = self.get(trace_id)
        if trace is None:
            return None
        with self._lock:
            return trace.steps[index].jpeg if 0 <= index < len(trace.steps) else None

    def save(self, trace_id: str, data_dir: str | Path) -> str:
        """Write the trace as a batch folder; returns its path relative to ``data_dir``.
        Saving the same trace again (with no new steps) returns the folder it went to.
        Raises KeyError for an unknown trace, OSError if the files cannot be written."""
        trace = self.get(trace_id)
        if trace is None:
            raise KeyError(trace_id)
        git_commit()  # runs git once, here: never while holding the lock the worker needs
        with self._save_lock:
            return self._save(trace, Path(data_dir))

    def _save(self, trace: RefineTrace, data_dir: Path) -> str:
        with self._lock:
            steps = list(trace.steps)
            settings = trace.settings()
            previous, previous_steps = trace.saved_as, trace.saved_steps
        root = data_dir / BATCHES_DIR
        if previous is not None and previous_steps == len(steps) and (data_dir / previous).is_dir():
            return previous
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{trace.created.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^a-z0-9_]', '', trace.model) or 'refine'}"
        name = stem
        for attempt in range(2, 100):
            if not (root / name).exists():
                break
            name = f"{stem}-{attempt}"
        temp = root / f".tmp-{trace.id}-{secrets.token_hex(3)}"
        try:
            temp.mkdir()
            (temp / SKETCH_FILE).write_bytes(trace.sketch_png)
            for step in steps:
                (temp / step_file(step.index)).write_bytes(step.jpeg)
            (temp / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")
            replace_with_retry(temp, root / name)
        except OSError:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        relative = f"{BATCHES_DIR}/{name}"
        with self._lock:
            trace.saved_as, trace.saved_steps = relative, len(steps)
        log.info("saved refinement batch %s (%d steps)", relative, len(steps))
        return relative
