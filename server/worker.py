"""GpuWorker: the one thread that touches the diffusion backend (and CUDA).

It takes the newest sketch from the single-slot Mailbox, turns it into a picture and
hands the JPEG to ``on_result``. Admin upload screening jobs go through a separate
queue so "latest frame wins" never drops them. The worker never touches sockets.
"""
from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from . import imaging
from .config import canvas_dims
from .mailbox import Mailbox
from .params import T_INDEX_MAX, GenParams, defaults_from_config

log = logging.getLogger("dreamer.worker")

MAX_CONSECUTIVE_FAILURES = 3
TAKE_TIMEOUT = 0.5
# A backend with pipeline latency (StreamDiffusion with 2+ steps returns the picture for
# an earlier frame) is re-run on the last sketch once no new sketch came for this long.
SETTLE_DELAY = 0.15
# Refinement after drawing pauses: each round repaints the PREVIOUS round's picture with less
# noise (a later first t_index), so the picture keeps its composition and gets cleaner instead
# of being repainted from scratch. First t_index of the first and the last round; rounds in
# between are evenly spaced. params.refine_steps is the step budget over at most 5 rounds.
REFINE_FIRST_START = 18
REFINE_LAST_START = 44
REFINE_MAX_ROUNDS = 5
REFINE_STEPS_PER_ROUND = 4
REFINE_POLL = 0.005  # between refinement passes: just a check for a new stroke
FPS_EMA_ALPHA = 0.2
FPS_IDLE_GAP = 2.0  # seconds


def refine_t_index(first: int, steps: int) -> tuple[int, ...]:
    """``steps`` t_index values from ``first`` (the creativity) to the last one, evenly
    spread; fewer if there are not that many distinct values left."""
    steps = max(1, min(steps, T_INDEX_MAX - first + 1))
    if steps == 1:
        return (first,)
    return tuple(round(first + i * (T_INDEX_MAX - first) / (steps - 1)) for i in range(steps))


@dataclass
class FrameJob:
    seq: int
    png_bytes: bytes
    prompt: str
    # Opaque tag of the socket that sent the sketch; handed back to ``on_result`` so a
    # picture finished after that socket was replaced is not sent to its successor.
    owner: Any = None
    # The page's background layer colour; set when the PNG is a stroke layer
    # (transparent background). None: an opaque sketch on white (older pages).
    background: str | None = None
    # "landscape" / "portrait": the drawing's shape (config.canvas_dims). None: landscape.
    orientation: str | None = None


@dataclass
class ScreenUploadJob:
    image: Image.Image
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)


class GpuWorker(threading.Thread):
    """Daemon thread running warmup, then the frame loop until ``stop()``.

    ``backend`` may be None when ``backend_factory`` is given: the backend is then
    built on this thread, so model loading also happens on the GPU thread.
    ``on_result(seq, jpeg_bytes, owner)`` (owner = ``FrameJob.owner``) and
    ``on_status(state, msg)`` must be thread-safe.
    """

    def __init__(self, backend: Any, mailbox: Mailbox,
                 on_result: Callable[[int, bytes, Any], None],
                 on_status: Callable[[str, str], None],
                 config: Any,
                 backend_factory: Callable[[], Any] | None = None,
                 params: GenParams | None = None,
                 on_params_error: Callable[[GenParams], None] | None = None) -> None:
        super().__init__(name="gpu-worker", daemon=True)
        if backend is None and backend_factory is None:
            raise ValueError("need a backend or a backend_factory")
        self.backend = backend
        self.mailbox = mailbox
        self.on_result = on_result
        self.on_status = on_status
        self.config = config
        self.backend_factory = backend_factory
        self.uploads: queue.Queue[ScreenUploadJob] = queue.Queue(maxsize=8)
        # Live params: ``_params`` is what the backend has applied; ``_pending_params``
        # is a latest-wins slot filled from any thread and applied before the next frame.
        # ``on_params_error(previous)`` (thread-safe) is told when a change is reverted.
        self._defaults = defaults_from_config(config)
        self._params = params or self._defaults
        self._pending_params: GenParams | None = None
        self._params_lock = threading.Lock()
        self.on_params_error = on_params_error
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._state = "busy"
        self._last_prompt: str | None = None
        self._failures = 0
        self._last_ms = 0.0
        self._fps = 0.0
        self._last_done: float | None = None
        self._frames = 0
        self._errors = 0
        # Pipeline latency handling (see _latency): the last backend job, how many settle
        # frames it still needs, and whether the pipeline holds a stale or empty state.
        self._settle_job: FrameJob | None = None
        self._settle_left = 0
        self._needs_prime = True
        # Refinement once drawing pauses: the step lists still to run for _settle_job, and
        # the upscaler (loaded on first use, if its weights exist).
        self._refine_plan: list[tuple[int, ...]] = []
        self._refine_waiting = True
        # The picture the next refinement round starts from: the settled drawing picture,
        # then each round's result (before upscaling).
        self._refine_base: Image.Image | None = None
        self._upscaler: Any = None
        # Picture size the backend is set up for (backends start in landscape).
        self._dims = canvas_dims(config)["landscape"]

    # ----- public API (any thread) ---------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def backend_name(self) -> str:
        backend = self.backend
        if backend is not None:
            try:
                return str(backend.name)
            except Exception:  # a broken backend must not break /healthz
                pass
        return str(getattr(self.config, "backend", "unknown"))

    def submit_upload(self, image: Image.Image) -> concurrent.futures.Future:
        """Queue an admin upload for screening; the future resolves to True (safe) / False.

        Raises queue.Full if too many uploads are pending.
        """
        job = ScreenUploadJob(image)
        self.uploads.put_nowait(job)
        return job.future

    @property
    def params(self) -> GenParams:
        """The params the backend currently runs with."""
        return self._params

    def set_params(self, params: GenParams) -> None:
        """Queue new live params (latest wins); applied on the worker before the next frame."""
        with self._params_lock:
            self._pending_params = params

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)
        self._fail_pending_uploads()

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "state": self._state,
                "backend": self.backend_name,
                "last_ms": round(self._last_ms, 1),
                "fps": round(self._fps, 2),
                "frames": self._frames,
                "errors": self._errors,
            }

    # ----- thread body -------------------------------------------------------------

    def run(self) -> None:
        self._start_backend()
        while not self._stop_event.is_set():
            self._drain_uploads()
            self._apply_pending_params()
            # Settling waits for the drawing to pause; once refining, only check quickly
            # whether a new stroke came in between passes.
            if self._settle_left:
                timeout = SETTLE_DELAY
            elif self._refine_plan:
                timeout = SETTLE_DELAY if self._refine_waiting else REFINE_POLL
            else:
                timeout = TAKE_TIMEOUT
            job = self.mailbox.take(timeout=timeout)
            if self._stop_event.is_set():
                continue
            if job is None:
                if self._settle_left and self._settle_job is not None:
                    self._settle_left -= 1
                    self._process_frame(self._settle_job, settling=True)
                elif self._refine_plan and self._settle_job is not None:
                    self._refine_waiting = False
                    self._process_frame(self._settle_job, refine=self._refine_plan.pop(0))
                continue
            if isinstance(job, FrameJob):
                self._apply_pending_params()  # may have arrived while we waited
                self._process_frame(job)
            else:
                log.warning("ignoring unexpected mailbox item %r", type(job).__name__)
        log.info("GPU worker stopped")

    def _set_state(self, state: str, msg: str = "") -> None:
        self._state = state
        try:
            self.on_status(state, msg)
        except Exception:
            log.exception("on_status callback failed")

    def _start_backend(self) -> bool:
        """Build (if needed) and warm up the backend. Returns True when ready."""
        self._set_state("busy", "warming up")
        try:
            if self.backend is None:
                self.backend = self.backend_factory()
            started = time.perf_counter()
            self.backend.warmup()
            log.info("backend %s ready in %.1fs", self.backend_name, time.perf_counter() - started)
        except Exception as exc:
            log.exception("backend warmup failed")
            self._record_failure(f"warmup failed: {exc}")
            return False
        self._last_prompt = None  # a fresh backend needs its prompt set again
        self._needs_prime, self._settle_left = True, 0
        self._dims = canvas_dims(self.config)["landscape"]  # a fresh backend's size
        self._failures = 0
        self._set_state("ready")
        # A fresh backend starts from config defaults: give it the current params.
        if self._params != self._defaults:
            self._apply_params(self._params, revert_to=self._defaults)
        return True

    def _apply_pending_params(self) -> None:
        if self.backend is None:
            return  # kept pending until a backend is up
        with self._params_lock:
            params, self._pending_params = self._pending_params, None
        if params is not None and params != self._params:
            self._apply_params(params, revert_to=self._params)

    def _apply_params(self, params: GenParams, revert_to: GenParams) -> bool:
        """Push ``params`` into the backend. On failure the backend keeps its previous
        params (Backend contract), we report the error and tell the app to revert."""
        try:
            self.backend.set_params(params)
        except Exception as exc:
            log.exception("applying params failed; reverting")
            self._params = revert_to
            self._set_state("error", f"Could not apply picture settings: {exc}")
            if self.on_params_error is not None:
                try:
                    self.on_params_error(revert_to)
                except Exception:
                    log.exception("on_params_error callback failed")
            return False
        self._params = params
        # A step-count change rebuilds the stream's buffers (and may change its latency).
        self._needs_prime, self._settle_left, self._refine_plan = True, 0, []
        return True

    def _record_failure(self, msg: str) -> None:
        self._failures += 1
        with self._stats_lock:
            self._errors += 1
        self._set_state("error", msg)

    def _maybe_rebuild(self) -> None:
        if self._failures < MAX_CONSECUTIVE_FAILURES or self.backend_factory is None:
            return
        log.warning("%d consecutive failures; rebuilding backend", self._failures)
        self.backend = None
        self._failures = 0
        if not self._start_backend():
            # Avoid a hot loop when the GPU is wedged; frames keep being dropped meanwhile.
            self._stop_event.wait(2.0)

    def _drain_uploads(self) -> None:
        while True:
            try:
                job = self.uploads.get_nowait()
            except queue.Empty:
                return
            if not job.future.set_running_or_notify_cancel():
                continue  # the HTTP request gave up
            try:
                if self.backend is None:
                    raise RuntimeError("backend not available")
                job.future.set_result(bool(self.backend.check_image_safe(job.image)))
            except Exception as exc:
                log.exception("upload screening failed")
                job.future.set_exception(exc)

    def _fail_pending_uploads(self) -> None:
        while True:
            try:
                job = self.uploads.get_nowait()
            except queue.Empty:
                return
            if job.future.set_running_or_notify_cancel():
                job.future.set_exception(RuntimeError("server shutting down"))

    def _latency(self) -> int:
        """Frames between a sketch going in and its picture coming out of the backend."""
        try:
            return max(0, int(getattr(self.backend, "latency_frames", 0)))
        except Exception:
            return 0

    def _ensure_dims(self, dims: tuple[int, int]) -> None:
        """Switch the backend to another picture size (a drawing in the other orientation).
        StreamDiffusion rebuilds its pipeline for that (seconds, engines are reused)."""
        if dims == self._dims:
            return
        set_size = getattr(self.backend, "set_size", None)
        if set_size is not None:
            self._set_state("busy", "turning the page")
            set_size(*dims)
            self._needs_prime, self._settle_left = True, 0
        self._dims = dims

    def _plan_refinement(self, params: GenParams) -> list[tuple[int, ...]]:
        """Step lists for the rounds run once drawing pauses (see REFINE_FIRST_START)."""
        if getattr(self.backend, "refine", None) is None or params.refine_steps <= 0:
            return []
        rounds = min(REFINE_MAX_ROUNDS, max(1, -(-params.refine_steps // REFINE_STEPS_PER_ROUND)))
        steps = max(1, round(params.refine_steps / rounds))
        if rounds == 1:
            starts = [REFINE_FIRST_START]
        else:
            span = REFINE_LAST_START - REFINE_FIRST_START
            starts = [round(REFINE_FIRST_START + i * span / (rounds - 1)) for i in range(rounds)]
        return [refine_t_index(start, steps) for start in starts]

    def _upscale(self, picture: Image.Image) -> Image.Image:
        """The refined picture through the AI upscaler (x2); unchanged if unavailable."""
        if self._upscaler is None:
            from .upscale import Upscaler

            self._upscaler = Upscaler(self.config.model.upscaler_model)
            if not self._upscaler.available:
                log.warning("upscaler weights not found at %s; pictures are not upscaled",
                            self._upscaler.model_path)
        if not self._upscaler.available:
            return picture
        return self._upscaler.upscale(picture, 2)

    def _process_frame(self, job: FrameJob, settling: bool = False,
                       refine: tuple[int, ...] | None = None) -> None:
        """Make and send the picture for ``job``. With pipeline latency, a fresh pipeline
        is primed with extra runs, and the frame is queued to settle once drawing pauses;
        a settle frame re-sends the same seq with the now up-to-date picture. After that,
        ``refine`` passes re-send it with more steps (and upscaled, if set)."""
        cfg = self.config
        params = self._params
        started = time.perf_counter()
        all_dims = canvas_dims(cfg, params.resolution)
        dims = all_dims.get(job.orientation or "landscape", all_dims["landscape"])
        layered = job.background is not None
        try:
            sketch = imaging.decode_sketch(job.png_bytes, dims, layered=layered)
        except ValueError as exc:
            # Bad client input, not a backend fault: skip without counting a failure.
            log.warning("dropping sketch seq=%d: %s", job.seq, exc)
            return
        used_backend = False
        try:
            if imaging.is_blank(sketch):
                picture = Image.new("RGB", dims, job.background or "white")
                # The pipeline still holds the old drawing: prime it before the next one.
                self._needs_prime, self._settle_left, self._refine_plan = True, 0, []
            else:
                used_backend = True
                if self.backend is None:
                    raise RuntimeError("backend not available")
                self._ensure_dims(dims)
                if job.prompt != self._last_prompt:
                    self.backend.set_prompt(job.prompt)
                    self._last_prompt = job.prompt
                line_map = None
                if cfg.model.controlnet and params.controlnet_enabled:
                    line_map = imaging.make_line_map(sketch, params.line_polarity,
                                                     params.line_threshold, params.line_dilate)
                # Without a line map the strokes are the only structure, so keep them.
                init = imaging.compose(sketch, job.background)
                if line_map is not None:
                    init = imaging.make_init_image(sketch, params.init_mode, params.init_blur,
                                                   params.line_threshold, job.background)
                if refine is not None:
                    base = self._refine_base if self._refine_base is not None else init
                    picture = self.backend.refine(base, line_map, list(refine))
                    self._refine_base = picture
                    if params.upscale == "x2":
                        picture = self._upscale(picture)
                    self._needs_prime = True  # the stream now runs the refine steps
                else:
                    latency = self._latency()
                    runs = 1 + (latency if self._needs_prime else 0)
                    for _ in range(runs):
                        picture = self.backend.generate(init, line_map)
                    self._needs_prime = False
                    self._refine_base = picture  # refinement starts from what is shown
                    if not settling:
                        self._settle_job, self._settle_left = job, latency
                        self._refine_plan = self._plan_refinement(params)
                        self._refine_waiting = True  # the first pass waits for a pause
            jpeg = imaging.encode_jpeg(picture, params.jpeg_quality)
        except Exception as exc:
            log.exception("frame seq=%d failed", job.seq)
            self._needs_prime, self._settle_left, self._refine_plan = True, 0, []
            self._last_prompt = None  # re-send the prompt after an error, to be safe
            self._record_failure(str(exc) or type(exc).__name__)
            self._maybe_rebuild()
            return

        # Only a frame that went through the backend proves it works; a blank canvas is
        # answered without it and must not hide a broken backend.
        if used_backend and (self._failures or self._state != "ready"):
            self._failures = 0
            self._set_state("ready")
        if not settling and refine is None:
            self._update_stats(started)
        try:
            self.on_result(job.seq, jpeg, job.owner)
        except Exception:
            log.exception("on_result callback failed")

    def _update_stats(self, started: float) -> None:
        now = time.perf_counter()
        with self._stats_lock:
            self._last_ms = (now - started) * 1000.0
            # Gaps longer than FPS_IDLE_GAP are pauses in drawing, not slow frames.
            if self._last_done is not None and now - self._last_done < FPS_IDLE_GAP:
                instant = 1.0 / max(now - self._last_done, 1e-6)
                self._fps = instant if self._fps == 0 else (
                    FPS_EMA_ALPHA * instant + (1 - FPS_EMA_ALPHA) * self._fps)
            self._last_done = now
            self._frames += 1
