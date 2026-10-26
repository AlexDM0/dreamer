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

from . import imaging, offline, refine
from .config import canvas_dims
from .mailbox import Mailbox
from .params import T_INDEX_MAX, GenParams, defaults_from_config

log = logging.getLogger("dreamer.worker")

MAX_CONSECUTIVE_FAILURES = 3
TAKE_TIMEOUT = 0.5
# A backend with pipeline latency (StreamDiffusion with 2+ steps returns the picture for
# an earlier frame) is re-run on the last sketch once no new sketch came for this long.
SETTLE_DELAY = 0.15
FPS_EMA_ALPHA = 0.2
FPS_IDLE_GAP = 2.0  # seconds


# The step path refinement continues on (kept here for callers of the old name).
refine_path = refine.continue_path


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
        # Whether to refine _settle_job once drawing pauses, and the picture now on screen
        # (refinement models keep its broad terms).
        self._refine_plan: bool = False
        self._shown: Image.Image | None = None
        self._upscaler: Any = None
        # Picture size the backend is set up for (backends start in landscape).
        self._dims = canvas_dims(config)["landscape"]
        # The picture model (params.picture_model) the current backend was built for.
        self._backend_kind: str | None = None

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
            # Settling and refining start once the drawing has paused for SETTLE_DELAY.
            timeout = SETTLE_DELAY if self._settle_left or self._refine_plan else TAKE_TIMEOUT
            job = self.mailbox.take(timeout=timeout)
            if self._stop_event.is_set():
                continue
            if job is None:
                if self._settle_left and self._settle_job is not None:
                    self._settle_left -= 1
                    self._process_frame(self._settle_job, settling=True)
                elif self._refine_plan and self._settle_job is not None:
                    self._refine_plan = False
                    self._refine_frame(self._settle_job)
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
                kind = self._params.picture_model
                # Factories without the argument (tests) only make the default model.
                self.backend = self.backend_factory() if kind == "turbo" else self.backend_factory(kind)
                self._backend_kind = kind
            started = time.perf_counter()
            self.backend.warmup()
            log.info("backend %s ready in %.1fs", self.backend_name, time.perf_counter() - started)
            if self.backend_name == "streamdiffusion":
                offline.mark_ready(self.config)  # the models are cached now
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

    def _switch_picture_model(self, params: GenParams, revert_to: GenParams) -> bool:
        """Unload the backend and build the one for ``params.picture_model`` (both do not
        fit in GPU memory together). If it fails to start, go back to ``revert_to``'s."""
        log.info("switching picture model %s -> %s", self._backend_kind, params.picture_model)
        self._set_state("busy", "switching picture model")
        self._close_backend()
        self._params = params
        self._needs_prime, self._settle_left, self._refine_plan = True, 0, False
        self._shown = None
        if self._start_backend():
            return True
        self._close_backend()
        self._params = revert_to
        self._start_backend()
        if self.on_params_error is not None:
            try:
                self.on_params_error(revert_to)
            except Exception:
                log.exception("on_params_error callback failed")
        return False

    def _close_backend(self) -> None:
        backend, self.backend = self.backend, None
        close = getattr(backend, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                log.exception("closing the backend failed")
        self._last_prompt = None

    def _apply_params(self, params: GenParams, revert_to: GenParams) -> bool:
        """Push ``params`` into the backend. On failure the backend keeps its previous
        params (Backend contract), we report the error and tell the app to revert."""
        if (params.picture_model != self._backend_kind and self.backend_factory is not None
                and self._backend_kind is not None):
            return self._switch_picture_model(params, revert_to)
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
        self._needs_prime, self._settle_left, self._refine_plan = True, 0, False
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

    def _plan_refinement(self, params: GenParams) -> bool:
        """Whether to refine once drawing pauses (on, and a backend that can)."""
        backend = self.backend
        can_refine = (getattr(backend, "refine_steps", None) is not None
                      or getattr(backend, "refine_own", None) is not None)
        return can_refine and params.refine_steps > 0

    def _frame_inputs(self, sketch: Image.Image, job: FrameJob,
                      params: GenParams) -> tuple[Image.Image, Image.Image | None]:
        """(init image, line map) for a drawing, as the backend gets them."""
        line_map = None
        if self.config.model.controlnet and params.controlnet_enabled:
            line_map = imaging.make_line_map(sketch, params.line_polarity,
                                             params.line_threshold, params.line_dilate)
        # Without a line map the strokes are the only structure, so keep them.
        init = imaging.compose(sketch, job.background)
        if line_map is not None:
            init = imaging.make_init_image(sketch, params.init_mode, params.init_blur,
                                           params.line_threshold, job.background)
        return init, line_map

    def _refine_frame(self, job: FrameJob) -> None:
        """Refine the drawing after it paused with params.refine_model (server/refine.py),
        sending every picture it makes (same seq, upscaled if set). Stops as soon as a new
        sketch is waiting, so drawing again is answered right away."""
        params = self._params
        all_dims = canvas_dims(self.config, params.resolution)
        dims = all_dims.get(job.orientation or "landscape", all_dims["landscape"])
        try:
            sketch = imaging.decode_sketch(job.png_bytes, dims, layered=job.background is not None)
            if imaging.is_blank(sketch):
                return
            init, line_map = self._frame_inputs(sketch, job, params)
            first = self._shown if self._shown is not None else init
            self._needs_prime = True  # the stream now runs refinement steps
            refine_own = getattr(self.backend, "refine_own", None)
            if refine_own is not None:  # a picture model with its own refinement (TCD)
                upscales_itself = False
                steps = refine_own(init, line_map, params.refine_steps)
            else:
                upscales_itself = refine.MODELS.get(params.refine_model, refine.MODELS[refine.DEFAULT_MODEL])[2]
                if upscales_itself and self._upscaler is None:
                    self._upscale(first)  # loads the upscaler (and warns once if it is missing)
                steps = refine.run(params.refine_model, self.backend, first, init, line_map,
                                   tuple(params.t_index_list), params.refine_steps, self._upscaler)
            try:
                for picture in steps:
                    if params.upscale == "x2" and not upscales_itself:
                        picture = self._upscale(picture)
                    self.on_result(job.seq, imaging.encode_jpeg(picture, params.jpeg_quality), job.owner)
                    if self.mailbox.pending or self._stop_event.is_set():
                        break
            finally:
                steps.close()
        except Exception as exc:
            log.exception("refining seq=%d failed", job.seq)
            self._needs_prime, self._settle_left, self._refine_plan = True, 0, False
            self._record_failure(str(exc) or type(exc).__name__)
            self._maybe_rebuild()

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

    def _process_frame(self, job: FrameJob, settling: bool = False) -> None:
        """Make and send the picture for ``job``. With pipeline latency, a fresh pipeline
        is primed with extra runs, and the frame is queued to settle once drawing pauses;
        a settle frame re-sends the same seq with the now up-to-date picture. After that,
        refinement continues the path (_refine_frame)."""
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
                self._needs_prime, self._settle_left, self._refine_plan = True, 0, False
            else:
                used_backend = True
                if self.backend is None:
                    raise RuntimeError("backend not available")
                self._ensure_dims(dims)
                if job.prompt != self._last_prompt:
                    self.backend.set_prompt(job.prompt)
                    self._last_prompt = job.prompt
                init, line_map = self._frame_inputs(sketch, job, params)
                latency = self._latency()
                runs = 1 + (latency if self._needs_prime else 0)
                for _ in range(runs):
                    picture = self.backend.generate(init, line_map)
                self._needs_prime = False
                self._shown = picture
                if not settling:
                    self._settle_job, self._settle_left = job, latency
                    self._refine_plan = self._plan_refinement(params)
            jpeg = imaging.encode_jpeg(picture, params.jpeg_quality)
        except Exception as exc:
            log.exception("frame seq=%d failed", job.seq)
            self._needs_prime, self._settle_left, self._refine_plan = True, 0, False
            self._last_prompt = None  # re-send the prompt after an error, to be safe
            self._record_failure(str(exc) or type(exc).__name__)
            self._maybe_rebuild()
            return

        # Only a frame that went through the backend proves it works; a blank canvas is
        # answered without it and must not hide a broken backend.
        if used_backend and (self._failures or self._state != "ready"):
            self._failures = 0
            self._set_state("ready")
        if not settling:
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
