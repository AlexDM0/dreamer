"""StreamDiffusionBackend: real-time img2img via the Daydream StreamDiffusion fork.

Fork: https://github.com/daydreamlive/StreamDiffusion. Install steps: requirements-gpu.md.

This module must stay importable without torch: torch / streamdiffusion are imported in
the constructor, and every CUDA-touching call (wrapper build, prepare, frames) happens
on the GPU worker thread via warmup() / generate(). It cannot be exercised in CI; the
third-party calls below were checked against the installed fork's wrapper.py and
stream_parameter_updater.py.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterator

from PIL import Image

from ..config import canvas_dims
from .base import GenParamsLike

log = logging.getLogger("dreamer.backend.streamdiffusion")

# The wrapper's own default seed (StreamDiffusionWrapper(seed=2)).
DEFAULT_SEED = 2

# Used for prepare() before the first real prompt arrives from the worker.
DEFAULT_PROMPT = "a colorful illustration, for children, friendly, bright colors, wholesome"

_INSTALL_HINT = (
    "StreamDiffusion is not installed. Follow requirements-gpu.md, or set "
    "'backend: mock' in config.yaml to run without a GPU."
)


class _WarningCollector(logging.Handler):
    """Collects WARNING+ messages of a logger while attached."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _model_attr(config: Any, name: str, default: Any) -> Any:
    model = getattr(config, "model", None)
    return getattr(model, name, default) if model is not None else default


class StreamDiffusionBackend:
    def __init__(self, config: Any) -> None:
        # The only place torch/streamdiffusion get imported (SPEC hard rule).
        try:
            from streamdiffusion.wrapper import StreamDiffusionWrapper
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        self._wrapper_cls = StreamDiffusionWrapper

        # Pictures have the drawing's shape; set_size() switches orientation.
        self._width, self._height = canvas_dims(config)["landscape"]
        self._model_id = _model_attr(config, "base", "stabilityai/sd-turbo")
        self._controlnet_id = _model_attr(config, "controlnet", "thibaud/controlnet-sd21-scribble-diffusers")
        self._engine_dir = _model_attr(config, "engine_dir", "engines")
        self._max_steps = max(1, int(_model_attr(config, "max_steps", 8)))
        # Steps up to this run as one UNet batch; more run one after another (see _batched).
        self._batch_steps = min(self._max_steps, max(1, int(_model_attr(config, "max_batch_steps", 4))))

        # Live params (SPEC addendum); start from config.model defaults.
        self._t_index_list = [int(t) for t in _model_attr(config, "t_index_list", [6, 26])]
        self._seed: int | None = None  # None = wrapper's own default seed
        self._controlnet_enabled = self._controlnet_id is not None
        self._controlnet_scale = float(_model_attr(config, "controlnet_scale", 0.55))
        self._safety_on = bool(_model_attr(config, "safety_checker", True))
        self._safety_threshold = float(_model_attr(config, "safety_threshold", 0.5))

        self._wrapper: Any = None
        # What the stream runs right now: the draw steps (_t_index_list), or a refine pass's
        # steps one after another (refine()); generate() switches back by itself.
        self._stream_t_index: list[int] = []
        self._stream_batched = True
        self._prompt = DEFAULT_PROMPT
        self._pending_prompt: str | None = None
        self._pending_params: GenParamsLike | None = None
        self._warned_no_screening = False

    @property
    def name(self) -> str:
        return "streamdiffusion"

    @property
    def latency_frames(self) -> int:
        """With the denoising batch, each call advances every step by one, so a frame's
        picture comes out len(t_index_list) - 1 calls later (frame_buffer_size=1). The
        worker primes and settles the pipeline with that many extra calls. Steps run one
        after another have no such delay."""
        return len(self._t_index_list) - 1 if self._batched(self._t_index_list) else 0

    def _batched(self, t_index: list[int]) -> bool:
        """Whether ``t_index`` runs as one UNet batch (the fork's denoising batch) or, with
        more steps than the engine's batch range, one step after another at batch 1."""
        return len(t_index) <= self._batch_steps

    def _update_t_index(self, wrapper: Any, updates: dict[str, Any],
                        batched: bool | None = None) -> None:
        """``wrapper.update_stream_params(**updates)``, switching the stream between batched
        and one-after-another steps when the new t_index_list needs it. The wrapper only
        accepts img2img with the denoising batch at construction, but the pipeline and its
        parameter updater support both: a step-count change recomputes batch size, noise and
        buffers for the current mode, and crossing max_batch_steps always changes the count.
        ``batched`` forces the mode (refine passes always run one after another)."""
        stream = wrapper.stream
        previous = stream.use_denoising_batch
        new_t_index = updates.get("t_index_list")
        if new_t_index is not None and "seed" not in updates:
            # A step-count change regenerates the stream's noise from its generator, which
            # has moved on since: the same drawing would get different noise every time and
            # a refinement round could repaint a different picture. Re-seed in the same call
            # (the updater applies the seed after the step change).
            updates = {**updates, "seed": self._seed if self._seed is not None else DEFAULT_SEED}
        if new_t_index is not None:
            mode = self._batched(new_t_index) if batched is None else batched
            if mode != previous and len(new_t_index) == len(self._stream_t_index):
                # The updater only rebuilds batch buffers when the step COUNT changes; pass
                # through another count first so a mode switch gets buffers that fit.
                stream.use_denoising_batch = mode
                wrapper.update_stream_params(t_index_list=new_t_index[:-1] or [new_t_index[0], 49])
            stream.use_denoising_batch = mode
        try:
            wrapper.update_stream_params(**updates)
        except Exception:
            stream.use_denoising_batch = previous
            raise
        if new_t_index is not None:
            self._stream_t_index, self._stream_batched = list(new_t_index), stream.use_denoising_batch

    def _ensure_stream(self, t_index: list[int], batched: bool) -> None:
        """Make the stream run ``t_index`` in the given mode (no-op when it already does)."""
        if self._stream_t_index == t_index and self._stream_batched == batched:
            return
        self._update_t_index(self._wrapper, {"t_index_list": t_index}, batched=batched)

    # -- lifecycle -----------------------------------------------------------------

    def _controlnet_config(self) -> list[dict[str, Any]]:
        return [{
            "model_id": self._controlnet_id,
            "preprocessor": "passthrough",  # we send our own line map
            "conditioning_scale": self._controlnet_scale,
        }]

    def warmup(self) -> None:
        """Build the wrapper (first run compiles TensorRT engines: slow, once) and prepare."""
        if self._wrapper is not None:
            return
        use_controlnet = self._controlnet_id is not None
        # The fork's UNet engine path does not encode ControlNet support, so a UNet built
        # without ControlNet would be reused with it and silently ignore the control image.
        # Keep the two variants in separate folders.
        engine_dir = os.path.join(self._engine_dir, "controlnet" if use_controlnet else "plain")
        log.info("Dreamer: loading %s (TensorRT engines in %s; first run builds them, "
                 "which can take many minutes)", self._model_id, engine_dir)
        # The wrapper must start batched; more steps than the batch range are applied below.
        start_t_index = self._t_index_list[:self._batch_steps]
        # The fork only logs a warning when the ControlNet engine fails to build or load and
        # carries on in a state that later crashed with a CUDA illegal memory access (seen
        # with max_batch_steps 3). Catch that warning and fail loudly instead.
        engine_warnings = _WarningCollector()
        fork_log = logging.getLogger("streamdiffusion.wrapper")
        fork_log.addHandler(engine_warnings)
        try:
            wrapper = self._construct_wrapper(start_t_index, engine_dir, use_controlnet)
        finally:
            fork_log.removeHandler(engine_warnings)
        failed = [msg for msg in engine_warnings.messages if "Failed to compile/load ControlNet engine" in msg]
        if failed:
            del wrapper
            raise RuntimeError(
                f"{failed[0]}. Delete the ControlNet engine folder under {engine_dir} for this "
                "max_batch_steps and restart, or use another model.max_batch_steps.")
        self._finish_warmup(wrapper, start_t_index)

    def _construct_wrapper(self, start_t_index: list[int], engine_dir: str,
                           use_controlnet: bool) -> Any:
        return self._wrapper_cls(
            model_id_or_path=self._model_id,
            t_index_list=start_t_index,
            mode="img2img",
            width=self._width,
            height=self._height,
            frame_buffer_size=1,
            use_denoising_batch=True,
            # UNet batch = steps; the engine's batch range caps the live step count.
            min_batch_size=1,
            max_batch_size=self._batch_steps,
            cfg_type="none",
            use_tiny_vae=True,
            acceleration="tensorrt",
            engine_dir=engine_dir,
            use_controlnet=use_controlnet,
            controlnet_config=self._controlnet_config() if use_controlnet else None,
            use_safety_checker=self._safety_on,
            # On an NSFW frame the wrapper returns the previous safe frame.
            safety_checker_fallback_type="previous",
            safety_checker_threshold=self._safety_threshold,
            output_type="pil",
        )

    def _finish_warmup(self, wrapper: Any, start_t_index: list[int]) -> None:
        wrapper.prepare(prompt=self._pending_prompt or self._prompt, num_inference_steps=50)
        if self._pending_prompt is not None:
            self._prompt, self._pending_prompt = self._pending_prompt, None
        self._wrapper = wrapper
        self._stream_t_index, self._stream_batched = list(start_t_index), True
        log.info("Dreamer: StreamDiffusion ready (%dx%d)", self._width, self._height)
        # A rebuilt wrapper (set_size) starts from the constructor's seed and an enabled
        # ControlNet; carry over what the live params had changed.
        restore: dict[str, Any] = {}
        if start_t_index != self._t_index_list:
            restore["t_index_list"] = self._t_index_list
        if self._seed is not None:
            restore["seed"] = self._seed
        if self._controlnet_id is not None and not self._controlnet_enabled:
            cfg = self._controlnet_config()
            cfg[0]["enabled"] = False
            restore["controlnet_config"] = cfg
        if restore:
            self._update_t_index(wrapper, restore)
        if self._pending_params is not None:
            params, self._pending_params = self._pending_params, None
            self.set_params(params)

    def set_size(self, width: int, height: int) -> None:
        """Make pictures of another size (a drawing in the other orientation). The fork
        restarts the pipeline for a new resolution, so the wrapper is rebuilt; its
        TensorRT engines have dynamic shapes (384..1024), so they are reused, not rebuilt."""
        if (width, height) == (self._width, self._height):
            return
        self._width, self._height = int(width), int(height)
        if self._wrapper is None:
            return  # warmup() builds it at the new size
        old, self._wrapper = self._wrapper, None
        del old
        import gc

        gc.collect()
        try:
            import torch
        except ImportError:  # only in tests with a fake wrapper
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.warmup()

    # -- per-frame controls ----------------------------------------------------------

    def set_prompt(self, prompt: str) -> None:
        # Applied on the next generate(); wrapper(prompt=...) calls update_prompt internally.
        # Always overwrite the slot: if a frame failed after an earlier set_prompt, a
        # stale pending prompt must not win over the one asked for now.
        self._pending_prompt = None if prompt == self._prompt else prompt

    def set_params(self, params: GenParamsLike) -> None:
        """Push changed live params into the stream. On error, local state is unchanged."""
        if self._wrapper is None:
            self._pending_params = params  # applied right after prepare()
            return
        t_index = [int(t) for t in params.t_index_list]
        cn_enabled = bool(params.controlnet_enabled) and self._controlnet_id is not None
        cn_scale = float(params.controlnet_scale)

        updates: dict[str, Any] = {}
        if t_index != self._t_index_list:
            # Up to max_batch_steps steps share one UNet batch; more run one by one, so
            # any step count up to max_steps fits the engine's batch range.
            updates["t_index_list"] = t_index
        if int(params.seed) != self._seed:
            updates["seed"] = int(params.seed)
        if self._controlnet_id is not None and (
            cn_enabled != self._controlnet_enabled or cn_scale != self._controlnet_scale
        ):
            cfg = self._controlnet_config()
            cfg[0]["conditioning_scale"] = cn_scale
            # The updater diffs by model_id and sets enabled_list[i] from "enabled".
            cfg[0]["enabled"] = cn_enabled
            updates["controlnet_config"] = cfg
        if bool(params.safety_checker) != self._safety_on:
            # The wrapper only loads a checker when built with it on (or its engine
            # exists); turning it on without one would crash every frame.
            if params.safety_checker and getattr(self._wrapper, "safety_checker", None) is None:
                raise RuntimeError("safety filter needs a restart with model.safety_checker: true")
            updates["use_safety_checker"] = bool(params.safety_checker)
        if float(params.safety_threshold) != self._safety_threshold:
            updates["safety_checker_threshold"] = float(params.safety_threshold)
        if not updates:
            return

        self._update_t_index(self._wrapper, updates)
        # Only commit local state once the wrapper accepted the change.
        self._t_index_list = t_index
        self._seed = int(params.seed)
        if self._controlnet_id is not None:
            self._controlnet_enabled, self._controlnet_scale = cn_enabled, cn_scale
        self._safety_on = bool(params.safety_checker)
        self._safety_threshold = float(params.safety_threshold)
        log.info("Dreamer: applied stream params %s", sorted(updates))

    def generate(self, init_image: Image.Image, line_map: Image.Image | None) -> Image.Image:
        """A picture with the draw steps (batched up to max_batch_steps)."""
        if self._wrapper is None:
            self.warmup()
        self._ensure_stream(self._t_index_list, self._batched(self._t_index_list))
        return self._run(init_image, line_map)

    def refine_steps(self, init_image: Image.Image, line_map: Image.Image | None,
                     t_index: list[int], first_shown: int,
                     noise: str = "fresh") -> Iterator[Image.Image]:
        """One denoising path over ``t_index``, run one step after another, yielding the
        picture after each step from step ``first_shown`` (1-based) on.

        This is the fork's own step-by-step loop (StreamDiffusion.predict_x0_batch, non-batched
        LCM branch), opened up so every step's prediction can be shown: the picture keeps the
        same composition and each step changes less than the last, because the noise put back
        between steps shrinks along the path. With the same seed, the first steps are the
        quick drawing picture: draw 1 step + 3 more here == a 4-step picture.
        ``noise``: what is put back between steps: "fresh" random noise (the fork's way),
        "fixed" (the same noise every step: converges instead of wandering).
        """
        import torch

        if self._wrapper is None:
            self.warmup()
        wrapper = self._wrapper
        stream = wrapper.stream
        self._ensure_stream([int(t) for t in t_index], False)
        image = self._prepare_inputs(init_image, line_map)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed if self._seed is not None else DEFAULT_SEED)
        if self._pending_prompt is not None:
            wrapper.update_prompt(self._pending_prompt)
            self._prompt, self._pending_prompt = self._pending_prompt, None
        with torch.no_grad():
            x = wrapper.preprocess_image(image)
            sample = stream.encode_image(x)
            steps = len(stream.sub_timesteps_tensor)
            for idx, timestep in enumerate(stream.sub_timesteps_tensor):
                t = timestep.to(stream.device).view(1,).repeat(stream.frame_bff_size,)
                x_0_pred, _ = stream.unet_step(sample, t, idx)
                if idx + 1 >= first_shown:
                    yield self._checked_picture(stream.decode_image(x_0_pred))
                if idx < steps - 1:
                    sample = stream.alpha_prod_t_sqrt[idx + 1] * x_0_pred
                    if stream.do_add_noise:
                        added = stream.init_noise[0:1] if noise == "fixed" else torch.randn_like(
                            x_0_pred, device=stream.device, dtype=stream.dtype)
                        sample = sample + stream.beta_prod_t_sqrt[idx + 1] * added

    def latent_ladder(self, first: Image.Image, line_map: Image.Image | None,
                      t_index: list[int], *, chain: bool, anchor_sigma: float | None,
                      gamma: float, anchor_weights: list[float] | None = None,
                      control_images: list[Image.Image | None] | None = None,
                      anchor_image: Image.Image | None = None) -> Iterator[Image.Image]:
        """Refine ``first`` (the picture on screen) in single-step rungs at ``t_index`` (rising:
        less noise each rung), yielding the picture after every rung.

        - ``chain``: each rung starts from the previous rung's result (detail accumulates);
          otherwise every rung starts again from ``first`` (SDEdit, drift-free).
        - ``anchor_sigma``: if set, the low frequencies of every rung's prediction are replaced
          by ``first``'s (Gaussian low-pass on the latent, sigma in latent pixels; ILVR-style),
          so layout, colours and light cannot move; ``anchor_weights`` per rung (default 1).
        - ``gamma``: re-noise with sqrt(1-gamma) * the predicted noise + sqrt(gamma) * fresh
          noise (TCD-style stochasticity): 0 is deterministic, 1 is fresh noise only.
        - ``control_images``: per-rung ControlNet image (None keeps ``line_map``).
        - ``anchor_image``: pin the low frequencies to this picture instead of ``first`` (e.g.
          the upscaled first picture while refining a later rung's result).
        """
        import torch
        from torch.nn import functional as F

        if self._wrapper is None:
            self.warmup()
        wrapper, stream = self._wrapper, self._wrapper.stream
        self._ensure_stream([int(t) for t in t_index], False)
        image = self._prepare_inputs(first, line_map)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed if self._seed is not None else DEFAULT_SEED)

        def low_pass(latent: torch.Tensor) -> torch.Tensor:
            radius = max(1, int(round(anchor_sigma * 3)))
            coords = torch.arange(-radius, radius + 1, device=latent.device, dtype=torch.float32)
            kernel = torch.exp(-(coords ** 2) / (2 * anchor_sigma ** 2))
            kernel = (kernel / kernel.sum()).to(latent.dtype)
            channels = latent.shape[1]
            horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
            vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
            padded = F.pad(latent, (radius, radius, 0, 0), mode="replicate")
            latent = F.conv2d(padded, horizontal, groups=channels)
            padded = F.pad(latent, (0, 0, radius, radius), mode="replicate")
            return F.conv2d(padded, vertical, groups=channels)

        with torch.no_grad():
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
                retrieve_latents,
            )

            x = wrapper.preprocess_image(image).to(device=stream.device, dtype=stream.vae.dtype)
            # The clean latent of the picture on screen (encode_image() would add noise).
            anchor = retrieve_latents(stream.vae.encode(x), stream.generator)
            anchor = (anchor * stream.vae.config.scaling_factor).to(stream.dtype)
            current = anchor
            if anchor_image is not None:
                pinned = wrapper.preprocess_image(self._prepare_inputs(anchor_image, None))
                pinned = retrieve_latents(stream.vae.encode(pinned.to(device=stream.device, dtype=stream.vae.dtype)),
                                          stream.generator)
                anchor = (pinned * stream.vae.config.scaling_factor).to(stream.dtype)
            anchor_low = low_pass(anchor) if anchor_sigma else None
            eps = stream.init_noise[0:1]
            for idx, timestep in enumerate(stream.sub_timesteps_tensor):
                if control_images is not None and control_images[idx] is not None:
                    self._wrapper.update_control_image(0, control_images[idx])
                start = current if chain else (anchor if anchor_image is None else current)
                noise = eps
                if gamma > 0:
                    noise = (1 - gamma) ** 0.5 * eps + gamma ** 0.5 * torch.randn_like(eps)
                sample = stream.alpha_prod_t_sqrt[idx] * start + stream.beta_prod_t_sqrt[idx] * noise
                t = timestep.to(stream.device).view(1,).repeat(stream.frame_bff_size,)
                x_0_pred, model_pred = stream.unet_step(sample, t, idx)
                if anchor_low is not None:
                    weight = anchor_weights[idx] if anchor_weights else 1.0
                    predicted_low = low_pass(x_0_pred)
                    x_0_pred = x_0_pred - predicted_low + weight * anchor_low + (1 - weight) * predicted_low
                current = x_0_pred
                if model_pred is not None and model_pred.shape == eps.shape:
                    eps = model_pred  # the model's own noise estimate, for the next rung
                yield self._checked_picture(stream.decode_image(x_0_pred))

    def _checked_picture(self, image_tensor: Any) -> Image.Image:
        """Decoded tensor -> PIL, through the wrapper's safety checker like img2img()."""
        wrapper = self._wrapper
        picture = wrapper.postprocess_image(image_tensor, output_type="pil")
        if wrapper.use_safety_checker:
            denormalized = (image_tensor / 2 + 0.5).clamp(0, 1)
            if wrapper.safety_checker(denormalized, wrapper.safety_checker_threshold):
                return wrapper.nsfw_fallback_img
            wrapper.nsfw_fallback_img = picture
        return picture

    def _prepare_inputs(self, init_image: Image.Image, line_map: Image.Image | None) -> Image.Image:
        """Init image at the picture size, and the line map pushed into the ControlNet."""
        size = (self._width, self._height)
        image = init_image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        if line_map is not None and self._controlnet_id is not None and self._controlnet_enabled:
            lm = line_map.convert("RGB")
            if lm.size != size:
                lm = lm.resize(size, Image.Resampling.NEAREST)
            self._wrapper.update_control_image(0, lm)
        return image

    def _run(self, init_image: Image.Image, line_map: Image.Image | None) -> Image.Image:
        image = self._prepare_inputs(init_image, line_map)
        if not self._stream_batched:
            # One-after-another steps re-noise between steps with torch.randn_like on the
            # global CUDA generator, so every frame got new noise and the picture flickered
            # (~16/255 per frame on an unchanged drawing). Re-seeding per frame makes that
            # noise the same each frame; the surprise number still picks which noise.
            try:
                import torch
            except ImportError:  # only in tests with a fake wrapper
                torch = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.manual_seed(self._seed if self._seed is not None else DEFAULT_SEED)
        if self._pending_prompt is not None:
            prompt = self._pending_prompt
            out = self._wrapper(image=image, prompt=prompt)
            self._prompt, self._pending_prompt = prompt, None
        else:
            out = self._wrapper(image=image)

        # output_type="pil" with frame_buffer_size=1 returns one PIL image (also the
        # black/previous fallback on an NSFW frame); tolerate a list anyway.
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        if not isinstance(out, Image.Image):
            raise TypeError(f"unexpected StreamDiffusion output type: {type(out).__name__}")
        return out

    # -- admin upload screening ----------------------------------------------------

    def _safety_checker(self) -> Any:
        """The wrapper's loaded safety checker, if it exposes one."""
        if self._wrapper is None or not self._safety_on:
            return None
        # wrapper.safety_checker is an NSFWDetectorEngine (TensorRT only), created when
        # use_safety_checker=True or its engine already exists.
        checker = getattr(self._wrapper, "safety_checker", None)
        return checker if callable(checker) else None

    def check_image_safe(self, image: Image.Image) -> bool:
        """Screen an admin upload with the wrapper's own safety checker (SPEC).

        Without one, uploads are allowed and a warning is logged once. If the checker
        exists but fails or answers in an unknown shape, the exception propagates and
        the upload is rejected (fail closed): the admin sees "Could not check".
        """
        checker = self._safety_checker()
        if checker is None:
            if not self._warned_no_screening:
                self._warned_no_screening = True
                log.warning("Dreamer: upload screening unavailable (no safety checker loaded); "
                            "style pictures are stored unscreened")
            return True
        size = (self._width, self._height)
        checker_input = self._checker_input(image.convert("RGB").resize(size, Image.Resampling.LANCZOS))
        # NSFWDetectorEngine(image_tensor, threshold) returns True when NSFW.
        result = checker(checker_input, self._safety_threshold)
        return not _is_flagged(result, self._safety_threshold)

    def _checker_input(self, image: Image.Image) -> Any:
        """RGB image -> the (1, 3, H, W) fp16 CUDA tensor in 0..1 the checker takes
        (it resizes to 448 itself). Separate so tests without torch can stub it."""
        import numpy as np
        import torch

        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device="cuda", dtype=torch.float16) / 255.0


def _is_flagged(result: Any, threshold: float) -> bool:
    """Interpret a safety checker's answer. Raises TypeError for unknown shapes.

    The fork's NSFWDetectorEngine returns a bool. Also handled, for other checkers: a
    probability in 0..1 compared with the configured threshold; diffusers-style
    ``(images, flags)``; or a sequence of per-image flags/scores.
    """
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        return float(result) >= threshold
    if isinstance(result, (list, tuple)) and result:
        if len(result) == 2 and isinstance(result[1], (list, tuple, bool)):
            result = result[1]  # (images, has_nsfw_concepts)
        if isinstance(result, bool):
            return result
        if isinstance(result, (list, tuple)) and result and all(
                isinstance(r, (bool, int, float)) for r in result):
            return any(_is_flagged(r, threshold) for r in result)
    if hasattr(result, "item"):  # 0-d tensor / numpy scalar
        return _is_flagged(result.item(), threshold)
    raise TypeError(f"unexpected safety checker result: {type(result).__name__}")
