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
from typing import Any

from PIL import Image

from ..config import canvas_dims
from .base import GenParamsLike

log = logging.getLogger("dreamer.backend.streamdiffusion")

# Used for prepare() before the first real prompt arrives from the worker.
DEFAULT_PROMPT = "a colorful illustration, for children, friendly, bright colors, wholesome"

_INSTALL_HINT = (
    "StreamDiffusion is not installed. Follow requirements-gpu.md, or set "
    "'backend: mock' in config.yaml to run without a GPU."
)


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

        # Live params (SPEC addendum); start from config.model defaults.
        self._t_index_list = [int(t) for t in _model_attr(config, "t_index_list", [6, 26])]
        self._seed: int | None = None  # None = wrapper's own default seed
        self._controlnet_enabled = self._controlnet_id is not None
        self._controlnet_scale = float(_model_attr(config, "controlnet_scale", 0.55))
        self._safety_on = bool(_model_attr(config, "safety_checker", True))
        self._safety_threshold = float(_model_attr(config, "safety_threshold", 0.5))

        self._wrapper: Any = None
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
        worker primes and settles the pipeline with that many extra calls."""
        return len(self._t_index_list) - 1

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
        wrapper = self._wrapper_cls(
            model_id_or_path=self._model_id,
            t_index_list=self._t_index_list,
            mode="img2img",
            width=self._width,
            height=self._height,
            frame_buffer_size=1,
            use_denoising_batch=True,
            # UNet batch = steps; the engine's batch range caps the live step count.
            min_batch_size=1,
            max_batch_size=self._max_steps,
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
        wrapper.prepare(prompt=self._pending_prompt or self._prompt, num_inference_steps=50)
        if self._pending_prompt is not None:
            self._prompt, self._pending_prompt = self._pending_prompt, None
        self._wrapper = wrapper
        log.info("Dreamer: StreamDiffusion ready (%dx%d)", self._width, self._height)
        # A rebuilt wrapper (set_size) starts from the constructor's seed and an enabled
        # ControlNet; carry over what the live params had changed.
        restore: dict[str, Any] = {}
        if self._seed is not None:
            restore["seed"] = self._seed
        if self._controlnet_id is not None and not self._controlnet_enabled:
            cfg = self._controlnet_config()
            cfg[0]["enabled"] = False
            restore["controlnet_config"] = cfg
        if restore:
            wrapper.update_stream_params(**restore)
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
            # Changing the list LENGTH changes the UNet batch, which may exceed the
            # TensorRT engine's batch range and fail; the worker then reverts params.
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

        self._wrapper.update_stream_params(**updates)
        # Only commit local state once the wrapper accepted the change.
        self._t_index_list = t_index
        self._seed = int(params.seed)
        if self._controlnet_id is not None:
            self._controlnet_enabled, self._controlnet_scale = cn_enabled, cn_scale
        self._safety_on = bool(params.safety_checker)
        self._safety_threshold = float(params.safety_threshold)
        log.info("Dreamer: applied stream params %s", sorted(updates))

    def generate(self, init_image: Image.Image, line_map: Image.Image | None) -> Image.Image:
        if self._wrapper is None:
            self.warmup()
        size = (self._width, self._height)
        image = init_image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)

        if line_map is not None and self._controlnet_id is not None and self._controlnet_enabled:
            lm = line_map.convert("RGB")
            if lm.size != size:
                lm = lm.resize(size, Image.Resampling.NEAREST)
            self._wrapper.update_control_image(0, lm)

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
