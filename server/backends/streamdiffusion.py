"""StreamDiffusionBackend: real-time img2img via the Daydream StreamDiffusion fork.

Fork: https://github.com/daydreamlive/StreamDiffusion. Install steps: requirements-gpu.md.

This module must stay importable without torch: torch / streamdiffusion are imported in
the constructor, and every CUDA-touching call (wrapper build, prepare, frames) happens
on the GPU worker thread via warmup() / generate(). It cannot be exercised in CI, so
uncertain third-party API use is marked with "# VERIFY:".
"""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from .base import GenParamsLike

log = logging.getLogger("dreamer.backend.streamdiffusion")

# Used for prepare() before the first real prompt arrives from the worker.
DEFAULT_PROMPT = "a child's drawing turned into a picture, for children, friendly, bright colors, wholesome"
NSFW_MODEL_ID = "Freepik/nsfw_image_detector"
# VERIFY: label names of Freepik/nsfw_image_detector. The model card describes the
# levels neutral / low / medium / high; "medium" and "high" are treated as unsafe.
_UNSAFE_LABELS = {"medium", "high", "nsfw", "porn", "hentai", "sexy", "explicit", "unsafe"}

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
            # VERIFY: import path. Upstream demos used "from utils.wrapper import
            # StreamDiffusionWrapper"; the fork ships src/streamdiffusion/wrapper.py.
            from streamdiffusion.wrapper import StreamDiffusionWrapper
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        self._wrapper_cls = StreamDiffusionWrapper

        self._size = int(getattr(config, "size", 512))
        self._model_id = _model_attr(config, "base", "stabilityai/sd-turbo")
        self._controlnet_id = _model_attr(config, "controlnet", "thibaud/controlnet-sd21-scribble-diffusers")
        self._engine_dir = _model_attr(config, "engine_dir", "engines")

        # Live params (SPEC addendum); start from config.model defaults.
        self._t_index_list = [int(t) for t in _model_attr(config, "t_index_list", [30, 42])]
        self._seed: int | None = None  # None = wrapper's own default seed
        self._controlnet_enabled = self._controlnet_id is not None
        self._controlnet_scale = float(_model_attr(config, "controlnet_scale", 0.8))
        self._safety_on = bool(_model_attr(config, "safety_checker", True))
        self._safety_threshold = float(_model_attr(config, "safety_threshold", 0.5))

        self._wrapper: Any = None
        self._prompt = DEFAULT_PROMPT
        self._pending_prompt: str | None = None
        self._pending_params: GenParamsLike | None = None
        self._nsfw_pipe: Any = None
        self._nsfw_failed = False

    @property
    def name(self) -> str:
        return "streamdiffusion"

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
        log.info("Dreamer: loading %s (TensorRT engines in %s; first run builds them, "
                 "which can take many minutes)", self._model_id, self._engine_dir)
        wrapper = self._wrapper_cls(
            model_id_or_path=self._model_id,
            t_index_list=self._t_index_list,
            mode="img2img",
            width=self._size,
            height=self._size,
            frame_buffer_size=1,
            use_denoising_batch=True,
            cfg_type="none",
            use_tiny_vae=True,
            acceleration="tensorrt",
            engine_dir=self._engine_dir,
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
        log.info("Dreamer: StreamDiffusion ready")
        if self._pending_params is not None:
            params, self._pending_params = self._pending_params, None
            self.set_params(params)

    # -- per-frame controls ----------------------------------------------------------

    def set_prompt(self, prompt: str) -> None:
        # Applied on the next generate(); wrapper(prompt=...) calls update_prompt internally.
        if prompt != self._prompt:
            self._pending_prompt = prompt

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
            # VERIFY: changing the list LENGTH may exceed the TensorRT engine's batch
            # range and fail; the worker reverts params and reports the error.
            updates["t_index_list"] = t_index
        if int(params.seed) != self._seed:
            updates["seed"] = int(params.seed)
        if self._controlnet_id is not None and (
            cn_enabled != self._controlnet_enabled or cn_scale != self._controlnet_scale
        ):
            cfg = self._controlnet_config()
            cfg[0]["conditioning_scale"] = cn_scale
            cfg[0]["enabled"] = cn_enabled  # VERIFY: "enabled" key honoured by update_stream_params
            updates["controlnet_config"] = cfg
        if bool(params.safety_checker) != self._safety_on:
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
        size = (self._size, self._size)
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

        # VERIFY: with output_type="pil" and frame_buffer_size=1 the wrapper returns a
        # PIL image; tolerate a one-element list just in case.
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        if not isinstance(out, Image.Image):
            raise TypeError(f"unexpected StreamDiffusion output type: {type(out).__name__}")
        return out

    # -- admin upload screening ----------------------------------------------------

    def _load_nsfw_pipe(self) -> Any:
        if self._nsfw_pipe is None and not self._nsfw_failed:
            try:
                from transformers import pipeline

                device = -1
                try:
                    import torch

                    device = 0 if torch.cuda.is_available() else -1
                except ImportError:
                    pass
                self._nsfw_pipe = pipeline("image-classification", model=NSFW_MODEL_ID, device=device)
            except Exception:  # missing package, download failure, unsupported model...
                self._nsfw_failed = True
                log.warning("Dreamer: upload screening unavailable (could not load %s); "
                            "style pictures will be allowed unscreened", NSFW_MODEL_ID, exc_info=True)
        return self._nsfw_pipe

    def check_image_safe(self, image: Image.Image) -> bool:
        pipe = self._load_nsfw_pipe()
        if pipe is None:
            return True
        try:
            results = pipe(image.convert("RGB"), top_k=10)
        except Exception:
            log.warning("Dreamer: upload screening failed; allowing image", exc_info=True)
            return True
        unsafe = sum(float(r["score"]) for r in results if str(r["label"]).lower() in _UNSAFE_LABELS)
        log.info("Dreamer: upload screening %s -> unsafe score %.3f", results, unsafe)
        return unsafe < self._safety_threshold
