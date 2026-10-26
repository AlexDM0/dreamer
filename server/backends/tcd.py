"""SD 1.5 + TCD backend: a picture model made for many steps (plain diffusers, no TensorRT).

Selected live with the "picture_model" param ("tcd"); the worker unloads the SD-Turbo backend
first, as both do not fit in 10 GB together. Drawing runs ``tcd_steps`` steps per picture
(about 0.3-0.4 s for 4 on an RTX 3080); once the drawing pauses, ``refine_own`` makes the
same picture again with more steps from the same seed. TCD (Trajectory Consistency
Distillation) keeps the scene as steps are added, so this refines instead of replacing
(docs/research/refinement-models.md, R6).

The start image only matters below strength 1.0: the measured default 1.0 starts from noise
and follows the line map, so the pictures are sharp at few steps but do not keep the stroke
colours. The safety checker is the same Freepik classifier as the TensorRT backend's, run in
PyTorch.
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Iterator

from PIL import Image

from .. import offline
from ..config import canvas_dims
from .base import DEFAULT_SEED, model_attr as _model_attr

log = logging.getLogger("dreamer.backend.tcd")

SAFETY_MODEL = "Freepik/nsfw_image_detector"
# Refinement re-renders at these step counts (up to the refine budget), after the drawing's.
REFINE_LADDER = (6, 8, 12, 16, 20)


class _OutputOnly:
    """Attention processor wrapper that returns only the output of a (output, cache) pair."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.processor(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result


class TcdBackend:
    def __init__(self, config: Any) -> None:
        self._base = _model_attr(config, "tcd_base", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self._lora = _model_attr(config, "tcd_lora", "h1t/TCD-SD15-LoRA")
        self._lora_file = _model_attr(config, "tcd_lora_file", "pytorch_lora_weights.safetensors")
        self._controlnet_id = _model_attr(config, "tcd_controlnet", "lllyasviel/control_v11p_sd15_scribble")
        self._strength = float(_model_attr(config, "tcd_strength", 1.0))
        self._gamma = float(_model_attr(config, "tcd_gamma", 0.3))
        self._safety_on = bool(_model_attr(config, "safety_checker", True))
        self._safety_threshold = float(_model_attr(config, "safety_threshold", 0.5))
        self._pipeline: Any = None
        self._checker: Any = None
        self._prompt_embeds: Any = None
        self._seed = DEFAULT_SEED
        self._steps = 4
        self._controlnet_scale = 0.55
        self._controlnet_on = True
        self._last_safe: Image.Image | None = None
        self._warmup_size = canvas_dims(config)["landscape"]
        self._config = config if hasattr(config, "model") else None

    @property
    def name(self) -> str:
        return "tcd"

    @property
    def latency_frames(self) -> int:
        return 0

    # ----- lifecycle -------------------------------------------------------------------

    def warmup(self) -> None:
        if self._config is not None and offline.is_offline() and not offline.tcd_ready(self._config):
            # Dreamer went offline after SD-Turbo's first start; TCD was never downloaded.
            offline.fetch_online(self._config, SAFETY_MODEL)
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, TCDScheduler

        log.info("Dreamer: loading SD 1.5 + TCD (%s, %s)", self._base, self._lora)
        controlnet = ControlNetModel.from_pretrained(self._controlnet_id, torch_dtype=torch.float16)
        pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self._base, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
            requires_safety_checker=False).to("cuda")
        # The fork's diffusers build returns (output, kv cache) from attention; the UNet
        # expects that, the VAE does not: its processors drop the cache (still fast SDPA;
        # the plain processor made decoding ~10x slower).
        pipeline.vae.set_attn_processor({name: _OutputOnly(processor)
                                         for name, processor in pipeline.vae.attn_processors.items()})
        pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights(self._lora, weight_name=self._lora_file)
        pipeline.fuse_lora()
        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        self._load_checker()
        if self._prompt_embeds is None:
            self.set_prompt("a colorful illustration")
        self.generate(Image.new("RGB", self._warmup_size, "black"), None)  # first call compiles kernels
        log.info("Dreamer: SD 1.5 + TCD ready")

    def _load_checker(self) -> None:
        import torch
        from transformers import AutoModelForImageClassification

        self._checker = AutoModelForImageClassification.from_pretrained(
            SAFETY_MODEL, torch_dtype=torch.float16).to("cuda").eval()

    def close(self) -> None:
        """Free the GPU memory (before switching to the other picture model)."""
        self._pipeline = self._checker = self._prompt_embeds = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # ----- params ----------------------------------------------------------------------

    def set_prompt(self, prompt: str) -> None:
        embeds, _ = self._pipeline.encode_prompt(prompt, "cuda", 1, False)
        self._prompt_embeds = embeds

    def set_params(self, params: Any) -> None:
        self._seed = int(getattr(params, "seed", DEFAULT_SEED))
        self._steps = max(1, int(getattr(params, "tcd_steps", 4)))
        self._controlnet_scale = float(params.controlnet_scale)
        self._controlnet_on = bool(params.controlnet_enabled)
        self._safety_on = bool(params.safety_checker)
        self._safety_threshold = float(params.safety_threshold)

    def set_size(self, width: int, height: int) -> None:
        """Any multiple of 8 works without a rebuild: the size follows the start image."""

    # ----- pictures --------------------------------------------------------------------

    def _render(self, init: Image.Image, line_map: Image.Image | None, steps: int) -> Image.Image:
        import torch

        use_lines = line_map is not None and self._controlnet_on
        control = line_map if use_lines else Image.new("RGB", init.size, "black")
        picture = self._pipeline(
            prompt_embeds=self._prompt_embeds, image=init, control_image=control,
            num_inference_steps=steps, strength=self._strength, guidance_scale=0.0, eta=self._gamma,
            controlnet_conditioning_scale=self._controlnet_scale if use_lines else 0.0,
            width=init.width, height=init.height,
            generator=torch.Generator("cuda").manual_seed(self._seed)).images[0]
        return self._checked(picture)

    def generate(self, init_image: Image.Image, line_map: Image.Image | None) -> Image.Image:
        return self._render(init_image, line_map, self._steps)

    def refine_own(self, init_image: Image.Image, line_map: Image.Image | None,
                   total_steps: int) -> Iterator[Image.Image]:
        """The same picture again with more steps (same seed), up to ``total_steps``."""
        for steps in [count for count in REFINE_LADDER if self._steps < count <= total_steps]:
            yield self._render(init_image, line_map, steps)

    # ----- safety ----------------------------------------------------------------------

    def _nsfw(self, image: Image.Image) -> bool:
        import numpy as np
        import torch
        import torchvision.transforms.v2.functional as functional

        tensor = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).to("cuda")
        # The whole picture squeezed to 448x448, like the TensorRT backend's checker: a centre
        # crop would leave the outer ~30% of a 640x448 picture unchecked.
        tensor = functional.resize(tensor, [448, 448], antialias=True,
                                   interpolation=functional.InterpolationMode.BICUBIC)
        tensor = tensor.float().div(255).clamp(0, 1)
        tensor = functional.normalize(tensor, [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
        with torch.no_grad():
            logits = self._checker(pixel_values=tensor.unsqueeze(0).half()).logits
        nsfw_probability = 1 - torch.softmax(logits.float(), dim=-1)[0, 0].item()
        return nsfw_probability >= self._safety_threshold

    def _checked(self, picture: Image.Image) -> Image.Image:
        """Flagged pictures are replaced by the last safe one (or black), like the
        TensorRT backend's "previous" fallback."""
        if not self._safety_on:
            return picture
        if self._nsfw(picture):
            log.info("safety checker flagged a picture; showing the previous one")
            return self._last_safe or Image.new("RGB", picture.size, "black")
        self._last_safe = picture
        return picture

    def check_image_safe(self, image: Image.Image) -> bool:
        return not self._nsfw(image)
