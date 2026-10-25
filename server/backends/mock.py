"""MockBackend: deterministic, GPU-free stand-in for StreamDiffusion.

Produces a stylised version of the sketch (posterize + blur + tint) so the whole app,
tests and e2e runs work without torch. Output depends only on the inputs, the prompt
and the params, never on time or randomness.
"""
from __future__ import annotations

import colorsys
import hashlib
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .base import GenParamsLike

MAGENTA = (255, 0, 255)


class MockBackend:
    def __init__(self, mock_delay_ms: int = 30) -> None:
        self._delay_s = max(0, mock_delay_ms) / 1000.0
        self._prompt = ""
        # Defaults mirror config.model: t_index_list [10, 30], seed 2, ControlNet 0.8.
        self._creativity = 1.0 - 30 / 49.0
        self._seed = 2
        self._line_strength = 0.8

    @property
    def name(self) -> str:
        return "mock"

    def warmup(self) -> None:
        time.sleep(min(self._delay_s * 3, 0.2))

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def set_params(self, params: GenParamsLike) -> None:
        t_index = list(params.t_index_list)
        if not t_index:
            raise ValueError("t_index_list must not be empty")
        # Lower first index = more noise added = more "creative" (as in StreamDiffusion).
        self._creativity = 1.0 - min(max(int(t_index[0]), 0), 49) / 49.0
        self._seed = int(params.seed)
        self._line_strength = float(params.controlnet_scale) if params.controlnet_enabled else 0.0

    def _tint(self) -> tuple[int, int, int]:
        """Bright tint whose hue comes from the prompt hash, shifted by the seed."""
        digest = hashlib.sha256(self._prompt.encode("utf-8")).digest()
        hue = (digest[0] / 255.0 + (self._seed % 360) / 360.0) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    def generate(self, init_image: Image.Image, line_map: Image.Image | None) -> Image.Image:
        if self._delay_s:
            time.sleep(self._delay_s)
        img = ImageOps.posterize(init_image.convert("RGB"), 3)
        img = img.filter(ImageFilter.GaussianBlur(0.5 + 4.0 * self._creativity))
        tint = Image.new("RGB", img.size, self._tint())
        img = Image.blend(img, tint, 0.2 + 0.4 * self._creativity)

        # Darken along the drawn lines, scaled by the ControlNet strength, so the
        # sketch structure stays visible like it would with a real ControlNet.
        if line_map is not None and self._line_strength > 0:
            lines = np.asarray(line_map.convert("L").resize(img.size), dtype=np.float32) / 255.0
            if lines.mean() > 0.5:  # black_on_white polarity: lines are dark
                lines = 1.0 - lines
            arr = np.asarray(img, dtype=np.float32)
            k = min(self._line_strength, 2.0) / 2.0 * lines[..., None]
            img = Image.fromarray((arr * (1.0 - 0.7 * k)).clip(0, 255).astype(np.uint8))
        return img

    def check_image_safe(self, image: Image.Image) -> bool:
        # Test hook: a pure magenta image is "unsafe".
        arr = np.asarray(image.convert("RGB"))
        return not bool((arr == np.array(MAGENTA, dtype=arr.dtype)).all())
