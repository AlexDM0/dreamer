"""AI upscaler for refined pictures: Real-ESRGAN's compact network (SRVGGNetCompact).

The network is re-implemented here (it is a plain stack of convolutions, PReLUs and a pixel
shuffle) so no extra package is needed; architecture and weights are from Real-ESRGAN by
Xintao Wang et al., https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause). The default
weights, realesr-animevideov3.pth, are a 4x model trained on anime frames, which suits flat,
cartoon-like pictures; the output is resized to the wanted factor (Real-ESRGAN's "outscale").

Imported and run only on the GPU worker thread; torch is imported lazily.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger("dreamer.upscale")


def _build_network(state: dict[str, Any]) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as F

    conv_keys = sorted((int(key.split(".")[1]) for key in state if key.endswith(".weight")
                        and state[key].dim() == 4), key=int)
    first, last = state[f"body.{conv_keys[0]}.weight"], state[f"body.{conv_keys[-1]}.weight"]
    num_feat, num_in = first.shape[0], first.shape[1]
    num_conv = len(conv_keys) - 2
    scale = int(round((last.shape[0] / num_in) ** 0.5))

    class SRVGGNetCompact(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = scale
            layers: list[nn.Module] = [nn.Conv2d(num_in, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
            for _ in range(num_conv):
                layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
            layers.append(nn.Conv2d(num_feat, num_in * scale * scale, 3, 1, 1))
            self.body = nn.ModuleList(layers)
            self.upsampler = nn.PixelShuffle(scale)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = x
            for layer in self.body:
                out = layer(out)
            # The network predicts the detail on top of a nearest-neighbour enlargement.
            return self.upsampler(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")

    network = SRVGGNetCompact()
    network.load_state_dict(state, strict=True)
    return network


class Upscaler:
    """``upscale(picture, factor)`` with a Real-ESRGAN compact model, loaded on first use."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._network: Any = None

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _load(self) -> Any:
        if self._network is None:
            import torch

            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
            network = _build_network(state)
            self._network = network.to("cuda", dtype=torch.float16).eval()
            self._network = self._network.to(memory_format=torch.channels_last)
            log.info("upscaler loaded: %s (x%d)", self.model_path.name, network.scale)
        return self._network

    def upscale(self, picture: Image.Image, factor: int = 2) -> Image.Image:
        import numpy as np
        import torch
        from torch.nn import functional as F

        network = self._load()
        rgb = np.asarray(picture.convert("RGB"))
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to("cuda", dtype=torch.float16).div_(255).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            out = network(tensor)
            size = (picture.height * factor, picture.width * factor)
            if out.shape[-2:] != size:
                out = F.interpolate(out, size=size, mode="area")  # 4x model -> wanted factor
            out = out.clamp_(0, 1).mul_(255).round_().to(torch.uint8)
        return Image.fromarray(out[0].permute(1, 2, 0).cpu().numpy())
