"""Small helpers shared by the GPU backends (torch-free, so the backends stay importable)."""
from __future__ import annotations

from typing import Any

from PIL import Image

# The StreamDiffusion wrapper's own default seed (StreamDiffusionWrapper(seed=2)); TCD uses
# the same so both picture models start from the same surprise number.
DEFAULT_SEED = 2


def model_attr(config: Any, name: str, default: Any) -> Any:
    """``config.model.<name>``, or ``default`` when there is no model section or field."""
    return getattr(getattr(config, "model", None), name, default)


def previous_or_black(previous: Any, size: tuple[int, int]) -> Image.Image:
    """What to show instead of a flagged picture of ``size``: the previous safe picture if it
    has that size, else black. A previous picture of another size (made before an orientation
    change, during warm-up, or the fork's initial fallback, which swaps width and height)
    would not fit the drawing."""
    if isinstance(previous, Image.Image) and previous.size == tuple(size):
        return previous
    return Image.new("RGB", tuple(size), "black")
