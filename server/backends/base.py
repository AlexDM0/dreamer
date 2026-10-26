"""Backend protocol: the interface the GPU worker uses to make pictures.

All methods are called ONLY from the GPU worker thread.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from PIL import Image

# The StreamDiffusion wrapper's own default seed (StreamDiffusionWrapper(seed=2)); the TCD
# backend uses the same so both picture models start from the same "surprise number".
DEFAULT_SEED = 2


def model_attr(config: Any, name: str, default: Any) -> Any:
    """``config.model.<name>``, or ``default`` (tests pass partial configs)."""
    return getattr(getattr(config, "model", None), name, default)


class GenParamsLike(Protocol):
    """Structural view of ``GenParams`` (see SPEC addendum) that backends rely on.

    Backends read only these attributes, so any object exposing them works.
    """

    t_index_list: Sequence[int]
    seed: int
    controlnet_enabled: bool
    controlnet_scale: float
    safety_checker: bool
    safety_threshold: float


@runtime_checkable
class Backend(Protocol):
    def warmup(self) -> None:
        """Load models / build engines. Called once on the worker thread."""

    def set_prompt(self, prompt: str) -> None:
        """Set the prompt used for subsequent frames."""

    def set_params(self, params: GenParamsLike) -> None:
        """Apply live generation params before the next frame. Raise on failure;
        the backend must keep its previous params in that case."""

    def generate(self, init_image: Image.Image, line_map: Image.Image | None) -> Image.Image:
        """Make a picture from the sketch (and optional ControlNet line map)."""

    def check_image_safe(self, image: Image.Image) -> bool:
        """Screen an admin-uploaded image. True means allowed."""

    @property
    def name(self) -> str:
        """Short backend identifier, e.g. "mock" or "streamdiffusion"."""

    # Optional: ``latency_frames`` (int, default 0) is how many generate() calls later a
    # sketch's picture comes out (StreamDiffusion's denoising batch: steps - 1). The worker
    # then primes a fresh pipeline and re-runs the last sketch once drawing pauses.
