"""Backend factory. Heavy backends are imported lazily so tests never need torch."""
from __future__ import annotations

from .base import Backend, GenParamsLike
from .mock import MockBackend

__all__ = ["Backend", "GenParamsLike", "MockBackend", "make_backend"]


def make_backend(config, picture_model: str = "turbo") -> Backend:
    """Build the backend named by ``config.backend`` ("mock" or "streamdiffusion"); with
    the real backend, ``picture_model`` "tcd" picks SD 1.5 + TCD instead of SD-Turbo.

    Construction is cheap; models load in ``warmup()`` on the GPU worker thread.
    """
    kind = (getattr(config, "backend", "mock") or "mock").lower()
    if kind == "mock":
        return MockBackend(mock_delay_ms=getattr(config, "mock_delay_ms", 30))
    if kind == "streamdiffusion" and picture_model == "tcd":
        from .tcd import TcdBackend

        return TcdBackend(config)
    if kind == "streamdiffusion":
        from .streamdiffusion import StreamDiffusionBackend

        return StreamDiffusionBackend(config)
    raise ValueError(f"unknown backend {kind!r} (expected 'mock' or 'streamdiffusion')")
