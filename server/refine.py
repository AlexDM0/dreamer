"""Refinement models: what happens to the picture once the child stops drawing.

The owner's rule: the picture the child first sees must not change in broad terms; each step
should only add detail. Every model here is a generator that yields the pictures to show, one
per refinement step, given:

- ``backend``: the StreamDiffusion backend (``refine_steps`` runs one denoising path step by
  step; see there),
- ``first``: the drawing picture already on screen,
- ``init`` / ``line_map``: the drawing's img2img start image and ControlNet line map,
- ``draw_t_index``: the draw steps; ``total_steps``: the refine step budget (params).

Pick one with params.refine_model; tools/refine_lab.py measures them all (broad change vs
detail added) and makes comparison sheets.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image, ImageFilter

T_INDEX_MAX = 49


def _spread(first: int, steps: int) -> tuple[int, ...]:
    """``steps`` t_index values from ``first`` to the last one, evenly spaced."""
    steps = max(1, min(steps, T_INDEX_MAX - first + 1))
    if steps == 1:
        return (first,)
    return tuple(round(first + i * (T_INDEX_MAX - first) / (steps - 1)) for i in range(steps))


def continue_path(draw: tuple[int, ...], total_steps: int) -> tuple[int, ...] | None:
    """The draw steps, then the rest of ``total_steps`` evenly spaced after the last draw step
    up to the last t_index (draw 1 + 3 more == the evenly spread 4-step path). None if there
    is nothing to add."""
    last = draw[-1]
    extra = min(total_steps - len(draw), T_INDEX_MAX - last)
    if extra <= 0:
        return None
    more = tuple(round(last + i * (T_INDEX_MAX - last) / extra) for i in range(1, extra + 1))
    return tuple(draw) + more


# ---- building blocks ------------------------------------------------------------------


def _path(backend: Any, init: Image.Image, line_map: Image.Image | None,
          draw: tuple[int, ...], total_steps: int, noise: str) -> Iterator[Image.Image]:
    path = continue_path(draw, total_steps)
    if path is None:
        return
    yield from backend.refine_steps(init, line_map, list(path), len(draw) + 1, noise=noise)


def _img2img(backend: Any, image: Image.Image, line_map: Image.Image | None,
             t_index: tuple[int, ...]) -> Image.Image:
    """One complete low-noise pass from ``image`` (steps one after another)."""
    picture = None
    for picture in backend.refine_steps(image, line_map, list(t_index), len(t_index)):
        pass
    return picture


def lock_broad_terms(first: Image.Image, picture: Image.Image, radius: float) -> Image.Image:
    """Keep ``first``'s broad terms (its blurred version: layout, colours, light) and take
    only the fine detail (picture minus its blurred version) from ``picture``."""
    if picture.size != first.size:
        first = first.resize(picture.size, Image.Resampling.BICUBIC)
    radius = radius * picture.width / 640
    base = np.asarray(first.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    new = np.asarray(picture.convert("RGB"), dtype=np.float32)
    detail = new - np.asarray(picture.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    return Image.fromarray(np.clip(base + detail, 0, 255).astype(np.uint8))


def _locked(pictures: Iterator[Image.Image], first: Image.Image,
            radius: float = 8.0) -> Iterator[Image.Image]:
    for picture in pictures:
        yield lock_broad_terms(first, picture, radius)


def edge_map(picture: Image.Image, threshold: float = 48.0) -> Image.Image:
    """Thin white-on-black edges of a picture (Sobel), for locking outlines via ControlNet."""
    gray = np.asarray(picture.convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.hypot(gx, gy)
    return Image.fromarray(((magnitude > threshold) * 255).astype(np.uint8)).convert("RGB")


def _union(first: Image.Image, second: Image.Image) -> Image.Image:
    if second.size != first.size:
        second = second.resize(first.size, Image.Resampling.NEAREST)
    return Image.fromarray(np.maximum(np.asarray(first.convert("RGB")), np.asarray(second.convert("RGB"))))


# Rungs of the latent ladders: single steps at rising t_index (less noise each time), so each
# rung can only change less than the one before (research: ILVR/TCD-style; see docs).
ANCHORED_RUNGS = (34, 39, 43, 45, 47, 48)
SDEDIT_RUNGS = (36, 40, 44, 47)
ANCHOR_SIGMA = 2.5   # latent pixels (about 20 image pixels at 640x448)
ANCHOR_GAMMA = 0.25  # share of fresh noise when re-noising
SDEDIT_GAMMA = 0.15


def _rungs(schedule: tuple[int, ...], draw: tuple[int, ...], total_steps: int) -> list[int]:
    count = max(1, min(len(schedule), total_steps - len(draw)))
    return list(schedule[:count])


def model_sdedit(backend, first, init, line_map, draw, total_steps):
    """Low-noise passes that each start again from the first picture (steady noise): cannot
    drift, but detail does not build up from one picture to the next."""
    yield from backend.latent_ladder(first, line_map, _rungs(SDEDIT_RUNGS, draw, total_steps),
                                     chain=False, anchor_sigma=None, gamma=SDEDIT_GAMMA)


def model_anchored(backend, first, init, line_map, draw, total_steps):
    """Each rung refines the previous one with less noise, while its broad terms (low
    frequencies of the latent) are pinned to the first picture: detail builds up, layout,
    colours and light stay."""
    rungs = _rungs(ANCHORED_RUNGS, draw, total_steps)
    weights = [1.0 - 0.2 * i / max(1, len(rungs) - 1) for i in range(len(rungs))]
    yield from backend.latent_ladder(first, line_map, rungs, chain=True, anchor_sigma=ANCHOR_SIGMA,
                                     gamma=ANCHOR_GAMMA, anchor_weights=weights)


def model_anchored_edges(backend, first, init, line_map, draw, total_steps):
    """Anchored, and the ControlNet also sees the first picture's outlines (with the child's
    lines), so object shapes stay where they are too."""
    rungs = _rungs(ANCHORED_RUNGS, draw, total_steps)
    weights = [1.0 - 0.2 * i / max(1, len(rungs) - 1) for i in range(len(rungs))]
    control = edge_map(first) if line_map is None else _union(line_map, edge_map(first))
    yield from backend.latent_ladder(first, line_map, rungs, chain=True, anchor_sigma=ANCHOR_SIGMA,
                                     gamma=ANCHOR_GAMMA, anchor_weights=weights,
                                     control_images=[control] * len(rungs))


# Hires in tiles: upscale 2x, then single anchored low-noise steps over overlapping tiles of the
# drawing's own size (the TensorRT engines are dynamic only up to 1024 px, and a bigger
# pipeline would need a rebuild), feather-blended. Real detail at twice the resolution while
# every tile keeps its own broad terms.
TILE_RUNGS = (40, 44, 47)
TILE_SIGMA = 2.5
TILE_SCALE = 1.5  # 960x672 from 640x448: 4 tiles (2x needs 9, ~600 ms per rung)


def _feather(width: int, height: int, overlap_x: int, overlap_y: int) -> np.ndarray:
    def ramp(length: int, overlap: int) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        if overlap > 0:
            edge = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32))
            weights[:overlap] = np.maximum(edge, 1e-3)
            weights[-overlap:] = np.maximum(edge[::-1], 1e-3)
        return weights
    return np.outer(ramp(height, overlap_y), ramp(width, overlap_x))[..., None]


def _tiled_pass(backend: Any, picture: Image.Image, line_map: Image.Image | None,
                rung: int, tile_size: tuple[int, int], anchor: Image.Image) -> Image.Image:
    tile_w, tile_h = tile_size
    width, height = picture.size
    columns = max(1, -(-(width - tile_w) // (tile_w // 2)) + 1)
    rows = max(1, -(-(height - tile_h) // (tile_h // 2)) + 1)
    xs = [round(i * (width - tile_w) / max(1, columns - 1)) for i in range(columns)]
    ys = [round(i * (height - tile_h) / max(1, rows - 1)) for i in range(rows)]
    overlap_x = tile_w - (xs[1] - xs[0]) if len(xs) > 1 else 0
    overlap_y = tile_h - (ys[1] - ys[0]) if len(ys) > 1 else 0
    weight = _feather(tile_w, tile_h, overlap_x, overlap_y)
    total = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    big_map = line_map.resize((width, height), Image.Resampling.NEAREST) if line_map is not None else None
    for y in ys:
        for x in xs:
            box = (x, y, x + tile_w, y + tile_h)
            tile = picture.crop(box)
            tile_map = big_map.crop(box) if big_map is not None else None
            out = None
            # Broad terms pinned to the (upscaled) first picture, so drift cannot build up.
            for out in backend.latent_ladder(tile, tile_map, [rung], chain=True,
                                             anchor_sigma=TILE_SIGMA, gamma=ANCHOR_GAMMA,
                                             anchor_image=anchor.crop(box)):
                pass
            total[y:y + tile_h, x:x + tile_w] += np.asarray(out, dtype=np.float32) * weight
            weights[y:y + tile_h, x:x + tile_w] += weight
    return Image.fromarray(np.clip(total / weights, 0, 255).astype(np.uint8))


def model_hires_tiles(backend, first, init, line_map, draw, total_steps, upscaler=None):
    """Upscale (1.5x), then add detail tile by tile with low-noise steps (gentler each rung),
    every tile's broad terms pinned to the upscaled first picture."""
    size = (round(first.width * TILE_SCALE / 8) * 8, round(first.height * TILE_SCALE / 8) * 8)
    if upscaler is not None and upscaler.available:
        anchor = upscaler.upscale(first, 2).resize(size, Image.Resampling.LANCZOS)
    else:
        anchor = first.resize(size, Image.Resampling.LANCZOS)
    picture = anchor
    for rung in _rungs(TILE_RUNGS, draw, total_steps):
        picture = _tiled_pass(backend, picture, line_map, rung, first.size, anchor)
        yield picture


# ---- the models -----------------------------------------------------------------------

def model_path(backend, first, init, line_map, draw, total_steps):
    """Continue the drawing's own denoising path step by step (fresh noise each step)."""
    yield from _path(backend, init, line_map, draw, total_steps, "fresh")


def model_path_locked(backend, first, init, line_map, draw, total_steps):
    """The path, with the broad terms pinned to the first picture in pixel space."""
    yield from _locked(_path(backend, init, line_map, draw, total_steps, "fresh"), first)


# name -> (generator, label shown on the admin page, makes pictures bigger itself)
MODELS: dict[str, tuple[Callable[..., Iterator[Image.Image]], str, bool]] = {
    "path": (model_path, "A: Continue the path", False),
    "path_locked": (model_path_locked, "B: Continue the path, shapes locked", False),
    "sdedit": (model_sdedit, "C: Gentle touch-ups from the first picture", False),
    "anchored": (model_anchored, "D: Anchored detail ladder", False),
    "anchored_edges": (model_anchored_edges, "E: Anchored detail ladder + outlines", False),
    "hires_tiles": (model_hires_tiles, "F: Bigger, detail tile by tile", True),
}
DEFAULT_MODEL = "path"


def run(name: str, backend: Any, first: Image.Image, init: Image.Image,
        line_map: Image.Image | None, draw: tuple[int, ...], total_steps: int,
        upscaler: Any = None) -> Iterator[Image.Image]:
    generator, _, upscales_itself = MODELS.get(name, MODELS[DEFAULT_MODEL])
    if upscales_itself:
        return generator(backend, first, init, line_map, tuple(draw), total_steps, upscaler=upscaler)
    return generator(backend, first, init, line_map, tuple(draw), total_steps)
