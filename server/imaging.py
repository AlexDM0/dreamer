"""Image helpers for Dreamer: sketch decoding, line maps, blank detection, JPEG encoding.

A sketch is either a *stroke layer* (RGBA: the child's strokes on transparency, drawn over
a background colour the page reports separately) or, from older pages, an opaque RGB
drawing on white. Ink is "opaque" in a layer and "not white" in an opaque sketch.

Pure Pillow + numpy; no torch. All functions are stateless and thread-safe.
"""
from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image, ImageFilter

MAX_SKETCH_BYTES = 2 * 1024 * 1024
# Refuse absurd dimensions before decoding pixels (decompression-bomb guard).
MAX_SKETCH_PIXELS = 4096 * 4096
# Max channel distance from white above which a pixel counts as "ink".
DEFAULT_LINE_THRESHOLD = 40
# Blank detection is a bit stricter so faint JPEG/antialias noise never counts as drawing.
BLANK_THRESHOLD = 24

POLARITIES = ("white_on_black", "black_on_white")
INIT_MODES = ("colors", "sketch", "blank")
# Chroma (max - min channel) a stroke needs to count as colour rather than an outline.
COLOR_CHROMA = 48
# Neutral ground the "colors" and "blank" init images start from without a background.
INIT_BACKGROUND = (240, 240, 240)
COLOR_RE = re.compile(r"#[0-9a-fA-F]{6}")


def parse_color(value: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> (r, g, b). Raises ValueError for anything else."""
    if not isinstance(value, str) or not COLOR_RE.fullmatch(value):
        raise ValueError(f"not a #rrggbb colour: {value!r}")
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def decode_sketch(png_bytes: bytes, size: int | tuple[int, int],
                  layered: bool = False) -> Image.Image:
    """Decode a sketch upload, resized to ``size`` (an int for a square, or (w, h)).

    ``layered``: the page sends its stroke layer (RGBA, transparent background), returned
    as RGBA. Otherwise the sketch is composited onto white and returned as RGB.
    Raises ValueError for oversized payloads or data that is not a decodable image.
    """
    dims = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    if not png_bytes:
        raise ValueError("empty sketch payload")
    if len(png_bytes) > MAX_SKETCH_BYTES:
        raise ValueError(f"sketch payload too large ({len(png_bytes)} bytes)")
    try:
        # PNG only (the protocol says so): never expose other Pillow decoders to the
        # unauthenticated kid socket.
        img = Image.open(io.BytesIO(png_bytes), formats=["PNG"])
        if img.width * img.height > MAX_SKETCH_PIXELS:
            raise ValueError(f"sketch dimensions too large ({img.width}x{img.height})")
        img.load()
    except ValueError:
        raise
    except Exception as exc:  # PIL raises a variety of types for garbage input
        raise ValueError(f"not a valid image: {exc}") from exc

    rgba = img.convert("RGBA")
    if layered:
        # Pillow resizes RGBA premultiplied, so transparent pixels do not bleed colour.
        return rgba if rgba.size == dims else rgba.resize(dims, Image.Resampling.LANCZOS)
    # Composite any transparency onto white so the canvas background is always white.
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(background, rgba).convert("RGB")
    if rgb.size != dims:
        rgb = rgb.resize(dims, Image.Resampling.LANCZOS)
    return rgb


def compose(sketch: Image.Image, background: str | None) -> Image.Image:
    """The drawing as the child sees it: a stroke layer over its background (RGB)."""
    if sketch.mode != "RGBA":
        return sketch.convert("RGB")
    ground = Image.new("RGBA", sketch.size, (*parse_color(background or "#ffffff"), 255))
    return Image.alpha_composite(ground, sketch).convert("RGB")


def _channel_extremes(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (max, min) over the RGB channels of an HxWx3 uint8 array. Channel by
    channel: ``arr.max(axis=2)`` walks the array with a 3-byte stride and is several
    times slower on a 640x448 frame."""
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.maximum(np.maximum(red, green), blue), np.minimum(np.minimum(red, green), blue)


def _ink_mask(img: Image.Image, threshold: int) -> np.ndarray:
    """Boolean mask of drawn pixels: alpha above ``threshold`` in a stroke layer, else
    max channel distance from white above ``threshold``."""
    if img.mode == "RGBA":
        return np.asarray(img.getchannel("A"), dtype=np.uint8) > threshold
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    # Max distance from white over channels == 255 - min channel value.
    return (255 - _channel_extremes(arr)[1]) > threshold


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow True regions by ``radius`` pixels (square kernel, like MaxFilter(2r+1)), with
    shifted ORs: pixel-identical to Pillow's MaxFilter and ~30x faster at 640x448."""
    out = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            out[max(0, dy):height + min(0, dy), max(0, dx):width + min(0, dx)] |= \
                mask[max(0, -dy):height + min(0, -dy), max(0, -dx):width + min(0, -dx)]
    return out


def make_line_map(
    sketch: Image.Image,
    polarity: str = "white_on_black",
    threshold: int = DEFAULT_LINE_THRESHOLD,
    dilate: int = 1,
) -> Image.Image:
    """Turn a sketch into a binary line map for ControlNet.

    ``white_on_black`` (default) gives white lines on black; ``black_on_white`` the inverse.
    ``dilate`` thickens lines by that many pixels (0 disables).
    """
    if polarity not in POLARITIES:
        raise ValueError(f"unknown line polarity: {polarity!r}")
    ink = _ink_mask(sketch, threshold)
    if dilate > 0:
        ink = _dilate(ink, int(dilate))
    if polarity == "black_on_white":
        ink = ~ink
    return Image.fromarray(ink.astype(np.uint8) * 255, mode="L").convert("RGB")


def make_init_image(
    sketch: Image.Image,
    mode: str = "colors",
    blur: int = 24,
    threshold: int = DEFAULT_LINE_THRESHOLD,
    background: str | None = None,
) -> Image.Image:
    """The image img2img starts from, so the sketch guides the picture without being traced.

    ``sketch`` passes the drawing through unchanged (strokes show up in the picture).
    ``colors`` drops the strokes and keeps only where the child used which colour: each
    coloured stroke becomes a soft cloud of its colour (Gaussian ``blur`` radius) on a
    neutral ground. Dark, unsaturated strokes are outlines and add no colour; the line map
    carries the outline instead. ``blank`` starts from the neutral ground alone.
    With a ``background`` ("#rrggbb", the page's background layer) that colour is the
    ground, so the background choice carries into the picture.
    """
    if mode not in INIT_MODES:
        raise ValueError(f"unknown init mode: {mode!r}")
    if mode == "sketch":
        return compose(sketch, background)
    ink = _ink_mask(sketch, threshold)
    rgb = sketch.convert("RGB")  # a layer's stroke colours (alpha dropped)
    ground_rgb = parse_color(background) if background else INIT_BACKGROUND
    ground = Image.new("RGB", rgb.size, ground_rgb)
    if mode == "blank":
        return ground
    arr = np.asarray(rgb, dtype=np.uint8)
    high, low = _channel_extremes(arr)
    weight = ink & ((high - low) >= COLOR_CHROMA)  # uint8: high >= low, no wrap-around
    if not weight.any():
        return ground
    # The colour clouds are soft, so blur at a quarter of the size (up to; the radius stays
    # at least 4 px there) and scale back up: same look (mean difference under 1/255), a
    # fraction of the time of two full-size wide Gaussian blurs.
    radius = max(1, int(blur))
    factor = max(1, min(4, radius // 4))
    small = (max(1, rgb.width // factor), max(1, rgb.height // factor))
    w8 = Image.fromarray(weight.astype(np.uint8) * 255, mode="L")
    premul = Image.fromarray(arr * weight[..., None].astype(np.uint8), mode="RGB")
    if factor > 1:
        w8 = w8.resize(small, Image.Resampling.BOX)
        premul = premul.resize(small, Image.Resampling.BOX)
    w8 = w8.filter(ImageFilter.GaussianBlur(radius / factor))
    premul = premul.filter(ImageFilter.GaussianBlur(radius / factor))
    wsum = np.asarray(w8, dtype=np.float32) * (1.0 / 255.0)
    colour = np.asarray(premul, dtype=np.float32) / np.maximum(wsum, 1e-3)[..., None]
    # Thin strokes spread thin: boost coverage so a coloured area reads as that colour.
    alpha = np.clip(wsum * 4.0, 0.0, 1.0)[..., None]
    out = colour * alpha + np.asarray(ground_rgb, dtype=np.float32) * (1.0 - alpha)
    picture = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
    # Everything here is smooth, so the small result scales up without visible loss.
    return picture.resize(rgb.size, Image.Resampling.BILINEAR) if factor > 1 else picture


def is_blank(sketch: Image.Image, threshold: int = BLANK_THRESHOLD) -> bool:
    """True when nothing has been drawn: a layer is (near) transparent everywhere, an
    opaque sketch (near) white."""
    return not _ink_mask(sketch, threshold).any()


def encode_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    """Encode an image as baseline JPEG bytes."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()
