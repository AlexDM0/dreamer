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


def _ink_mask(img: Image.Image, threshold: int) -> np.ndarray:
    """Boolean mask of drawn pixels: alpha above ``threshold`` in a stroke layer, else
    max channel distance from white above ``threshold``."""
    if img.mode == "RGBA":
        return np.asarray(img.getchannel("A"), dtype=np.uint8) > threshold
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    # Max distance from white over channels == 255 - min channel value.
    distance = 255 - arr.min(axis=2)
    return distance > threshold


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
    mask = Image.fromarray(_ink_mask(sketch, threshold).astype(np.uint8) * 255, mode="L")
    if dilate > 0:
        # MaxFilter with kernel 2r+1 grows white regions by r pixels.
        mask = mask.filter(ImageFilter.MaxFilter(2 * int(dilate) + 1))
    if polarity == "black_on_white":
        mask = Image.eval(mask, lambda v: 255 - v)
    return mask.convert("RGB")


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
    chroma = arr.max(axis=2).astype(np.int16) - arr.min(axis=2)
    weight = ink & (chroma >= COLOR_CHROMA)
    if not weight.any():
        return ground
    radius = max(1, int(blur))
    w8 = Image.fromarray(weight.astype(np.uint8) * 255, mode="L")
    premul = Image.fromarray(arr * weight[..., None].astype(np.uint8), mode="RGB")
    wsum = np.asarray(w8.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    csum = np.asarray(premul.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    colour = csum / np.maximum(wsum, 1e-3)[..., None]
    # Thin strokes spread thin: boost coverage so a coloured area reads as that colour.
    alpha = np.clip(wsum * 4.0, 0.0, 1.0)[..., None]
    out = colour * alpha + np.asarray(ground, dtype=np.float32) * (1.0 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def is_blank(sketch: Image.Image, threshold: int = BLANK_THRESHOLD) -> bool:
    """True when nothing has been drawn: a layer is (near) transparent everywhere, an
    opaque sketch (near) white."""
    return not _ink_mask(sketch, threshold).any()


def encode_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    """Encode an image as baseline JPEG bytes."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()
