"""Image helpers for Dreamer: sketch decoding, line maps, blank detection, JPEG encoding.

Pure Pillow + numpy; no torch. All functions are stateless and thread-safe.
"""
from __future__ import annotations

import io

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
# Neutral ground the "colors" and "blank" init images start from.
INIT_BACKGROUND = (240, 240, 240)


def decode_sketch(png_bytes: bytes, size: int) -> Image.Image:
    """Decode a sketch upload into an RGB ``size`` x ``size`` image on a white background.

    Raises ValueError for oversized payloads or data that is not a decodable image.
    """
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

    # Composite any transparency onto white so the canvas background is always white.
    rgba = img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(background, rgba).convert("RGB")
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
    return rgb


def _ink_mask(img: Image.Image, threshold: int) -> np.ndarray:
    """Boolean mask of pixels whose max channel distance from white exceeds threshold."""
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
) -> Image.Image:
    """The image img2img starts from, so the sketch guides the picture without being traced.

    ``sketch`` passes the drawing through unchanged (strokes show up in the picture).
    ``colors`` drops the strokes and keeps only where the child used which colour: each
    coloured stroke becomes a soft cloud of its colour (Gaussian ``blur`` radius) on a
    neutral ground. Dark, unsaturated strokes are outlines and add no colour; the line map
    carries the outline instead. ``blank`` starts from the neutral ground alone.
    """
    if mode not in INIT_MODES:
        raise ValueError(f"unknown init mode: {mode!r}")
    rgb = sketch.convert("RGB")
    if mode == "sketch":
        return rgb
    ground = Image.new("RGB", rgb.size, INIT_BACKGROUND)
    if mode == "blank":
        return ground
    arr = np.asarray(rgb, dtype=np.uint8)
    chroma = arr.max(axis=2).astype(np.int16) - arr.min(axis=2)
    weight = _ink_mask(rgb, threshold) & (chroma >= COLOR_CHROMA)
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
    """True when every pixel is (near) white, i.e. nothing has been drawn."""
    return not _ink_mask(sketch, threshold).any()


def encode_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    """Encode an image as baseline JPEG bytes."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()
