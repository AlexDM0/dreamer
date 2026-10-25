"""Tests for server.imaging."""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from server import imaging


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _sketch(size=64) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(img).line([(10, 32), (54, 32)], fill=(20, 20, 200), width=3)
    return img


def test_decode_resizes_to_square_rgb():
    out = imaging.decode_sketch(_png(_sketch(100)), 64)
    assert out.mode == "RGB" and out.size == (64, 64)


def test_decode_composites_alpha_onto_white():
    transparent = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    out = imaging.decode_sketch(_png(transparent), 32)
    assert np.asarray(out).min() == 255


def test_decode_rejects_oversized_payload():
    with pytest.raises(ValueError):
        imaging.decode_sketch(b"\x89PNG" + b"0" * (imaging.MAX_SKETCH_BYTES + 1), 64)


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\nbroken"])
def test_decode_rejects_garbage(data):
    with pytest.raises(ValueError):
        imaging.decode_sketch(data, 64)


def test_line_map_white_on_black():
    lm = np.asarray(imaging.make_line_map(_sketch(), "white_on_black"))
    assert lm.shape == (64, 64, 3)
    assert (lm[32, 32] == 255).all()  # on the line
    assert (lm[5, 5] == 0).all()  # background
    assert set(np.unique(lm)) <= {0, 255}


def test_line_map_black_on_white():
    lm = np.asarray(imaging.make_line_map(_sketch(), "black_on_white"))
    assert (lm[32, 32] == 0).all()
    assert (lm[5, 5] == 255).all()


def test_line_map_ignores_faint_marks_and_dilates():
    img = Image.new("RGB", (32, 32), "white")
    img.putpixel((5, 5), (230, 230, 230))  # faint: distance 25 < 40
    img.putpixel((20, 20), (0, 0, 0))
    lm = np.asarray(imaging.make_line_map(img, dilate=1).convert("L"))
    assert lm[5, 5] == 0
    assert lm[20, 20] == 255 and lm[21, 21] == 255 and lm[22, 22] == 0
    thin = np.asarray(imaging.make_line_map(img, dilate=0).convert("L"))
    assert thin.sum() == 255


def test_line_map_threshold_param():
    img = Image.new("RGB", (8, 8), (200, 200, 200))  # distance 55
    assert np.asarray(imaging.make_line_map(img, threshold=40, dilate=0)).min() == 255
    assert np.asarray(imaging.make_line_map(img, threshold=80, dilate=0)).max() == 0


def test_line_map_rejects_bad_polarity():
    with pytest.raises(ValueError):
        imaging.make_line_map(_sketch(), "rainbow")


def test_is_blank():
    assert imaging.is_blank(Image.new("RGB", (16, 16), "white"))
    assert imaging.is_blank(Image.new("RGB", (16, 16), (250, 250, 250)))
    assert not imaging.is_blank(_sketch())


def test_encode_jpeg_roundtrip():
    data = imaging.encode_jpeg(_sketch().convert("RGBA"), 80)
    assert data[:2] == b"\xff\xd8"
    out = Image.open(io.BytesIO(data))
    assert out.format == "JPEG" and out.size == (64, 64)


def test_decode_accepts_only_png():
    buf = io.BytesIO()
    _sketch().save(buf, format="JPEG")
    with pytest.raises(ValueError):
        imaging.decode_sketch(buf.getvalue(), 64)
