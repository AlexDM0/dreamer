"""Background layer + orientation: the {"t":"canvas"} message, stroke-layer sketches,
picture sizes, and what the gallery keeps (stroke layer + background, never merged)."""
import io
import time

import numpy as np
from conftest import is_json, is_picture, recv_until
from PIL import Image, ImageDraw

from server import imaging, protocol
from server.gallery import is_blank_sketch


def layer_png(width=384, height=256, color=(255, 0, 0, 255), blank=False) -> bytes:
    """A page's stroke layer: strokes on transparency."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not blank:
        ImageDraw.Draw(img).line((20, 20, width - 20, height - 40), fill=color, width=8)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def hello(ws):
    ws.send_json({"t": "hello", "v": 1})
    return recv_until(ws, is_json("ready"))[1]


def picture(ws, seq, png):
    ws.send_bytes(protocol.pack(protocol.SKETCH, seq, png))
    return Image.open(io.BytesIO(recv_until(ws, is_picture(seq))[1][2]))


def test_ready_announces_drawing_sizes(client):
    with client.websocket_connect("/ws") as ws:
        ready = hello(ws)
    assert ready["canvas"] == {"landscape": [384, 256], "portrait": [256, 384]}


def test_portrait_layer_gives_portrait_picture_on_its_background(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "canvas", "background": "#000000", "orientation": "portrait"})
        img = picture(ws, 1, layer_png(256, 384))
        assert img.size == (256, 384)
        # A blank layer is answered with the background colour, not white.
        blank = picture(ws, 2, layer_png(256, 384, blank=True)).convert("RGB")
        assert blank.size == (256, 384) and max(blank.getpixel((10, 10))) < 30


def test_bad_canvas_messages_are_ignored(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        for bad in ({"background": "red", "orientation": "portrait"},
                    {"background": "#000000", "orientation": "upside-down"},
                    {"background": 5}):
            ws.send_json({"t": "canvas", **bad})
        img = picture(ws, 1, layer_png(384, 256))
    assert img.size == (384, 256)  # still the landscape default
    assert client.app.state.session.canvas is None


def test_gallery_keeps_layer_and_background(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "canvas", "background": "#112233", "orientation": "portrait"})
        stroke_layer = layer_png(256, 384, color=(255, 255, 255, 255))  # white on dark
        picture(ws, 1, stroke_layer)
        ws.send_json({"t": "save"})
        art = recv_until(ws, is_json("saved"))[1]["artwork"]
        assert art["background"] == "#112233" and art["orientation"] == "portrait"
        assert client.get(art["sketch"]).content == stroke_layer  # the layer, unmerged
        ws.send_json({"t": "continue", "id": art["id"]})
        msg = recv_until(ws, is_json("continue"))[1]
    assert msg["background"] == "#112233" and msg["orientation"] == "portrait"


def test_draft_carries_the_canvas(client):
    client.app.state.runtime.drafts.interval = 0.05
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "canvas", "background": "#ffd84d", "orientation": "landscape"})
        picture(ws, 1, layer_png())
        end = time.monotonic() + 5
        while not client.app.state.gallery.drafts() and time.monotonic() < end:
            time.sleep(0.02)
    with client.websocket_connect("/ws") as ws:
        draft = hello(ws)["draft"]
    assert draft["background"] == "#ffd84d" and draft["orientation"] == "landscape"


# ----- imaging ------------------------------------------------------------------------


def test_layer_ink_comes_from_alpha():
    white_strokes = imaging.decode_sketch(layer_png(256, 192, color=(255, 255, 255, 255)), (256, 192),
                                          layered=True)
    assert white_strokes.mode == "RGBA" and not imaging.is_blank(white_strokes)
    line_map = np.asarray(imaging.make_line_map(white_strokes).convert("L"))
    assert line_map.max() == 255 and line_map.min() == 0  # white strokes are lines too
    empty = imaging.decode_sketch(layer_png(256, 192, blank=True), (256, 192), layered=True)
    assert imaging.is_blank(empty)


def test_init_image_uses_the_background():
    layer = imaging.decode_sketch(layer_png(256, 192, color=(255, 255, 255, 255)), (256, 192), layered=True)
    for mode in ("colors", "blank"):  # white strokes are outlines: only the ground remains
        init = imaging.make_init_image(layer, mode, background="#000080")
        assert init.getpixel((5, 5)) == (0, 0, 128)
    composed = imaging.make_init_image(layer, "sketch", background="#000080")
    assert composed.getpixel((5, 5)) == (0, 0, 128) and composed.getpixel((24, 21)) == (255, 255, 255)
    assert imaging.compose(layer, "#000080").size == (256, 192)


def test_is_blank_sketch_handles_layers():
    assert is_blank_sketch(layer_png(blank=True))
    assert not is_blank_sketch(layer_png(color=(255, 255, 255, 255)))
