"""Render the example pictures for the admin help page (static/help.html) on the GPU.

One demo drawing (coloured strokes on the default black background) goes through the same
steps as the server worker; for each picture control a few values are rendered, so the
help page shows what the control really does. Output: static/help/*.jpg + static/help/data.json.

    .venv\\Scripts\\python.exe tools\\make_help_images.py

Needs the StreamDiffusion backend (config.yaml) and its engines; takes a minute or two.
"""
from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from server import imaging  # noqa: E402
from server.config import canvas_dims, load_config  # noqa: E402
from server.params import defaults_from_config  # noqa: E402
from server.styles import PRESET_STYLES, PRESET_TOPICS, PROMPT_HEAD, SAFETY_TAIL  # noqa: E402

OUT_DIR = REPO_ROOT / "static" / "help"
THUMB_WIDTH = 360
BACKGROUND = "#000000"
TOPIC, STYLE = "nature", "storybook"


def demo_layer(width: int, height: int) -> Image.Image:
    """A child's drawing as a stroke layer: white outlines, a few coloured scribbles."""
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = width / 640
    line = int(7 * scale)

    def pts(*coords: float) -> list[float]:
        return [value * scale for value in coords]

    draw.rectangle(pts(170, 210, 360, 380), outline=(255, 255, 255, 255), width=line)   # house
    draw.line(pts(150, 220, 265, 120, 380, 220), fill=(235, 64, 52, 255), width=line)    # red roof
    draw.rectangle(pts(240, 300, 290, 380), outline=(255, 255, 255, 255), width=line)   # door
    draw.ellipse(pts(470, 50, 570, 150), outline=(255, 214, 10, 255), width=line)       # sun
    for offset in range(0, 120, 30):                                                     # sun rays
        draw.line(pts(450 - offset // 3, 170 + offset // 6, 420 - offset // 2, 200 + offset // 4),
                  fill=(255, 214, 10, 255), width=line // 2)
    draw.line(pts(40, 400, 600, 405), fill=(52, 199, 89, 255), width=line)               # grass
    for x in range(60, 600, 45):
        draw.line(pts(x, 405, x + 10, 370), fill=(52, 199, 89, 255), width=line // 2)
    draw.ellipse(pts(420, 260, 520, 360), outline=(255, 255, 255, 255), width=line)     # tree crown
    draw.line(pts(470, 360, 470, 400), fill=(156, 102, 31, 255), width=line)             # trunk
    return layer


def save_thumb(image: Image.Image, name: str) -> str:
    image = image.convert("RGB")
    height = round(image.height * THUMB_WIDTH / image.width)
    image.resize((THUMB_WIDTH, height), Image.Resampling.LANCZOS).save(
        OUT_DIR / name, "JPEG", quality=82, optimize=True)
    return f"help/{name}"


def main() -> int:
    from server.backends.streamdiffusion import StreamDiffusionBackend

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(None)
    width, height = canvas_dims(config)["landscape"]
    defaults = defaults_from_config(config)
    topics = {topic_id: fragment for topic_id, _, _, fragment in PRESET_TOPICS}
    styles = {style_id: fragment for style_id, _, _, fragment in PRESET_STYLES}

    backend = StreamDiffusionBackend(config)
    backend.set_prompt(", ".join([PROMPT_HEAD, topics[TOPIC], styles[STYLE], SAFETY_TAIL]))
    backend.warmup()

    layer = demo_layer(width, height)
    data: dict = {"defaults": defaults.to_dict(), "size": [width, height], "topic": TOPIC,
                  "style": STYLE, "background": BACKGROUND, "controls": {}}
    data["drawing"] = save_thumb(imaging.compose(layer, BACKGROUND), "drawing.jpg")

    def render(params, name: str, timed: bool = False) -> dict:
        backend.set_params(params)
        # Re-seed every example: some changes (e.g. the step count) re-roll the noise, and
        # the default picture should look the same in every row of the help page.
        backend._wrapper.update_stream_params(seed=int(params.seed))
        line_map = None
        if params.controlnet_enabled:
            line_map = imaging.make_line_map(layer, params.line_polarity, params.line_threshold,
                                             params.line_dilate)
            init = imaging.make_init_image(layer, params.init_mode, params.init_blur,
                                           params.line_threshold, BACKGROUND)
        else:
            init = imaging.compose(layer, BACKGROUND)
        for _ in range(backend.latency_frames + 1):  # prime: the picture for this drawing
            picture = backend.generate(init, line_map)
        entry = {"picture": save_thumb(picture, f"{name}.jpg")}
        if timed:
            started = time.perf_counter()
            for _ in range(20):
                backend.generate(init, line_map)
            entry["ms"] = round((time.perf_counter() - started) * 1000 / 20)
        return entry

    def sweep(control: str, values: list, make_params, label=str, timed=False, extra=None) -> None:
        items = []
        for value in values:
            params = make_params(value)
            entry = render(params, f"{control}-{str(value).replace('.', '_').replace(',', '-')}", timed)
            entry["value"] = label(value)
            entry["default"] = params == defaults
            if extra is not None:
                entry.update(extra(value, params))
            items.append(entry)
            print(control, value, entry.get("ms", ""), flush=True)
        data["controls"][control] = items

    replace = dataclasses.replace
    steps_default = len(defaults.t_index_list)

    def with_first_index(first: int):
        rest = [min(49, first + 20 * i) for i in range(1, steps_default)]
        return replace(defaults, t_index_list=tuple([first] + rest))

    sweep("creativity", [30, 18, defaults.t_index_list[0], 2],
          with_first_index, label=lambda first: f"{round((1 - first / 49) * 100)}%")
    first = defaults.t_index_list[0]
    sweep("steps", [1, 2, 4, 5, 6, 8],
          lambda steps: replace(defaults, t_index_list=tuple(
              [first] if steps == 1 else
              [round(first + i * (min(49, first + 20 * (steps - 1)) - first) / (steps - 1))
               for i in range(steps)])),
          timed=True,
          extra=lambda steps, params: {"mode": "together" if backend._batched(list(params.t_index_list))
                                       else "one by one"})
    sweep("lines", ["off", 0.3, defaults.controlnet_scale, 0.9, 1.5],
          lambda value: replace(defaults, controlnet_enabled=False) if value == "off"
          else replace(defaults, controlnet_scale=float(value)))
    sweep("seed", [defaults.seed, 7, 42, 1234], lambda seed: replace(defaults, seed=int(seed)))

    def line_map_extra(name):
        def extra(value, params):
            line_map = imaging.make_line_map(layer, params.line_polarity, params.line_threshold,
                                             params.line_dilate)
            return {"linemap": save_thumb(line_map, f"{name}-{str(value).replace('_', '-')}-map.jpg")}
        return extra

    sweep("line_threshold", [10, defaults.line_threshold, 120],
          lambda value: replace(defaults, line_threshold=int(value)), extra=line_map_extra("line_threshold"))
    sweep("line_dilate", [0, defaults.line_dilate, 3],
          lambda value: replace(defaults, line_dilate=int(value)), extra=line_map_extra("line_dilate"))
    sweep("line_polarity", ["white_on_black", "black_on_white"],
          lambda value: replace(defaults, line_polarity=value),
          label=lambda value: value.replace("_", " "), extra=line_map_extra("line_polarity"))

    def init_extra(value, params):
        init = imaging.make_init_image(layer, params.init_mode, params.init_blur,
                                       params.line_threshold, BACKGROUND)
        return {"init": save_thumb(init, f"init-{params.init_mode}-{params.init_blur}.jpg")}

    sweep("init_mode", ["colors", "sketch", "blank"],
          lambda value: replace(defaults, init_mode=value), extra=init_extra)
    sweep("init_blur", [4, defaults.init_blur, 64],
          lambda value: replace(defaults, init_blur=int(value)), extra=init_extra)

    # Refinement after the drawing pauses: the drawing picture, then complete passes with
    # more steps run one after another (what the worker sends while the child looks).
    from server.worker import refine_t_index

    backend.set_params(defaults)
    backend._wrapper.update_stream_params(seed=int(defaults.seed))
    line_map = imaging.make_line_map(layer, defaults.line_polarity, defaults.line_threshold,
                                     defaults.line_dilate)
    init = imaging.make_init_image(layer, defaults.init_mode, defaults.init_blur,
                                   defaults.line_threshold, BACKGROUND)
    for _ in range(backend.latency_frames + 1):
        drawing_picture = backend.generate(init, line_map)
    items = [{"value": "while drawing", "default": False,
              "picture": save_thumb(drawing_picture, "refine-0.jpg")}]
    refined_pictures = {}
    for steps in (4, 8, 12, 20):
        t_index = list(refine_t_index(defaults.t_index_list[0], steps))
        refined = backend.refine(init, line_map, t_index)
        started = time.perf_counter()
        for _ in range(5):
            refined = backend.refine(init, line_map, t_index)
        ms = round((time.perf_counter() - started) * 1000 / 5)
        refined_pictures[steps] = refined
        items.append({"value": f"{steps} steps", "default": steps == defaults.refine_steps, "ms": ms,
                      "picture": save_thumb(refined, f"refine-{steps}.jpg")})
        print("refine", steps, ms, flush=True)
    data["controls"]["refine"] = items

    # AI upscale: a crop of the refined picture without and with the upscaler, enlarged.
    from server.upscale import Upscaler

    upscaler = Upscaler(REPO_ROOT / config.model.upscaler_model)
    if upscaler.available:
        refined = refined_pictures[max(refined_pictures)]
        upscaled = upscaler.upscale(refined, 2)
        started = time.perf_counter()
        for _ in range(10):
            upscaler.upscale(refined, 2)
        upscale_ms = round((time.perf_counter() - started) * 1000 / 10)
        box = (width // 3, height // 4, width // 3 + width // 4, height // 4 + height // 4)
        crop_size = (THUMB_WIDTH, round(THUMB_WIDTH * height / width))
        plain = refined.resize((width * 2, height * 2), Image.Resampling.BICUBIC).crop(
            tuple(value * 2 for value in box)).resize(crop_size, Image.Resampling.NEAREST)
        sharp = upscaled.crop(tuple(value * 2 for value in box)).resize(crop_size, Image.Resampling.NEAREST)
        data["controls"]["upscale"] = [
            {"value": "off (enlarged)", "default": True, "picture": save_thumb(plain, "upscale-off.jpg")},
            {"value": "AI 2×", "default": False, "ms": upscale_ms, "picture": save_thumb(sharp, "upscale-x2.jpg")},
        ]
        print("upscale", upscale_ms, flush=True)

    # Resolution: the same drawing at each offered size (a crop shown at the same scale).
    items = []
    for long_side in (512, 640, 768, 1024):
        side_dims = canvas_dims(dataclasses.replace(config, size=long_side))["landscape"]
        backend.set_size(*side_dims)
        backend.set_params(defaults)
        backend._wrapper.update_stream_params(seed=int(defaults.seed))
        side_layer = demo_layer(*side_dims)
        side_map = imaging.make_line_map(side_layer, defaults.line_polarity, defaults.line_threshold,
                                         defaults.line_dilate)
        side_init = imaging.make_init_image(side_layer, defaults.init_mode, defaults.init_blur,
                                            defaults.line_threshold, BACKGROUND)
        for _ in range(backend.latency_frames + 1):
            picture = backend.generate(side_init, side_map)
        started = time.perf_counter()
        for _ in range(10):
            backend.generate(side_init, side_map)
        ms = round((time.perf_counter() - started) * 1000 / 10)
        items.append({"value": f"{side_dims[0]}×{side_dims[1]}", "default": long_side == defaults.resolution,
                      "ms": ms, "picture": save_thumb(picture, f"resolution-{long_side}.jpg")})
        print("resolution", long_side, ms, flush=True)
    data["controls"]["resolution"] = items
    backend.set_size(width, height)

    # Picture quality: the same picture as JPEG at low / default / high quality, cropped
    # and enlarged so the difference is visible.
    backend.set_params(defaults)
    line_map = imaging.make_line_map(layer, defaults.line_polarity, defaults.line_threshold,
                                     defaults.line_dilate)
    init = imaging.make_init_image(layer, defaults.init_mode, defaults.init_blur,
                                   defaults.line_threshold, BACKGROUND)
    for _ in range(backend.latency_frames + 1):
        base_picture = backend.generate(init, line_map)
    crop_box = (width // 4, height // 4, width // 4 + width // 4, height // 4 + height // 4)
    items = []
    for quality in (50, defaults.jpeg_quality, 95):
        encoded = imaging.encode_jpeg(base_picture, quality)
        decoded = Image.open(io.BytesIO(encoded))
        crop = decoded.crop(crop_box).resize((THUMB_WIDTH, round(THUMB_WIDTH * height / width)),
                                             Image.Resampling.NEAREST)
        items.append({"value": str(quality), "default": quality == defaults.jpeg_quality,
                      "kb": round(len(encoded) / 1024), "picture": save_thumb(crop, f"jpeg-{quality}.jpg")})
    data["controls"]["jpeg_quality"] = items

    memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                             "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True).stdout.split(",")
    data["gpu_memory_mb"] = [int(value) for value in memory] if len(memory) == 2 else None
    (OUT_DIR / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", OUT_DIR / "data.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
