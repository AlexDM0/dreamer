"""Slideshows of every refinement mode, for every topic and style, on the GPU.

    .venv\\Scripts\\python.exe tools\\make_slideshows.py
    .venv\\Scripts\\python.exe tools\\make_slideshows.py --out D:\\development\\dreamer-slideshows --starts 1,2
    .venv\\Scripts\\python.exe tools\\make_slideshows.py --modes anchored,tcd --topics ocean --styles cartoon

For six seed drawings (three detailed ones from screenshots/examples, three abstract scribbles
drawn here), every topic and style in data/styles.json (hidden ones and custom ones too), and
every start step count (the draw reaction: 1-4 steps; TCD: 1-4 drawing steps), it makes the
drawing picture (step 0) and then up to 10 refinement steps with each mode: the seven SD-Turbo
refinement models (server/refine.py) and SD 1.5 + TCD (the same picture with 1..10 more steps).

Writes thumbnails, per-step change scores (tools/refine_lab.py metrics) and a viewer:
<out>/index.html (pick topic, style and start steps; all seeds on one page). Resumable: a
finished combination is skipped on the next run. Stop Dreamer first: both need the GPU.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # engines/ and models/ are relative to the repo

from PIL import Image, ImageDraw  # noqa: E402

from server import imaging, refine  # noqa: E402
from server.config import canvas_dims, load_config  # noqa: E402
from server.params import defaults_from_config  # noqa: E402
from server.refine_trace import mean_change, small  # noqa: E402
from server.styles import PROMPT_HEAD, SAFETY_TAIL, StyleStore  # noqa: E402
from tools.refine_lab import EXAMPLES, layer_from_example  # noqa: E402

DEFAULT_OUT = REPO_ROOT.parent / "dreamer-slideshows"
THUMB_WIDTH = 320
THUMB_QUALITY = 80
MORE_STEPS = 10
CREATIVITY = 88  # the default (t_index 6 first), as the admin page computes it
TURBO_MODES = list(refine.MODELS)
MODE_LABELS = {name: label for name, (_, label, _) in refine.MODELS.items()} | {"tcd": "SD 1.5 + TCD"}
DETAILED_SEEDS = {"house": "House, tree and sun", "cat": "Cat and fish", "dinosaur": "Dinosaur"}
ABSTRACT_SEEDS = {"spiral": "Spiral scribble", "blobs": "Colour blobs", "zigzag": "Zigzags and loops"}


# ----- seeds ------------------------------------------------------------------------

def _stroke(pen: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int) -> None:
    """A finger stroke: a polyline with round joins and ends."""
    pen.line(points, fill=color, width=width, joint="curve")
    radius = width / 2
    for x, y in (points[0], points[-1]):
        pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def abstract_seed(name: str, size: tuple[int, int]) -> tuple[Image.Image, str]:
    """An abstract scribble as (stroke layer, paper colour), drawn like a small child would."""
    width, height = size
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)
    jitter = random.Random(name)
    if name == "spiral":
        paper = "#141311"
        for color, turns, scale in (("#ff9f1c", 3.2, 1.0), ("#2fa8ff", 2.2, 0.6)):
            points = []
            for index in range(int(turns * 60)):
                angle = index / 60 * 2 * math.pi
                radius = scale * (8 + index * 0.9)
                points.append((width * 0.45 + radius * math.cos(angle) + jitter.uniform(-2, 2),
                               height * 0.5 + radius * 0.8 * math.sin(angle) + jitter.uniform(-2, 2)))
            _stroke(pen, points, color, 9)
        _stroke(pen, [(width * 0.78, height * 0.15), (width * 0.9, height * 0.3), (width * 0.8, height * 0.45)],
                "#ffd23f", 12)
    elif name == "blobs":
        paper = "#f7f5f2"
        for color, cx, cy, radius in (("#ef4d5a", 0.25, 0.35, 60), ("#3ecf6e", 0.6, 0.6, 80),
                                      ("#8b5cf6", 0.8, 0.25, 45), ("#ffd23f", 0.4, 0.75, 40)):
            for ring in range(0, radius, 9):  # coloured in with round scribbles
                points = [(width * cx + ring * math.cos(step / 8) + jitter.uniform(-3, 3),
                           height * cy + ring * 0.8 * math.sin(step / 8) + jitter.uniform(-3, 3))
                          for step in range(52)]
                _stroke(pen, points, color, 10)
        for _ in range(14):
            x, y = jitter.uniform(0.05, 0.95) * width, jitter.uniform(0.05, 0.95) * height
            pen.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#2fa8ff")
    else:  # zigzag
        paper = "#1e2a5a"
        for row, color in enumerate(("#3ecf6e", "#ff7eb6", "#ffd23f")):
            base = height * (0.25 + row * 0.25)
            points = [(width * (0.08 + step * 0.07), base + (24 if step % 2 else -24) + jitter.uniform(-4, 4))
                      for step in range(13)]
            _stroke(pen, points, color, 10)
        for loop in range(3):
            cx, cy = width * (0.25 + loop * 0.25), height * 0.5
            points = [(cx + 40 * math.cos(step / 10) + jitter.uniform(-2, 2),
                       cy + 55 * math.sin(step / 10 * 2) + jitter.uniform(-2, 2)) for step in range(64)]
            _stroke(pen, points, "#ffffff", 7)
    return layer, paper


def load_seeds(size: tuple[int, int]) -> list[dict]:
    seeds = []
    for name, label in DETAILED_SEEDS.items():
        layer, paper = layer_from_example(EXAMPLES / f"{name}-sketch.png", size)
        seeds.append({"id": name, "label": label, "kind": "detailed", "layer": layer, "paper": paper})
    for name, label in ABSTRACT_SEEDS.items():
        layer, paper = abstract_seed(name, size)
        seeds.append({"id": name, "label": label, "kind": "abstract", "layer": layer, "paper": paper})
    return seeds


# ----- prompts and steps ------------------------------------------------------------

def all_items(store: StyleStore, kind: str) -> list[dict]:
    """Every topic or style with a prompt, hidden and custom ones included."""
    return [item for item in store._items[kind] if item.get("prompt")]  # read-only peek


def prompt_for(topic: dict, style: dict) -> str:
    return ", ".join([PROMPT_HEAD] + [part for part in (topic["prompt"], style["prompt"]) if part] + [SAFETY_TAIL])


def t_index_for(steps: int, creativity: int = CREATIVITY, t_max: int = 49) -> tuple[int, ...]:
    """The admin page's mapping (static/admin.js tIndexList): creativity + steps -> t_index_list."""
    first = min(round(t_max * (1 - creativity / 100)), t_max - (steps - 1))
    if steps == 1:
        return (first,)
    last = min(t_max, first + 20 * (steps - 1))
    gap = max(last - first, steps - 1) / (steps - 1)
    return tuple(round(first + index * gap) for index in range(steps))


# ----- output -----------------------------------------------------------------------

def save_thumb(picture: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height = round(THUMB_WIDTH * picture.height / picture.width)
    picture.convert("RGB").resize((THUMB_WIDTH, height), Image.Resampling.LANCZOS).save(
        path, "JPEG", quality=THUMB_QUALITY, optimize=True)


def scores(first: Image.Image, pictures: list[Image.Image]) -> list[dict]:
    """Per step: change from the previous step and drift from step 0 (1/16 size, 0-255)."""
    size = (max(1, first.width // 16), max(1, first.height // 16))
    base = small(first, size)
    previous, out = base, []
    for picture in pictures:
        current = small(picture.resize(first.size) if picture.size != first.size else picture, size)
        out.append({"step": round(mean_change(previous, current), 1), "drift": round(mean_change(base, current), 1)})
        previous = current
    return out


def combo_key(seed: str, topic: str, style: str, start: int) -> str:
    return f"{seed}__{topic}__{style}__n{start}"


class Output:
    def __init__(self, out: Path, catalogue: dict) -> None:
        self.out = out
        self.catalogue = catalogue
        self.meta_dir = out / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.last_index = 0.0

    def image_path(self, seed: str, topic: str, style: str, start: int, name: str) -> Path:
        return self.out / "img" / seed / topic / style / f"n{start}" / f"{name}.jpg"

    def meta_path(self, key: str, family: str) -> Path:
        return self.meta_dir / f"{key}__{family}.json"

    def done(self, key: str, family: str) -> bool:
        return self.meta_path(key, family).exists()

    def finish(self, key: str, family: str, meta: dict) -> None:
        self.meta_path(key, family).write_text(json.dumps(meta), encoding="utf-8")
        if time.monotonic() - self.last_index > 20:
            self.write_index()

    def write_index(self) -> None:
        """data.js for the viewer (a script, so index.html also works from file://)."""
        done = {}
        for path in self.meta_dir.glob("*.json"):
            key, family = path.stem.rsplit("__", 1)
            done.setdefault(key, {})[family] = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(self.catalogue, done=done, updated=time.strftime("%Y-%m-%d %H:%M"))
        temp = self.out / "data.js.tmp"
        temp.write_text("window.SLIDES = " + json.dumps(payload) + ";\n", encoding="utf-8")
        os.replace(temp, self.out / "data.js")
        self.last_index = time.monotonic()


# ----- generation -------------------------------------------------------------------

def run_turbo(config, defaults, seeds, topics, styles, starts, modes, output: Output, log) -> None:
    from server.backends.streamdiffusion import StreamDiffusionBackend
    from server.upscale import Upscaler

    backend = StreamDiffusionBackend(config)
    backend.warmup()
    upscaler = Upscaler(REPO_ROOT / config.model.upscaler_model)
    for start in starts:
        draw = t_index_for(start)
        params = dataclasses.replace(defaults, t_index_list=draw)
        backend.set_params(params)
        for seed in seeds:
            init, line_map = seed["init"], seed["line_map"]
            for topic in topics:
                for style in styles:
                    key = combo_key(seed["id"], topic["id"], style["id"], start)
                    if output.done(key, "turbo"):
                        continue
                    started = time.perf_counter()
                    backend.set_prompt(prompt_for(topic, style))
                    backend.set_params(params)  # back to the drawing steps after refining
                    for _ in range(backend.latency_frames + 1):
                        first = backend.generate(init, line_map)
                    save_thumb(first, output.image_path(seed["id"], topic["id"], style["id"], start, "turbo-00"))
                    meta = {"modes": {}}
                    for mode in modes:
                        steps = refine.run(mode, backend, first, init, line_map, draw, start + MORE_STEPS, upscaler)
                        pictures = []
                        try:
                            for picture in steps:
                                pictures.append(picture)
                                if len(pictures) == MORE_STEPS:
                                    break
                        finally:
                            steps.close()
                        for index, picture in enumerate(pictures, start=1):
                            save_thumb(picture, output.image_path(seed["id"], topic["id"], style["id"], start,
                                                                  f"{mode}-{index:02d}"))
                        meta["modes"][mode] = {"count": len(pictures), "scores": scores(first, pictures)}
                    meta["seconds"] = round(time.perf_counter() - started, 2)
                    output.finish(key, "turbo", meta)
                    log(f"turbo {key} {meta['seconds']}s")
    backend.close()


def run_tcd(config, defaults, seeds, topics, styles, starts, output: Output, log) -> None:
    from server.backends.tcd import TcdBackend

    backend = TcdBackend(config)
    backend.warmup()
    for start in starts:
        for seed in seeds:
            init, line_map = seed["init"], seed["line_map"]
            for topic in topics:
                for style in styles:
                    key = combo_key(seed["id"], topic["id"], style["id"], start)
                    if output.done(key, "tcd"):
                        continue
                    started = time.perf_counter()
                    backend.set_prompt(prompt_for(topic, style))
                    pictures = []
                    for steps in range(start, start + MORE_STEPS + 1):  # step 0 and 10 more
                        backend.set_params(dataclasses.replace(defaults, picture_model="tcd", tcd_steps=steps))
                        pictures.append(backend.generate(init, line_map))
                    for index, picture in enumerate(pictures):
                        save_thumb(picture, output.image_path(seed["id"], topic["id"], style["id"], start,
                                                              f"tcd-{index:02d}"))
                    meta = {"modes": {"tcd": {"count": len(pictures) - 1,
                                              "scores": scores(pictures[0], pictures[1:])}},
                            "seconds": round(time.perf_counter() - started, 2)}
                    output.finish(key, "tcd", meta)
                    log(f"tcd {key} {meta['seconds']}s")
    backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--starts", default="1,2,3,4", help="start step counts (draw reaction)")
    parser.add_argument("--modes", default=",".join(TURBO_MODES + ["tcd"]))
    parser.add_argument("--topics", default="", help="comma-separated ids (default: all)")
    parser.add_argument("--styles", default="", help="comma-separated ids (default: all)")
    parser.add_argument("--seeds", default="", help="comma-separated ids (default: all six)")
    args = parser.parse_args()

    config = load_config(None)
    size = canvas_dims(config)["landscape"]
    defaults = defaults_from_config(config)
    store = StyleStore(config.data_dir)
    topics = [item for item in all_items(store, "topics") if not args.topics or item["id"] in args.topics.split(",")]
    styles = [item for item in all_items(store, "styles") if not args.styles or item["id"] in args.styles.split(",")]
    starts = [int(value) for value in args.starts.split(",")]
    modes = [mode for mode in args.modes.split(",") if mode]
    seeds = [seed for seed in load_seeds(size) if not args.seeds or seed["id"] in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed["line_map"] = imaging.make_line_map(seed["layer"], defaults.line_polarity, defaults.line_threshold,
                                                 defaults.line_dilate)
        seed["init"] = imaging.make_init_image(seed["layer"], defaults.init_mode, defaults.init_blur,
                                               defaults.line_threshold, seed["paper"])
        sketch = imaging.compose(seed["layer"], seed["paper"])
        sketch_path = args.out / "seeds" / f"{seed['id']}.png"
        sketch_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        sketch.save(buffer, "PNG")
        sketch_path.write_bytes(buffer.getvalue())

    catalogue = {
        "seeds": [{"id": seed["id"], "label": seed["label"], "kind": seed["kind"]} for seed in seeds],
        "topics": [{"id": item["id"], "label": item.get("label", item["id"]), "emoji": item.get("emoji", "")}
                   for item in topics],
        "styles": [{"id": item["id"], "label": item.get("label", item["id"]), "emoji": item.get("emoji", "")}
                   for item in styles],
        "modes": [{"id": mode, "label": MODE_LABELS.get(mode, mode)} for mode in modes],
        "starts": starts, "more_steps": MORE_STEPS, "size": list(size), "creativity": CREATIVITY,
        "t_index": {str(start): list(t_index_for(start)) for start in starts},
    }
    output = Output(args.out, catalogue)
    output.write_index()
    (args.out / "index.html").write_text((REPO_ROOT / "tools" / "slideshow-viewer.html").read_text(encoding="utf-8"),
                                         encoding="utf-8")
    started = time.monotonic()

    def log(message: str) -> None:
        print(f"[{(time.monotonic() - started) / 60:6.1f} min] {message}", flush=True)

    turbo_modes = [mode for mode in modes if mode != "tcd"]
    if turbo_modes:
        run_turbo(config, defaults, seeds, topics, styles, starts, turbo_modes, output, log)
    if "tcd" in modes:
        run_tcd(config, defaults, seeds, topics, styles, starts, output, log)
    output.write_index()
    log(f"done: {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
