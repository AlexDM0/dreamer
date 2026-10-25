"""Compare the refinement models (server/refine.py) on real test drawings, on the GPU.

    .venv\\Scripts\\python.exe tools\\refine_lab.py                       all models
    .venv\\Scripts\\python.exe tools\\refine_lab.py --models path,anchored --budget 8

For every test drawing (screenshots/examples/*-sketch.png, made by tools/make_showcase.py) it
makes the drawing picture P0 like the worker does, then runs each model and measures, per step,
against Alex's rule ("the first thing you see should not change in broad terms, but detail is
added every step"; metrics from docs/research/refinement-models.md):

- drift:  mean absolute difference (0-255) between P0 and the step at 1/16 size (layout and
          big colour areas only); should stay under ~6;
- detail: variance of the Laplacian of the step relative to P0 (fine detail); should rise;
- step:   mean absolute difference to the previous picture at full size; should shrink.

Writes docs/research/refinement-results.md (tables) and docs/research/img/refine-*.png
(one sheet per drawing: rows = models, columns = P0 and each step).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from server import imaging, refine  # noqa: E402
from server.config import canvas_dims, load_config  # noqa: E402
from server.params import defaults_from_config  # noqa: E402
from server.styles import PRESET_STYLES, PRESET_TOPICS, PROMPT_HEAD, SAFETY_TAIL  # noqa: E402

EXAMPLES = REPO_ROOT / "screenshots" / "examples"
OUT_MD = REPO_ROOT / "docs" / "research" / "refinement-results.md"
OUT_IMG = REPO_ROOT / "docs" / "research" / "img"
# Topic and style per test drawing (as in the showcase run).
DRAWINGS = {
    "house": ("nature", "storybook"), "cat": ("ocean", "cartoon"), "rocket": ("space", "watercolor"),
    "dinosaur": ("dinosaurs", "crayon"), "castle": ("castle", "clay"), "meadow": ("nature", "watercolor"),
}


def layer_from_example(path: Path, size: tuple[int, int]) -> tuple[Image.Image, str]:
    """A composited example sketch back into (stroke layer, paper colour): the most common
    colour is the paper; everything clearly different from it is a stroke."""
    rgb = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    pixels = np.asarray(rgb, dtype=np.int16)
    colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    paper = colors[counts.argmax()]
    distance = np.abs(pixels - paper).max(axis=2)
    alpha = np.clip((distance - 20) * 6, 0, 255).astype(np.uint8)
    layer = Image.fromarray(np.dstack([pixels.astype(np.uint8), alpha]), mode="RGBA")
    return layer, "#%02x%02x%02x" % tuple(int(value) for value in paper)


def small(picture: Image.Image) -> np.ndarray:
    return np.asarray(picture.resize((max(1, picture.width // 16), max(1, picture.height // 16)),
                                     Image.Resampling.BOX), dtype=np.float32)


def laplacian_variance(picture: Image.Image) -> float:
    gray = np.asarray(picture.convert("L"), dtype=np.float32)
    lap = (gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4 * gray[1:-1, 1:-1])
    return float(lap.var())


def measure(p0: Image.Image, steps: list[Image.Image]) -> dict:
    base_small, base_detail = small(p0), laplacian_variance(p0)
    drift, detail, step = [], [], []
    previous = np.asarray(p0, dtype=np.float32)
    for picture in steps:
        picture = picture.resize(p0.size, Image.Resampling.BICUBIC) if picture.size != p0.size else picture
        drift.append(round(float(np.abs(small(picture) - base_small).mean()), 1))
        detail.append(round(laplacian_variance(picture) / max(base_detail, 1e-6), 2))
        current = np.asarray(picture, dtype=np.float32)
        step.append(round(float(np.abs(current - previous).mean()), 1))
        previous = current
    rises = all(b >= a - 0.02 for a, b in zip(detail, detail[1:])) and (detail[-1] >= 1.0 if detail else False)
    return {"drift": drift, "detail": detail, "step": step, "max_drift": max(drift) if drift else 0.0,
            "final_detail": detail[-1] if detail else 1.0,
            "passes": bool(drift) and max(drift) < 6 and rises}


def main() -> int:
    from server.backends.streamdiffusion import StreamDiffusionBackend

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(refine.MODELS))
    parser.add_argument("--budget", type=int, default=8, help="refine_steps (total steps incl. the draw steps)")
    parser.add_argument("--draw", default="6", help="draw t_index list, e.g. 6 or 6,26")
    parser.add_argument("--drawings", default=",".join(DRAWINGS))
    args = parser.parse_args()
    models = [name for name in args.models.split(",") if name]
    draw = tuple(int(value) for value in args.draw.split(","))

    config = load_config(None)
    width, height = canvas_dims(config)["landscape"]
    backend = StreamDiffusionBackend(config)
    backend.warmup()
    defaults = defaults_from_config(config)
    topics = {topic: fragment for topic, _, _, fragment in PRESET_TOPICS}
    styles = {style: fragment for style, _, _, fragment in PRESET_STYLES}
    OUT_IMG.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict]] = {name: {} for name in models}
    timing: dict[str, list[float]] = {name: [] for name in models}
    for drawing in args.drawings.split(","):
        sketch_path = EXAMPLES / f"{drawing}-sketch.png"
        if not sketch_path.exists():
            print(f"skipping {drawing}: {sketch_path} missing (run tools/make_showcase.py)", flush=True)
            continue
        topic, style = DRAWINGS[drawing]
        orientation_dims = (height, width) if drawing == "meadow" else (width, height)
        backend.set_size(*orientation_dims)
        layer, paper = layer_from_example(sketch_path, orientation_dims)
        backend.set_prompt(", ".join([PROMPT_HEAD, topics[topic], styles[style], SAFETY_TAIL]))
        params = dataclasses.replace(defaults, t_index_list=draw)
        line_map = imaging.make_line_map(layer, params.line_polarity, params.line_threshold, params.line_dilate)
        init = imaging.make_init_image(layer, params.init_mode, params.init_blur, params.line_threshold, paper)
        rows = []
        for name in models:
            backend.set_params(params)
            for _ in range(backend.latency_frames + 1):
                p0 = backend.generate(init, line_map)
            started = time.perf_counter()
            steps = list(refine.run(name, backend, p0, init, line_map, draw, args.budget))
            if steps:
                timing[name].append((time.perf_counter() - started) * 1000 / len(steps))
            metrics = measure(p0, steps)
            results[name][drawing] = metrics
            rows.append((name, [p0] + steps))
            print(f"{drawing:9s} {name:15s} drift {metrics['drift']} detail {metrics['detail']} "
                  f"step {metrics['step']} {'PASS' if metrics['passes'] else ''}", flush=True)
        # Sheet: rows = models, columns = P0 + steps.
        tile_w = 200
        tile_h = round(tile_w * orientation_dims[1] / orientation_dims[0])
        columns = max(len(frames) for _, frames in rows)
        label_w = 170
        sheet = Image.new("RGB", (label_w + tile_w * columns, tile_h * len(rows)), "white")
        draw_text = ImageDraw.Draw(sheet)
        for row_index, (name, frames) in enumerate(rows):
            draw_text.text((6, row_index * tile_h + tile_h // 2 - 6), name, fill="black")
            for column, frame in enumerate(frames):
                sheet.paste(frame.resize((tile_w, tile_h)), (label_w + column * tile_w, row_index * tile_h))
        sheet.save(OUT_IMG / f"refine-{drawing}.png")

    # Summary table.
    lines = ["# Refinement models: measured results", "",
             f"Generated by `tools/refine_lab.py --budget {args.budget} --draw {args.draw}` on the RTX 3080. "
             "P0 is the drawing picture; columns are per model, averaged over the test drawings.", "",
             "| model | max drift (≤6) | final detail (×P0, ≥1) | passes rule | step sizes (first → last) | ms/step |",
             "|---|---|---|---|---|---|"]
    for name in models:
        per = list(results[name].values())
        if not per:
            continue
        max_drift = round(statistics.fmean(item["max_drift"] for item in per), 1)
        final_detail = round(statistics.fmean(item["final_detail"] for item in per), 2)
        passes = sum(item["passes"] for item in per)
        first_step = round(statistics.fmean(item["step"][0] for item in per if item["step"]), 1)
        last_step = round(statistics.fmean(item["step"][-1] for item in per if item["step"]), 1)
        ms = round(statistics.fmean(timing[name])) if timing[name] else 0
        lines.append(f"| {name} | {max_drift} | {final_detail} | {passes}/{len(per)} | {first_step} → {last_step} | {ms} |")
    lines += ["", "Per-drawing numbers: see the console output and `refinement-results.json`; "
              "sheets: `img/refine-<drawing>.png` (rows = models, columns = P0 then each step).", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    (OUT_MD.parent / "refinement-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # the tables contain characters cp1252 lacks
    sys.exit(main())
