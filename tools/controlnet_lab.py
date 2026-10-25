"""Compare ControlNets for following the child's drawing, on the GPU.

    .venv\\Scripts\\python.exe tools\\controlnet_lab.py
    .venv\\Scripts\\python.exe tools\\controlnet_lab.py --variants lineart:0.45:0,scribble:0.55:1

A variant is ``name:strength:line_dilate``; names map to the SD 2.1 ControlNets in VARIANTS.
Each ControlNet runs in its own process (first use builds its TensorRT engine, a few minutes).
For every test drawing (screenshots/examples) it renders the drawing picture (draw reaction,
1 step) and the refined picture (default refinement model), and measures ms per frame and GPU
memory. Writes docs/research/img/controlnet-<drawing>.png (rows = variants) and
docs/research/controlnet-results.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

VARIANTS = {
    "scribble": "thibaud/controlnet-sd21-scribble-diffusers",
    "lineart": "thibaud/controlnet-sd21-lineart-diffusers",
    "hed": "thibaud/controlnet-sd21-hed-diffusers",
}
DEFAULT_VARIANTS = "scribble:0.55:1,lineart:0.55:0,lineart:0.4:0,hed:0.55:0,hed:0.4:0"
OUT_IMG = REPO_ROOT / "docs" / "research" / "img"
OUT_JSON = REPO_ROOT / "docs" / "research" / "controlnet-results.json"


def gpu_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         stdout=subprocess.PIPE, text=True).stdout.split()
    return int(out[0])


def run_controlnet(name: str, settings: list[tuple[float, int]], drawings: list[str], out_dir: Path) -> dict:
    from server import imaging, refine
    from server.backends.streamdiffusion import StreamDiffusionBackend
    from server.config import canvas_dims, load_config
    from server.params import defaults_from_config
    from server.styles import PRESET_STYLES, PRESET_TOPICS, PROMPT_HEAD, SAFETY_TAIL
    from tools.refine_lab import DRAWINGS, EXAMPLES, layer_from_example

    config = load_config(None)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, controlnet=VARIANTS[name]))
    width, height = canvas_dims(config)["landscape"]
    backend = StreamDiffusionBackend(config)
    backend.warmup()
    memory = gpu_mb()
    defaults = defaults_from_config(config)
    topics = {topic: fragment for topic, _, _, fragment in PRESET_TOPICS}
    styles = {style: fragment for style, _, _, fragment in PRESET_STYLES}
    results: dict = {}
    for strength, dilate in settings:
        key = f"{name}:{strength}:{dilate}"
        params = dataclasses.replace(defaults, t_index_list=(6,), controlnet_scale=strength, line_dilate=dilate)
        backend.set_params(params)
        timings = []
        for drawing in drawings:
            topic, style = DRAWINGS[drawing]
            dims = (height, width) if drawing == "meadow" else (width, height)
            backend.set_size(*dims)
            backend.set_params(params)
            layer, paper = layer_from_example(EXAMPLES / f"{drawing}-sketch.png", dims)
            backend.set_prompt(", ".join([PROMPT_HEAD, topics[topic], styles[style], SAFETY_TAIL]))
            line_map = imaging.make_line_map(layer, params.line_polarity, params.line_threshold, dilate)
            init = imaging.make_init_image(layer, params.init_mode, params.init_blur, params.line_threshold, paper)
            for _ in range(3):
                picture = backend.generate(init, line_map)
            started = time.perf_counter()
            for _ in range(10):
                picture = backend.generate(init, line_map)
            timings.append((time.perf_counter() - started) * 100)
            refined = picture
            for refined in refine.run(refine.DEFAULT_MODEL, backend, picture, init, line_map, (6,), 8):
                pass
            picture.save(out_dir / f"{key.replace(':', '_')}-{drawing}-p0.png")
            refined.save(out_dir / f"{key.replace(':', '_')}-{drawing}-refined.png")
            print(f"{key} {drawing} done", flush=True)
        results[key] = {"ms_per_frame": round(statistics.fmean(timings), 1), "gpu_mb": memory}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--drawings", default="house,cat,rocket,dinosaur,castle,meadow")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--settings", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    drawings = args.drawings.split(",")
    if args.one:
        settings = [(float(strength), int(dilate)) for strength, dilate in json.loads(args.settings)]
        print("CN_RESULT " + json.dumps(run_controlnet(args.one, settings, drawings, Path(args.out))), flush=True)
        return 0

    import tempfile
    variants = [variant.split(":") for variant in args.variants.split(",")]
    by_controlnet: dict[str, list] = {}
    for name, strength, dilate in variants:
        by_controlnet.setdefault(name, []).append((float(strength), int(dilate)))
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    # ignore_cleanup_errors: Windows can still hold a just-read PNG open when the folder goes.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        for name, settings in by_controlnet.items():
            print(f"== {name} {settings}", flush=True)
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", name,
                                        "--settings", json.dumps(settings), "--drawings", args.drawings,
                                        "--out", temp], cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True)
            lines = [line for line in completed.stdout.splitlines() if line.startswith("CN_RESULT ")]
            if completed.returncode != 0 or not lines:
                print(f"   {name} failed (exit code {completed.returncode})", flush=True)
                continue
            results.update(json.loads(lines[-1][len("CN_RESULT "):]))
        keys = [f"{name}:{strength}:{dilate}" for name, strength, dilate in variants if f"{name}:{strength}:{dilate}" in results]
        for drawing in drawings:
            with Image.open(Path(temp) / f"{keys[0].replace(':', '_')}-{drawing}-p0.png") as sample:
                sample_size = sample.size
            tile_w = 240
            tile_h = round(tile_w * sample_size[1] / sample_size[0])
            label_w = 150
            sheet = Image.new("RGB", (label_w + tile_w * 3, tile_h * len(keys)), "white")
            text = ImageDraw.Draw(sheet)
            sketch = Image.open(REPO_ROOT / "screenshots" / "examples" / f"{drawing}-sketch.png")
            for row, key in enumerate(keys):
                text.text((6, row * tile_h + tile_h // 2 - 6), key, fill="black")
                stem = key.replace(":", "_")
                for column, image in enumerate((sketch, Image.open(Path(temp) / f"{stem}-{drawing}-p0.png"),
                                                Image.open(Path(temp) / f"{stem}-{drawing}-refined.png"))):
                    sheet.paste(image.resize((tile_w, tile_h)), (label_w + column * tile_w, row * tile_h))
            sheet.save(OUT_IMG / f"controlnet-{drawing}.png")
        OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for key, value in results.items():
        print(key, value, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
