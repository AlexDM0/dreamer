"""Measure GPU memory and speed of the StreamDiffusion backend per engine batch size and
picture size, for the admin page's memory estimate (static/vram.json).

    .venv\\Scripts\\python.exe tools\\measure_vram.py                 batch 1..4, default sizes
    .venv\\Scripts\\python.exe tools\\measure_vram.py --batches 3     one engine batch size

Each engine batch size (model.max_batch_steps) runs in its own process, so memory from one
does not linger into the next. A batch size without engines builds them first (slow, once).
GPU memory is the whole card as nvidia-smi reports it, so it includes the desktop's share.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_FILE = REPO_ROOT / "static" / "vram.json"
# Long sides to try (landscape, short side 0.7x); the engines' dynamic range is 384..1024.
DEFAULT_SIZES = [512, 640, 768, 896, 1024]


def gpu_memory_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         stdout=subprocess.PIPE, text=True).stdout.split()
    return int(out[0])


def measure_one(batch: int, sizes: list[int]) -> dict:
    from server import imaging
    from server.backends.streamdiffusion import StreamDiffusionBackend
    from server.config import canvas_dims, load_config
    from server.params import defaults_from_config
    from tools.make_help_images import demo_layer

    config = load_config(None)
    config = dataclasses.replace(config, model=dataclasses.replace(
        config.model, max_batch_steps=batch, t_index_list=[min(49, 6 + 20 * i) for i in range(batch)]))
    idle_mb = gpu_memory_mb()
    backend = StreamDiffusionBackend(config)
    backend.set_prompt("a colorful illustration, a sunny meadow, storybook illustration, for children")
    started = time.perf_counter()
    backend.warmup()
    warmup_s = time.perf_counter() - started
    defaults = defaults_from_config(config)
    results = []
    for long_side in sizes:
        dims_config = dataclasses.replace(config, size=long_side)
        width, height = canvas_dims(dims_config)["landscape"]
        backend.set_size(width, height)
        layer = demo_layer(width, height)
        line_map = imaging.make_line_map(layer)
        init = imaging.make_init_image(layer, "colors", 24, 40, "#000000")
        row = {"size": [width, height]}
        for steps in sorted({1, batch}):
            t_index = tuple(min(49, 6 + 20 * i) for i in range(steps)) if steps > 1 else (6,)
            backend.set_params(dataclasses.replace(defaults, t_index_list=t_index))
            for _ in range(backend.latency_frames + 3):
                backend.generate(init, line_map)
            timer = time.perf_counter()
            for _ in range(15):
                backend.generate(init, line_map)
            row[f"ms_{steps}_steps"] = round((time.perf_counter() - timer) * 1000 / 15, 1)
        row["gpu_mb"] = gpu_memory_mb()
        results.append(row)
        print(f"batch {batch} {width}x{height}: {row}", flush=True)
    return {"batch": batch, "idle_mb": idle_mb, "warmup_s": round(warmup_s, 1), "sizes": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", default="1,2,3,4")
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--one", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if args.one is not None:
        print("VRAM_RESULT " + json.dumps(measure_one(args.one, sizes)), flush=True)
        return 0

    data = json.loads(OUT_FILE.read_text(encoding="utf-8")) if OUT_FILE.exists() else {}
    data.setdefault("engines", {})
    for batch in [int(value) for value in args.batches.split(",")]:
        print(f"== engine batch 1..{batch}", flush=True)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", str(batch),
                                    "--sizes", args.sizes], cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("VRAM_RESULT ")]
        if completed.returncode != 0 or not lines:
            print(f"   batch {batch} failed (exit code {completed.returncode})", flush=True)
            continue
        data["engines"][str(batch)] = json.loads(lines[-1][len("VRAM_RESULT "):])
        OUT_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print("   " + lines[-1][:300], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
