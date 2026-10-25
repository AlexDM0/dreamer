"""GPU benchmark for the StreamDiffusion backend.

Feeds a few synthetic kid sketches through StreamDiffusionBackend (the same code the
server runs) and reports ms/frame and fps. Each case runs in its own subprocess, so
only one wrapper (and its TensorRT engines) is in VRAM at a time. The first run of a
case builds its engines into model.engine_dir, which can take many minutes.

    .venv\\Scripts\\python.exe tools\\bench_gpu.py                 all cases, markdown table
    .venv\\Scripts\\python.exe tools\\bench_gpu.py --case cn-2     one case
    .venv\\Scripts\\python.exe tools\\bench_gpu.py --save-dir out  also save input/line map/output PNGs
    .venv\\Scripts\\python.exe tools\\bench_gpu.py --no-safety     without the per-frame safety checker

Cases: nocn-1, nocn-2 (ControlNet off) and cn-1, cn-2 (ControlNet on), 1 or 2 steps.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from server import imaging  # noqa: E402
from server.config import load_config  # noqa: E402
from server.params import defaults_from_config  # noqa: E402

CASES = {
    "nocn-1": (False, [10]),
    "nocn-2": (False, [10, 30]),
    "cn-1": (True, [10]),
    "cn-2": (True, [10, 30]),
}
PROMPT = ("a friendly cat in a garden, children's book illustration, bright colors, "
          "for children, friendly, wholesome")


def make_sketches(size: int) -> list[Image.Image]:
    """Three kid-style line drawings (black strokes on white)."""
    sketches = []
    scale = size / 512

    def pts(*coords: float) -> list[float]:
        return [value * scale for value in coords]

    house = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(house)
    draw.rectangle(pts(140, 250, 370, 450), outline="black", width=int(8 * scale))
    draw.line(pts(120, 260, 255, 130, 390, 260), fill="black", width=int(8 * scale))
    draw.rectangle(pts(230, 350, 285, 450), outline="black", width=int(6 * scale))
    draw.ellipse(pts(390, 40, 480, 130), outline="black", width=int(6 * scale))
    sketches.append(house)

    cat = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(cat)
    draw.ellipse(pts(140, 160, 380, 400), outline="black", width=int(8 * scale))
    draw.line(pts(160, 200, 180, 100, 240, 170), fill="black", width=int(8 * scale))
    draw.line(pts(280, 170, 340, 100, 360, 200), fill="black", width=int(8 * scale))
    draw.ellipse(pts(200, 240, 230, 270), fill="black")
    draw.ellipse(pts(290, 240, 320, 270), fill="black")
    draw.line(pts(230, 320, 260, 340, 290, 320), fill="black", width=int(6 * scale))
    sketches.append(cat)

    flower = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(flower)
    draw.line(pts(256, 260, 256, 480), fill="green", width=int(8 * scale))
    for offset_x, offset_y in ((0, -70), (70, 0), (0, 70), (-70, 0)):
        draw.ellipse(pts(216 + offset_x, 160 + offset_y, 296 + offset_x, 240 + offset_y),
                     outline="red", width=int(8 * scale))
    draw.ellipse(pts(226, 170, 286, 230), outline="orange", width=int(8 * scale))
    sketches.append(flower)
    return sketches


def run_case(case: str, frames: int, warmup_frames: int, save_dir: Path | None,
             polarity: str | None, safety: bool = True) -> dict:
    from server.backends.streamdiffusion import StreamDiffusionBackend

    use_controlnet, t_index_list = CASES[case]
    config = load_config(None)
    model = dataclasses.replace(
        config.model,
        t_index_list=t_index_list,
        controlnet=config.model.controlnet if use_controlnet else None,
    )
    if polarity:
        model = dataclasses.replace(model, line_polarity=polarity)
    if not safety:
        model = dataclasses.replace(model, safety_checker=False)
    config = dataclasses.replace(config, backend="streamdiffusion", model=model)
    params = dataclasses.replace(defaults_from_config(config), t_index_list=tuple(t_index_list))

    backend = StreamDiffusionBackend(config)
    build_start = time.perf_counter()
    backend.set_prompt(PROMPT)
    backend.warmup()
    warmup_seconds = time.perf_counter() - build_start
    backend.set_params(params)

    sketches = make_sketches(config.size)
    line_maps = [imaging.make_line_map(sketch, params.line_polarity, params.line_threshold,
                                       params.line_dilate) for sketch in sketches]
    for index in range(warmup_frames):
        backend.generate(sketches[index % len(sketches)],
                         line_maps[index % len(sketches)] if use_controlnet else None)

    timings_ms = []
    for index in range(frames):
        sketch_index = index % len(sketches)
        start = time.perf_counter()
        line_map = imaging.make_line_map(sketches[sketch_index], params.line_polarity,
                                         params.line_threshold, params.line_dilate) \
            if use_controlnet else None
        picture = backend.generate(sketches[sketch_index], line_map)
        imaging.encode_jpeg(picture, params.jpeg_quality)
        timings_ms.append((time.perf_counter() - start) * 1000)

    if save_dir is not None:
        # Run each sketch latency+1 times, as the worker's prime/settle does, so the saved
        # picture belongs to that sketch and not to the previous one.
        outputs = {}
        for sketch_index, sketch in enumerate(sketches):
            for _ in range(backend.latency_frames + 1):
                outputs[sketch_index] = backend.generate(
                    sketch, line_maps[sketch_index] if use_controlnet else None)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"-{polarity}" if polarity else ""
        for sketch_index, picture in outputs.items():
            sketches[sketch_index].save(save_dir / f"sketch{sketch_index}.png")
            if use_controlnet:
                line_maps[sketch_index].save(save_dir / f"linemap{sketch_index}{suffix}.png")
            picture.save(save_dir / f"{case}{suffix}-out{sketch_index}.png")

    import torch

    gpu_memory_used = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True).stdout.split()
    mean_ms = statistics.fmean(timings_ms)
    return {
        "case": case,
        "controlnet": use_controlnet,
        "steps": len(t_index_list),
        "t_index_list": t_index_list,
        "polarity": params.line_polarity,
        "safety_checker": params.safety_checker,
        "frames": frames,
        "mean_ms": round(mean_ms, 1),
        "median_ms": round(statistics.median(timings_ms), 1),
        "p95_ms": round(sorted(timings_ms)[int(0.95 * (len(timings_ms) - 1))], 1),
        "fps": round(1000 / mean_ms, 1),
        "warmup_s": round(warmup_seconds, 1),
        # Whole-GPU memory in use (TensorRT allocations are invisible to torch), including
        # whatever the desktop uses.
        "gpu_mem_used_gb": round(int(gpu_memory_used[0]) / 1024, 2) if gpu_memory_used else None,
        "gpu": torch.cuda.get_device_name(0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES), help="run one case in this process")
    parser.add_argument("--cases", default=",".join(CASES), help="comma-separated cases for the table")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--polarity", choices=imaging.POLARITIES, help="override model.line_polarity")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--no-safety", action="store_true", help="run without the per-frame safety checker")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.case:
        result = run_case(args.case, args.frames, args.warmup_frames, args.save_dir, args.polarity,
                          safety=not args.no_safety)
        print("BENCH_RESULT " + json.dumps(result), flush=True)
        return 0

    results = []
    for case in args.cases.split(","):
        command = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                   "--frames", str(args.frames), "--warmup-frames", str(args.warmup_frames)]
        if args.polarity:
            command += ["--polarity", args.polarity]
        if args.save_dir:
            command += ["--save-dir", str(args.save_dir)]
        if args.no_safety:
            command += ["--no-safety"]
        print(f"== {case}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("BENCH_RESULT ")]
        if completed.returncode != 0 or not lines:
            print(f"   {case} failed (exit code {completed.returncode})", flush=True)
            results.append({"case": case, "error": f"exit code {completed.returncode}"})
            continue
        results.append(json.loads(lines[-1][len("BENCH_RESULT "):]))
        print("   " + lines[-1], flush=True)

    print()
    print("| case | ControlNet | steps | t_index_list | mean ms | median ms | p95 ms | fps | GPU memory used GB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for result in results:
        if "error" in result:
            print(f"| {result['case']} | | | | failed: {result['error']} | | | | |")
            continue
        print(f"| {result['case']} | {'on' if result['controlnet'] else 'off'} | {result['steps']} | "
              f"{result['t_index_list']} | {result['mean_ms']} | {result['median_ms']} | "
              f"{result['p95_ms']} | {result['fps']} | {result['gpu_mem_used_gb']} |")
    return 0 if all("error" not in result for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
