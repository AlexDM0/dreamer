"""Offline test of research idea R6: SD 1.5 + the TCD LoRA, a model made for many steps.

    .venv\\Scripts\\python.exe tools\\tcd_lab.py
    .venv\\Scripts\\python.exe tools\\tcd_lab.py --steps 2,4,8,12,20 --gamma 0.3

Plain diffusers (no StreamDiffusion, no TensorRT), so the timings are an upper bound. For each
test drawing (screenshots/examples) it makes the picture from the same start image, line map and
seed at each step count, and measures every picture against the fewest-steps one with the
refine_lab metrics (drift of the broad layout, detail, step size). If more steps only add
detail, TCD could serve as a refinement model. First run downloads about 4 GB (SD 1.5, the TCD
LoRA, the SD 1.5 scribble ControlNet). Writes docs/research/img/tcd-<drawing>.png and
docs/research/tcd-results.json.
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

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from server import imaging  # noqa: E402
from server.config import canvas_dims, load_config  # noqa: E402
from server.params import defaults_from_config  # noqa: E402
from server.styles import PRESET_STYLES, PRESET_TOPICS, PROMPT_HEAD, SAFETY_TAIL  # noqa: E402
from tools.refine_lab import DRAWINGS, EXAMPLES, layer_from_example, measure  # noqa: E402

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET = "lllyasviel/control_v11p_sd15_scribble"
LORA = "h1t/TCD-SD15-LoRA"
OUT_IMG = REPO_ROOT / "docs" / "research" / "img"
OUT_JSON = REPO_ROOT / "docs" / "research" / "tcd-results.json"


def load_pipeline():
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, TCDScheduler

    controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        BASE, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False).to("cuda")
    # The fork's diffusers build returns (output, kv cache) from attention; the UNet expects that,
    # the VAE does not, so the VAE gets the plain processor.
    pipeline.vae.set_default_attn_processor()
    pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(LORA, weight_name="pytorch_lora_weights.safetensors")
    pipeline.fuse_lora()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="2,4,8,12,20")
    parser.add_argument("--gamma", type=float, default=0.3, help="TCD eta: fresh noise per step")
    parser.add_argument("--strength", type=float, default=0.88, help="img2img strength (t_index 6 of 50)")
    parser.add_argument("--controlnet-scale", type=float, default=0.55)
    parser.add_argument("--drawings", default=",".join(DRAWINGS))
    parser.add_argument("--init-mode", default=None, help="start image: colors | sketch | blank")
    parser.add_argument("--init-blur", type=int, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    step_counts = [int(value) for value in args.steps.split(",")]

    config = load_config(None)
    width, height = canvas_dims(config)["landscape"]
    defaults = defaults_from_config(config)
    defaults = dataclasses.replace(defaults, init_mode=args.init_mode or defaults.init_mode,
                                   init_blur=defaults.init_blur if args.init_blur is None else args.init_blur)
    topics = {topic: fragment for topic, _, _, fragment in PRESET_TOPICS}
    styles = {style: fragment for style, _, _, fragment in PRESET_STYLES}
    pipeline = load_pipeline()
    OUT_IMG.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for drawing in args.drawings.split(","):
        topic, style = DRAWINGS[drawing]
        dims = (height, width) if drawing == "meadow" else (width, height)
        layer, paper = layer_from_example(EXAMPLES / f"{drawing}-sketch.png", dims)
        line_map = imaging.make_line_map(layer, defaults.line_polarity, defaults.line_threshold,
                                         defaults.line_dilate)
        init = imaging.make_init_image(layer, defaults.init_mode, defaults.init_blur,
                                       defaults.line_threshold, paper)
        prompt = ", ".join([PROMPT_HEAD, topics[topic], styles[style], SAFETY_TAIL])
        pictures, timings = [], {}
        for count in step_counts:
            torch.cuda.synchronize()
            started = time.perf_counter()
            picture = pipeline(prompt=prompt, image=init, control_image=line_map,
                               num_inference_steps=count, strength=args.strength, guidance_scale=0.0,
                               eta=args.gamma, controlnet_conditioning_scale=args.controlnet_scale,
                               width=dims[0], height=dims[1],
                               generator=torch.Generator("cuda").manual_seed(2)).images[0]
            torch.cuda.synchronize()
            timings[count] = round((time.perf_counter() - started) * 1000)
            pictures.append(picture)
        metrics = measure(pictures[0], pictures[1:])
        results[drawing] = {"steps": step_counts, "ms": timings, **metrics}
        print(f"{drawing:9s} drift {metrics['drift']} detail {metrics['detail']} step {metrics['step']} "
              f"ms {timings}", flush=True)
        tile_w = 200
        tile_h = round(tile_w * dims[1] / dims[0])
        sheet = Image.new("RGB", (tile_w * (len(pictures) + 1), tile_h + 16), "white")
        text = ImageDraw.Draw(sheet)
        for column, (label, image) in enumerate([("sketch", Image.open(EXAMPLES / f"{drawing}-sketch.png"))]
                                                + [(f"{count} steps", picture)
                                                   for count, picture in zip(step_counts, pictures)]):
            sheet.paste(image.convert("RGB").resize((tile_w, tile_h)), (column * tile_w, 16))
            text.text((column * tile_w + 4, 2), label, fill="black")
        sheet.save(OUT_IMG / f"tcd-{drawing}{args.tag}.png")

    summary = {
        "settings": vars(args),
        "max_drift": round(statistics.fmean(item["max_drift"] for item in results.values()), 1),
        "final_detail": round(statistics.fmean(item["final_detail"] for item in results.values()), 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "drawings": results,
    }
    OUT_JSON.with_name(f"tcd-results{args.tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "drawings"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
