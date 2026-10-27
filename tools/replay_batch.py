"""Replay a saved refinement batch (Refinement steps card, "Save batch") on the GPU.

    .venv\\Scripts\\python.exe tools\\replay_batch.py data\\refine-batches\\20261026-194552-anchored
    .venv\\Scripts\\python.exe tools\\replay_batch.py <batch> --saved-first
    .venv\\Scripts\\python.exe tools\\replay_batch.py <batch> --param refine_steps=6 --set ANCHOR_SIGMA=1.5

A batch folder holds the sketch as the iPad sent it (sketch.png, the stroke layer), every step
as it was shown (step-00.jpg = the picture when the brush lifted, step-01.jpg ...) and
settings.json (server/refine_trace.py). This rebuilds the drawing's inputs like the worker does
(imaging.decode_sketch, then the init image and line map of server/worker.frame_inputs),
applies the saved params, prompt and picture size to StreamDiffusionBackend (SD-Turbo), makes
the drawing picture and runs the same refinement (server/refine.run), upscaling like the worker
if that was on.

It writes replay-step-XX.jpg next to the originals and prints, per step, the recorded change
scores, the replay's (same metrics: mean absolute difference 0-255 at 1/16 size, to the
previous step and to step 0), and how far each replayed step is from the original one.

The drawing picture is made fresh, so it can differ from the one the child saw (the live
stream had other noise in it). ``--saved-first`` refines from the saved step-00.jpg instead,
which isolates the refinement model.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from server import imaging, refine  # noqa: E402
from server.config import load_config  # noqa: E402
from server.params import GenParams, apply_changes, defaults_from_config  # noqa: E402
from server.refine_trace import SETTINGS_FILE, mean_change, small, step_file  # noqa: E402
from server.worker import frame_inputs  # noqa: E402

REPLAY_PREFIX = "replay-step"


def load_batch(folder: Path) -> dict[str, Any]:
    settings = json.loads((folder / SETTINGS_FILE).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or "params" not in settings:
        raise ValueError(f"{folder / SETTINGS_FILE} is not a refinement batch")
    return settings


def batch_params(config: Any, settings: dict[str, Any], overrides: dict[str, Any] | None = None) -> GenParams:
    """The batch's params over this machine's defaults, validated like the admin page's.
    Params Dreamer no longer has (an older batch's) are left out."""
    known = {item.name for item in fields(GenParams)}
    saved = {name: value for name, value in settings["params"].items() if name in known}
    return apply_changes(defaults_from_config(config), {**saved, **(overrides or {})})


def scores(pictures: list[Image.Image]) -> list[tuple[float | None, float]]:
    """(change to the previous step, drift to step 0) per picture, as the worker records them."""
    if not pictures:
        return []
    base = small(pictures[0])
    size = (base.shape[1], base.shape[0])
    rows: list[tuple[float | None, float]] = [(None, 0.0)]
    previous = base
    for picture in pictures[1:]:
        current = small(picture, size)
        rows.append((mean_change(current, previous), mean_change(current, base)))
        previous = current
    return rows


def replay(folder: Path, config: Any, backend: Any, upscaler: Any = None, saved_first: bool = False,
           overrides: dict[str, Any] | None = None) -> list[Image.Image]:
    """Re-run the batch in ``folder`` on ``backend`` (warmed up); returns [drawing picture,
    refined pictures...] as they would be shown, and writes them as replay-step-XX.jpg."""
    settings = load_batch(folder)
    params = batch_params(config, settings, overrides)
    width, height = (int(side) for side in settings["picture_size"])
    background = settings.get("background")
    sketch = imaging.decode_sketch((folder / settings.get("sketch", "sketch.png")).read_bytes(),
                                   (width, height), layered=bool(settings.get("layered", background is not None)))
    init, line_map = frame_inputs(config, sketch, background, params)

    set_size = getattr(backend, "set_size", None)
    if set_size is not None:
        set_size(width, height)
    backend.set_params(params)
    backend.set_prompt(settings["prompt"])
    # A fresh pipeline with latency is primed like the worker does (_process_frame).
    for _ in range(max(0, int(getattr(backend, "latency_frames", 0))) + 1):
        drawing = backend.generate(init, line_map)
    first = drawing
    if saved_first and (folder / step_file(0)).is_file():
        first = Image.open(folder / step_file(0)).convert("RGB")
        if first.size != drawing.size:
            first = first.resize(drawing.size, Image.Resampling.BICUBIC)

    upscales_itself = refine.MODELS.get(params.refine_model, refine.MODELS[refine.DEFAULT_MODEL])[2]
    steps = refine.run(params.refine_model, backend, first, init, line_map,
                       tuple(params.t_index_list), params.refine_steps, upscaler)
    pictures = [drawing]
    try:
        for picture in steps:
            if params.upscale == "x2" and not upscales_itself and upscaler is not None and upscaler.available:
                picture = upscaler.upscale(picture, 2)
            pictures.append(picture)
    finally:
        steps.close()
    for index, picture in enumerate(pictures):
        (folder / step_file(index, REPLAY_PREFIX)).write_bytes(imaging.encode_jpeg(picture, params.jpeg_quality))
    return pictures


def report(folder: Path, pictures: list[Image.Image]) -> list[str]:
    """Lines comparing the recorded steps with the replayed ones."""
    settings = load_batch(folder)
    recorded = {step["index"]: step for step in settings.get("steps", [])}
    replayed = scores(pictures)
    base = small(pictures[0]) if pictures else None
    base_size = (base.shape[1], base.shape[0]) if base is not None else None
    lines = [f"{folder.name}: {settings.get('refine_model')} on {settings.get('backend')}, "
             f"{len(recorded) - 1} recorded / {len(pictures) - 1} replayed steps",
             "step | recorded change / drift | replay change / drift | replay vs recorded"]
    for index in range(max(len(recorded), len(pictures))):
        original = recorded.get(index)
        before = (f"{original['step'] if original['step'] is not None else '-':>5} / {original['drift']:>5}"
                  if original else "      -      ")
        after, versus = "      -      ", "-"
        if index < len(pictures):
            step_change, drift = replayed[index]
            after = f"{step_change if step_change is not None else '-':>5} / {drift:>5}"
            original_path = folder / step_file(index)
            if original_path.is_file():
                with Image.open(original_path) as original_picture:
                    versus = str(mean_change(small(pictures[index], base_size),
                                             small(original_picture, base_size)))
        lines.append(f"{index:>4} | {before:>23} | {after:>21} | {versus}")
    return lines


def make_backend(config: Any) -> Any:
    from server.backends.streamdiffusion import StreamDiffusionBackend

    return StreamDiffusionBackend(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", type=Path, help="a folder in data/refine-batches/")
    parser.add_argument("--config", default=None, help="config.yaml to use (default: as Dreamer finds it)")
    parser.add_argument("--saved-first", action="store_true",
                        help="refine from the saved step-00.jpg instead of a fresh drawing picture")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="override a saved param (JSON value), e.g. refine_steps=6")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a constant in server/refine.py, e.g. ANCHOR_SIGMA=1.5")
    args = parser.parse_args()
    for assignment in args.set:
        name, value = assignment.split("=", 1)
        setattr(refine, name, type(getattr(refine, name))(json.loads(value)))
    overrides = {name: json.loads(value) for name, value in (item.split("=", 1) for item in args.param)}

    folder = args.batch.resolve()
    config = load_config(args.config)
    settings = load_batch(folder)
    params = batch_params(config, settings, overrides)
    if settings.get("backend") != "streamdiffusion":
        print(f"note: recorded on the {settings.get('backend')!r} backend; replaying on the GPU", flush=True)
    backend = make_backend(config)
    width, height = (int(side) for side in settings["picture_size"])
    backend.set_size(width, height)  # before warmup: the pipeline is built at this size
    backend.warmup()
    upscaler = None
    upscales_itself = refine.MODELS.get(params.refine_model, refine.MODELS[refine.DEFAULT_MODEL])[2]
    if params.upscale == "x2" or upscales_itself:
        from server.upscale import Upscaler

        upscaler = Upscaler(REPO_ROOT / config.model.upscaler_model)
        if not upscaler.available:
            print(f"warning: upscaler weights missing ({upscaler.model_path}); not upscaling", flush=True)
    pictures = replay(folder, config, backend, upscaler, args.saved_first, overrides)
    print("\n".join(report(folder, pictures)), flush=True)
    print(f"wrote {len(pictures)} {REPLAY_PREFIX}-XX.jpg to {folder}", flush=True)
    close = getattr(backend, "close", None)
    if close is not None:
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

