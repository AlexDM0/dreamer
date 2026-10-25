# Dreamer: fact sheet

Every number here was measured on the development PC (Windows 11, NVIDIA RTX 3080 10 GB,
driver 591.59) on 2026-10-25. Each fact says where it comes from, so a slide can be checked.
Numbers marked *approx.* vary from run to run by a few percent.

## What it is (one line)

A drawing game for young children: they draw on an iPad, and a PC at home turns the drawing
into a picture while they draw, then keeps polishing it when they stop. Parents choose what
the pictures can be about.

## The experience

| Fact | Value | Source |
|---|---|---|
| First picture after a stroke (live server, default settings) | about 0.1-0.2 s round trip once running; the very first picture after start-up took 733 ms | `docs/gpu-results.md` (128 ms mean round trip at 512 px, 2 steps; first frame 733 ms) |
| Pictures per second while drawing (backend only, 640×448, ControlNet, 2 steps, the default draw reaction) | about 12-15 *approx.* (67-80 ms) | `static/help/data.json` "steps", `docs/gpu-results.md` (67, 73 and 79 ms in its tables) |
| Refinement after the child stops (default "anchored" model) | after a 150 ms pause, up to 6 single-step rungs of about 44 ms each: roughly 0.3 s, so the picture is polished within half a second | `docs/research/refinement-models.md` Results (44 ms/step), `server/refine.py` `ANCHORED_RUNGS`, default `refine_steps` 8 |
| Superseded: the earlier "continue the path" schedule | 4 → 8 → 12 → 16 → 20 steps, 153 / 286 / 418 / 681 ms per pass, finished about 4 s after the stroke. Do not quote these for the current app | `docs/gpu-results.md`, "draw fast, then refine" |
| New stroke during refinement | refinement stops after the step in progress; measured with the earlier (slower) schedule, its picture arrived within about 0.2-0.3 s | `docs/gpu-results.md` |
| AI upscale of refined pictures (off by default) | 2x (640×448 → 1280×896) in 31 ms. The showcase screenshots were made with it on | `docs/gpu-results.md`, `static/help/data.json` "upscale", `screenshots/manifest.json` |
| Picture shape | the iPad's shape: 640×448 landscape, 448×640 portrait (resolution adjustable 512-1024 px) | `server/config.py` `canvas_dims`, `SPEC-addendum-canvas.md` |
| Switching between a landscape and a portrait drawing | about 8 s pause (pipeline rebuild, engines reused) | `docs/gpu-results.md` |
| Start-up with prepared engines | about 8-10 s (7.7 s in gpu-results, 9-10 s in vram.json) | `docs/gpu-results.md`, `static/vram.json` "warmup_s" |
| First-ever start (building the TensorRT engines) | roughly 10-25 min, once | `docs/gpu-results.md` |

## Safety and privacy

| Fact | Source |
|---|---|
| Children never type: every prompt is built on the PC from topics and styles the parent allows, always ending with a fixed child-friendly phrase. | `server/styles.py` (`PROMPT_HEAD`, `SAFETY_TAIL`, `build_prompt`) |
| Every picture is checked by an image safety classifier (Freepik/nsfw_image_detector, as a TensorRT engine) before it is shown; if it is flagged, the previous picture stays. It costs about 4 ms per picture. | `server/backends/streamdiffusion.py`, `docs/gpu-results.md` |
| Style pictures uploaded by the parent are screened by the same checker while the safety filter is on; with the filter off, uploads are stored unscreened. | `server/backends/streamdiffusion.py` `check_image_safe`, `requirements-gpu.md` |
| Everything runs on the home network: one Python process on the PC; no account. Drawings and pictures never leave the PC. Models are downloaded once, on the first start (the setup script only fetches the 2.5 MB upscaler); after that Dreamer runs without contacting the internet (`HF_HUB_OFFLINE` once the models are cached). | `docs/setup.md`, `setup.ps1`, `requirements-gpu.md` |
| The parent page is protected by a PIN, with a growing lock-out after wrong tries. | `docs/setup.md` "Admin page", `server/auth.py` |

Be careful with wording: the safety checker is a classifier and can make mistakes; say
"checks every picture", not "guarantees".

## Under the hood

| Fact | Value | Source |
|---|---|---|
| Picture model | SD-Turbo (`stabilityai/sd-turbo`), via the Daydream StreamDiffusion fork | `config.example.yaml`, `requirements-gpu.md` |
| Following the drawing | scribble ControlNet (`thibaud/controlnet-sd21-scribble-diffusers`) on a white-on-black line map | `docs/gpu-results.md` "Scribble polarity" |
| Start image | soft colour clouds of the child's strokes on the paper colour, so the lines guide the picture without being traced | `server/imaging.py` `make_init_image`, `SPEC-addendum-params.md` |
| Acceleration | NVIDIA TensorRT engines (UNet, ControlNet, VAE, safety checker) | `docs/gpu-results.md` |
| Steps | draw reaction 2 steps by default (1 to `max_batch_steps`, default 4), run together as one batch; refinement runs one step at a time (budget up to 20; the anchored model uses at most 6 rungs) | `docs/gpu-results.md` hybrid section, `server/refine.py` |
| Upscaler | Real-ESRGAN compact model `realesr-animevideov3` (BSD-3-Clause), 2.5 MB | `server/upscale.py` |
| GPU memory (above the desktop) | 5.1 GB with engines for 1 step, 7.4 GB for up to 4 steps (default) | `static/vram.json`, `docs/gpu-results.md` |
| Kid page | plain HTML/CSS/JavaScript, no framework; the paper is a background colour behind a transparent stroke layer | `static/`, `SPEC-addendum-canvas.md` |
| Tests | 256 automated tests (mock backend) | `pytest -q` |

## What it is not (yet)

- Used on one real iPad at home so far; most testing was in a desktop browser and with a
  WebSocket test client.
- One child at a time.
- Needs a Windows PC with an NVIDIA RTX graphics card (tested on an RTX 3080, 10 GB).
- Uploaded "style pictures" are stored but do not change the picture yet.
