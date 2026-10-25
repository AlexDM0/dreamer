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
| First picture after a stroke (live server, default settings) | about 0.1-0.2 s round trip once running; the very first picture after start-up takes about 0.5-0.75 s | `docs/gpu-results.md` (128 ms mean round trip at 512 px, 2 steps; refine test at 640 px: first picture at +0.52-0.75 s) |
| Pictures per second while drawing (backend only, 640×448, ControlNet, 2 steps) | about 13-15 *approx.* (67-78 ms) | `static/help/data.json` "steps", `docs/gpu-results.md` hybrid table |
| Refinement after the child stops | 4 → 8 → 12 → 16 → 20 steps, finished about 4 s after the last stroke (with 2x upscale) | `docs/gpu-results.md`, "draw fast, then refine" |
| Time per refinement pass | 153 ms (4 steps), 286 ms (8), 418 ms (12), 681 ms (20) | `docs/gpu-results.md` |
| New stroke during refinement | its picture arrives within about 0.2-0.3 s | `docs/gpu-results.md` |
| AI upscale of refined pictures | 2x (640×448 → 1280×896) in 31 ms | `docs/gpu-results.md`, `static/help/data.json` "upscale" |
| Picture shape | the iPad's shape: 640×448 landscape, 448×640 portrait (resolution adjustable 512-1024 px) | `server/config.py` `canvas_dims`, `SPEC-addendum-canvas.md` |
| Switching between a landscape and a portrait drawing | about 8 s pause (pipeline rebuild, engines reused) | `docs/gpu-results.md` |
| Start-up with prepared engines | about 8 s (7.5-7.7 s measured) | `docs/gpu-results.md`, `static/vram.json` "warmup_s" |
| First-ever start (building the TensorRT engines) | roughly 10-25 min, once | `docs/gpu-results.md` |

## Safety and privacy

| Fact | Source |
|---|---|
| Children never type: every prompt is built on the PC from topics and styles the parent allows, always ending with a fixed child-friendly phrase. | `server/styles.py` (`PROMPT_HEAD`, `SAFETY_TAIL`, `build_prompt`) |
| Every picture is checked by an image safety classifier (Freepik/nsfw_image_detector, as a TensorRT engine) before it is shown; if it is flagged, the previous picture stays. It costs about 4 ms per picture. | `server/backends/streamdiffusion.py`, `docs/gpu-results.md` |
| Style pictures uploaded by the parent are screened by the same checker. | `server/backends/streamdiffusion.py` `check_image_safe` |
| Everything runs on the home network: one Python process on the PC; no account, no cloud service. Model files are downloaded once from Hugging Face / GitHub during setup and first start. | `README.md`, `requirements-gpu.md` |
| The parent page is protected by a PIN, with a growing lock-out after wrong tries. | `README.md` "Admin page", `server/auth.py` |

Be careful with wording: the safety checker is a classifier and can make mistakes; say
"checks every picture", not "guarantees".

## Under the hood

| Fact | Value | Source |
|---|---|---|
| Picture model | SD-Turbo (`stabilityai/sd-turbo`), via the Daydream StreamDiffusion fork | `config.example.yaml`, `requirements-gpu.md` |
| Following the drawing | scribble ControlNet (`thibaud/controlnet-sd21-scribble-diffusers`) on a white-on-black line map | `docs/gpu-results.md` "Scribble polarity" |
| Start image | soft colour clouds of the child's strokes on the paper colour, so the lines guide the picture without being traced | `server/imaging.py` `make_init_image`, `SPEC-addendum-params.md` |
| Acceleration | NVIDIA TensorRT engines (UNet, ControlNet, VAE, safety checker) | `docs/gpu-results.md` |
| Steps | up to 4 run together as one batch while drawing; refinement runs step by step (up to 20) | `docs/gpu-results.md` hybrid section |
| Upscaler | Real-ESRGAN compact model `realesr-animevideov3` (BSD-3-Clause), 2.5 MB | `server/upscale.py` |
| GPU memory (above the desktop) | 5.1 GB with engines for 1 step, 7.4 GB for up to 4 steps (default) | `static/vram.json`, `docs/gpu-results.md` |
| Kid page | plain HTML/CSS/JavaScript, no framework, two canvas layers (paper + strokes) | `static/`, `docs/design-system.md` |
| Tests | 247 automated tests (mock backend) | `pytest -q` |

## What it is not (yet)

- Not tested on a real iPad yet at the time of writing (driven from a desktop browser and a
  WebSocket test client).
- One child at a time.
- Needs a Windows PC with an NVIDIA RTX graphics card (tested on an RTX 3080, 10 GB).
- Uploaded "style pictures" are stored but do not change the picture yet.
