# Refinement models: overnight experiment plan

Written 2026-10-25 after the GPU measurements in `36c5c7f`: continuing SD-Turbo's own denoising
path is exact (draw 1 + refine 3 == 4 steps) but drifts from about 6 steps, and at 12–20 steps
it paints a different scene. This file lists the approaches to try, how to judge them, and the
sources. Each approach becomes a selectable **refinement model** on the admin page so Alex can
compare them on the iPad.

## Alex's rule of thumb

> The first thing you see should not change in broad terms, but detail is added every step.

Turned into two numbers, measured per refinement step against the **first picture shown when
the pen stopped** (call it P0):

| Metric | How | Pass |
|---|---|---|
| Structure drift | Mean absolute difference (0–255) between P0 and step k after both are downscaled to 1/16 size (40×28 at 640×448), so only layout and big colour areas count | Stays under ~6 at every step, and never jumps by more than ~2 between steps |
| Detail gain | Variance of the Laplacian (high-frequency energy) of step k, relative to P0 | Rises with k, or at least never falls |
| Step size | Mean absolute difference between step k-1 and step k at full size | Shrinks with k |

**Time budget (Alex, 20:24):** a refinement step may take up to **500 ms**, as long as it
refines and does not replace. That is about 15× an SD-Turbo step (33 ms), so heavier options
are in: more ControlNets at once, a hires pass at 1024 px every step (R5), and a bigger or
slower model for refinement only (R6), as long as memory fits in 10 GB.

Plus a strip of pictures per drawing (P0, then each step) for the eye. Use the same 6 test
drawings for every model (house, cat, rocket, dinosaur, flower, a scribble-only one), on black
and on white paper, 2 topics.

## Refinement models to try

R1 to R5 use only what is already installed (SD-Turbo, the SD 2.1 scribble ControlNet,
Real-ESRGAN). R6 and R7 need new weights.

### R1. `turbo-path` (current baseline)
Continue the drawing's own SD-Turbo path, capped at 4 steps total. Measured: clean up to 4,
drifts from ~6. Keep it as the reference row.

### R2. `low-freq-lock`
Run any refinement (R1 at up to 20 steps, or R3), but after every step replace the
**low frequencies** of the result with those of P0: `out = blur(P0) + (step − blur(step))`,
with a Gaussian of about 1/24 of the picture width. Doing it in latent space (on the 4-channel
latent before decode, blur of ~3 latent px) is cheaper and avoids colour halos; try both.
Broad layout and colour are then fixed by construction and only detail can change. Risk: if the
model wants to move an edge, the old and new edge both show (ghosting); a smaller blur or
blending only in flat regions helps.

### R3. `polish-loop`
After the 4-step path, repeat img2img from the latest picture at a **fixed low noise**
(t_index 46–48, about 19–28% noise), 1 step each, with a new noise sample per round, ControlNet
on. Each round can only touch fine detail. The RC run with fixed noise went flat and
poster-like; this variant uses fresh noise per round and a very late start. Watch for
over-sharpening and colour saturation creeping up; stop when step size falls below ~1/255.

### R4. `self-lines`
For refinement, swap the ControlNet input from the child's scribble to a **soft edge map of
P0** (Canny or an HED-style map of the picture itself, white on black, the format the scribble
ControlNet expects). The model is then held to what is already on screen, not to the rough
pen lines, so it can add detail inside the shapes without redrawing them. Combine with R3. The
same ControlNet engine is used, so no extra memory. thibaud also has an SD 2.1 **HED**
ControlNet (listed in `thibaud/controlnet-sd21`; check the exact diffusers repo name), which fits this better than scribble but
costs a second ControlNet engine (~0.7–2 GB, to measure).

### R5. `hires-pass`
The classic "hires fix": upscale P0 2× with Real-ESRGAN, then one or two img2img steps at low
noise (t_index 44–47) at the bigger size, ControlNet on with the upscaled line map. The upscaler
invents texture, the diffusion pass makes it coherent. The engines are dynamic only up to
1024 px per side, so run at 1024×704 (landscape) / 704×1024 (portrait), not 1280×896. Cost: one
big-picture step is roughly 2.5× a normal one (186 ms at 1024 with 2 steps, measured).

### R6. `sd15-tcd` (new model)
SD 1.5 with a distilled LoRA made for **many steps**: TCD (Trajectory Consistency
Distillation). Its paper reports that LCM's quality drops as steps increase, while TCD "yields
more detailed results compared to the teacher model at high NFEs"; its `gamma` parameter sets
how much fresh noise each step adds (higher gamma, more fine detail; default 0.2). Candidates:
`h1t/TCD-SD15-LoRA`, or ByteDance's Hyper-SD unified SD1.5 LoRA used with the TCD scheduler
(their card: lower eta gives more detail for multi-step runs). Needs the SD 1.5 scribble
ControlNet (`lllyasviel/control_v11p_sd15_scribble`) and new TensorRT engines. The Daydream
fork supports SD 1.5 via LCM-LoRA; TCD support has to be checked in its code.

Do this one **offline first**: a plain diffusers script, no StreamDiffusion, same test
drawings, 4/8/12/20 steps, to see if it passes the rule of thumb at all before spending engine
builds. Then decide whether it becomes the refinement model only (SD-Turbo for drawing, SD 1.5
for refinement, if both fit in 10 GB) or the whole pipeline (slower drawing, but also opens
IP-Adapter for "style from a picture").

### R7. `temporalnet` (later)
The Daydream fork ships configs for TemporalNet v2, a ControlNet that conditions on the
previous frame to keep video frames consistent. It exists for SD 1.5, not SD 2.1, so it only
applies together with R6. Listed so it is not forgotten.

## ControlNet alternatives (Alex's question)

Worth checking, because the ControlNet is the biggest cost today: about 27 ms of the 49 ms
1-step frame, and about 2.2 GB (`docs/gpu-results.md`).

| Option | What it is | For SD 2.1 / SD-Turbo? | Why it could help |
|---|---|---|---|
| ControlNet-XS (CVL Heidelberg) | A much smaller control network | SD 2.1 weights exist for **depth** (~14M parameters per its config name), edges only for SDXL | Far cheaper per step. No scribble/edge model for SD 2.1, so it would need training; park it |
| T2I-Adapter (TencentARC) | Runs **once per picture**, not once per step | Sketch adapters exist for SD 1.4/1.5 and SDXL; none confirmed for SD 2.1 | Nearly free with many steps; fits R6. Daydream fork does not list it, so it needs code |
| thibaud SD 2.1 HED / Canny ControlNets | Same architecture, other inputs | Yes | HED of P0 for R4 |
| Lower ControlNet strength during refinement | One setting | Yes | The lines matter less once P0 exists; test 0.55 → 0.3 → 0 for the polish steps |

Recommendation: keep the scribble ControlNet for drawing. For refinement, test R4 (self-lines)
and lower strength first, since they are free. Look at T2I-Adapter only if R6 wins.

## Order for tonight

1. Add a `refinement_model` setting (admin select, live) and the metrics script
   (`tools/refine_eval.py`: test drawings in, metric table and picture strips out).
2. R1 (baseline), R2, R3, R4, R3+R4, R5: all on the current engines, minutes each.
3. R6 offline in diffusers: download about 2 GB of SD 1.5 weights plus the LoRA and the SD 1.5
   scribble ControlNet. Only if it passes, build engines.
4. Write the results into this file: the metric table per model and one picture strip each.
   The ones that pass go on the admin page for Alex to try on the iPad.

## Sources

- TCD paper: [arXiv 2402.19159](https://arxiv.org/pdf/2402.19159v1)
- Hyper-SD model card: [huggingface.co/bytedance/Hyper-SD](https://huggingface.co/bytedance/Hyper-SD)
- Daydream StreamDiffusion fork (TemporalNet configs, latent processors, Real-ESRGAN TRT, LCM-LoRA): [github.com/daydreamlive/StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion)
- thibaud SD 2.1 ControlNets: [huggingface.co/thibaud/controlnet-sd21](https://huggingface.co/thibaud/controlnet-sd21)
- ControlNet-XS: [huggingface.co/CVL-Heidelberg/ControlNet-XS](https://huggingface.co/CVL-Heidelberg/ControlNet-XS)
- T2I-Adapter: [github.com/TencentARC/T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)
- SD-Turbo model card ("1 to 4 steps"): [huggingface.co/stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)

## Results (2026-10-25 night, RTX 3080)

Implemented in `server/refine.py` (selectable live as "Refinement model" on the admin page) and
measured with `tools/refine_lab.py` on the six real test drawings from the showcase run (house,
cat, rocket, dinosaur, castle, portrait meadow), draw reaction 1 step, refine budget 8. Full
per-drawing numbers: `refinement-results.json`; sheets: `img/refine-<drawing>.png` (rows =
models, columns = P0 then each step) and `img/refine-<drawing>-zoom.png` (centre crop of P0 and
the last picture).

| model | max drift (≤6) | final detail (×P0) | passes | step sizes | ms/step | what it is |
|---|---|---|---|---|---|---|
| path (R1) | 35.8 | 1.14 | 0/6 | 27.9 → 2.0 | 51 | continue the SD-Turbo path |
| path_locked (R2, pixels) | 7.3 | 1.09 | 0/6 | 15.0 → 1.4 | 69 | R1 with P0's low frequencies |
| sdedit (R3 variant) | 6.9 | 1.39 | 0/6 | 10.2 → 3.9 | 45 | each rung from P0, steady noise |
| **anchored** (R2 latent + ladder) | **5.6** | **1.75** | 2/6 | 9.0 → **0.9** | **44** | single-step rungs at t 38→48, chained, latent low-pass pinned to P0 (σ 2.0), TCD-style noise γ 0.05 |
| anchored_edges (+R4) | 5.8 | 1.53 | 3/6 | 9.5 → 1.0 | 46 | anchored, ControlNet also sees P0's edges |
| hires_tiles (R5) | 7.6 | 1.61 | 1/6 | 9.3 → 2.2 | 295 | upscale 1.5x, anchored tiles (4 × 640×448) |
| anchored_hires | 8.2 | 1.79 | 1/6 | 9.0 → 2.2 | 175 | 3 anchored rungs, then tiled 1.5x rungs |

"passes" = max drift < 6 **and** detail never falls; most anchored runs miss only the second
part by a hair (detail flat after the first rung rather than rising).

What we learned:
- **Continuing the path (R1) repaints the scene** (drift 36; in the zoom crops the cat's face is
  redrawn). SD-Turbo is not a many-step model; the research agrees (model card: 1-4 steps).
- **Locking low frequencies works, in latent space more than in pixels.** Pixel locking keeps
  the layout but the detail still decays because the underlying path drifts.
- **anchored is the best fit to the rule**: lowest drift, most detail, shrinking steps, 44 ms per
  rung. Tuning: σ 1.5-4 and γ 0-0.45 swept; σ 2.0 / γ 0.05 balanced best; starting the ladder at
  t 38 instead of 34 lowered drift from 6.1 to 5.6 at the same detail. It is now the default.
- **Outlines from P0 in the ControlNet (R4)** lower drift slightly but take some detail away.
- **Tiled hires (R5)** adds real resolution and detail rises every rung, but costs ~300 ms per
  rung and drifts a little more (the 1.5x resample). Anchoring every tile to the upscaled P0
  (not to the previous rung) was essential: before, drift grew 6 → 13 over three rungs.
- The first rung is still the most visible change (~9/255 average), mostly sharpening the soft
  1-step drawing picture.

### ControlNet alternatives for drawing (measured)

`tools/controlnet_lab.py` ran the six test drawings through each SD 2.1 ControlNet from thibaud
(1-step drawing picture, then the anchored refinement). Sheets: `img/controlnet-<drawing>.png`
(rows = variant; columns = sketch, drawing picture, refined); numbers: `controlnet-results.json`.

| variant (name:strength:dilate) | ms/frame (1 step, 640×448) | GPU memory in use | look |
|---|---|---|---|
| **scribble:0.55:1** (current) | 51.5 | 7.6 GB | pen lines become solid, coloured objects |
| lineart:0.55:0 | 52.7 | 7.6 GB | traces the pen strokes as glowing tubes; houses go see-through |
| lineart:0.4:0 | 52.9 | 7.6 GB | a little softer, same traced-tube look |
| hed:0.55:0 | 53.7 | 7.6 GB | follows the lines closely, hazy halos around shapes |
| hed:0.4:0 | 53.7 | 7.6 GB | best of the alternatives; still ghostlier than scribble |

All three ControlNets cost the same, so speed decides nothing. Lineart and HED are trained on
thin edges of real images and read a child's thick pen strokes as outlines to reproduce, so
they draw the strokes instead of the thing. Scribble is trained on rough strokes and turns them
into objects, which is what a drawing game needs. **Scribble stays.** ControlNet-XS and
T2I-Adapter have no SD 2.1 sketch weights (see the table above), so there is no cheaper drop-in.

### R6: SD 1.5 + TCD, offline (measured)

`tools/tcd_lab.py`: plain diffusers (no TensorRT), SD 1.5 + `h1t/TCD-SD15-LoRA` + the SD 1.5
scribble ControlNet (`lllyasviel/control_v11p_sd15_scribble`), same start image, line map and
seed per drawing, img2img strength 0.88, gamma 0.3, at 2/4/8/12/20 steps. Sheets:
`img/tcd-<drawing>.png`; numbers: `tcd-results.json` (drift and detail are measured against the
2-step picture).

| steps | 2 | 4 | 8 | 12 | 20 |
|---|---|---|---|---|---|
| look | blurry blobs | soft, still blurry | clean, appealing | same scene as 8, a bit more detail | starts to wander |
| ms per picture (diffusers, no TensorRT) | ~220 | ~335 | ~570 | ~740 | ~1170 |

Peak GPU memory 3.5 GB. What it means:
- **Not for drawing.** At 1-4 steps the pictures are blurry (with strength 0.88, "2 steps" runs
  1 real step), and even 4 steps take ~335 ms untuned. SD-Turbo stays the drawing model.
- **8 → 12 steps keeps the scene** (step size ~9-15, like the anchored ladder's first rung) and
  looks cleaner and more "illustrated" than SD-Turbo; 20 steps drifts more. So TCD does what its
  paper says: many steps add detail instead of falling apart.
- **As a refiner after SD-Turbo it would repaint**: it is a different model, so its picture
  differs from the SD-Turbo picture already on screen (drift 26-40 even between its own 2 and
  8 steps). It could only fit the rule as an anchored ladder on the SD-Turbo picture (both
  models share the same VAE latent space), which needs a second UNet (+~2 GB) and TensorRT
  engines. Possible, but a large step for a gain the anchored model mostly already gives.

Recommendation: keep SD-Turbo + scribble + anchored. Revisit SD 1.5 + TCD only if Alex wants a
different, more "illustrated" look overall, as a whole-pipeline swap (it would also unlock
IP-Adapter for "style from a picture"), not as a refiner.

### ControlNet strength during refinement only (measured)

`tools/refine_lab.py --models anchored --refine-cn <scale>` (drawing at 0.55, refinement at the
given strength), six drawings:

| refinement ControlNet | max drift | final detail | ms/rung |
|---|---|---|---|
| 0.55 (current) | 5.6 | 1.75 | 48 |
| 0.3 | 5.6 | 1.80 | 47 |
| 0 (off) | 5.7 | 1.85 | 39 |

The zoom crops are practically identical: the latent anchor already holds the shapes, so the
ControlNet adds little once the first picture exists. Turning it off would save ~9 ms per rung,
but switching the ControlNet config back on for the next stroke costs a stream update right
when the child draws again, so the default stays 0.55.

### SD 1.5 + TCD as a picture model

**Removed on 2026-10-27.** SD 1.5 + TCD was tried as a live picture model and removed: adding
steps changed the picture too much (see the slideshow comparison). The notes below are history.

Now selectable live on the admin page ("Picture model", `server/backends/tcd.py`): plain
diffusers, strength 1.0, 4 drawing steps; refinement re-renders the same seed at 6/8/12/16/20
steps. Measured on the RTX 3080: ~0.29 s (2 steps), ~0.41 s (4), ~0.52 s (6), ~0.62 s (8) per
picture; VAE decode ~80 ms of that, UNet + ControlNet ~65 ms per step; safety check 13 ms;
switching models 6-10 s. Faster would need TensorRT engines for SD 1.5 (the fork supports SD
1.5 via its LCM path; TCD's scheduler would have to be added).
