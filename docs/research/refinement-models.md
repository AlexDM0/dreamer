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
