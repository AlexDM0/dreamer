# GPU bring-up results (Windows, RTX 3080)

Measured on 2026-10-25 on Alex's PC.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 (10 GB) |
| Driver | 591.59 (`nvidia-smi`: CUDA Version 13.1) |
| OS | Windows 11 Pro (10.0.26200) |
| Python | 3.10 in `.venv` |
| PyTorch | 2.8.0+cu128 (CUDA 12.8), `torch.cuda.is_available()` True |
| StreamDiffusion | Daydream fork @ `4c90d9e437aa28cca7cae1acfab1e52157261939` (0.1.1) |
| TensorRT | 10.12.0.36, cuDNN 9.7.1.26, diffusers 0.36.0.dev0 (fork pin), transformers 4.56.0 |
| Model | `stabilityai/sd-turbo` + `thibaud/controlnet-sd21-scribble-diffusers`, TinyVAE, 512×512 |

## Performance

`.venv\Scripts\python.exe tools\bench_gpu.py`: three synthetic sketches, 10 warm-up frames,
then 60 timed frames per case. Each frame does the same work as the server worker, minus
PNG decoding and the WebSocket: build the line map, run `generate()`, and encode the JPEG.
The safety checker is on (the default). GPU memory is the whole card as `nvidia-smi` reports
it, including about 0.85 GB used by the Windows desktop.

| case | ControlNet | steps | t_index_list | mean ms | median ms | p95 ms | fps | GPU memory used GB |
|---|---|---|---|---|---|---|---|---|
| nocn-1 | off | 1 | [10] | 37.4 | 37.1 | 37.9 | 26.7 | 5.69 |
| nocn-2 | off | 2 | [10, 30] | 47.8 | 47.4 | 49.8 | 20.9 | 5.69 |
| cn-1 | on | 1 | [10] | 64.8 | 64.6 | 65.9 | 15.4 | 7.92 |
| cn-2 | on | 2 | [10, 30] | 84.7 | 84.6 | 87.0 | 11.8 | 7.96 |

Without the safety checker (`--no-safety`), each frame is about 4 ms faster: 32.9 / 43.1 / 60.8 /
80.4 ms. Keep it on.

- **ControlNet** costs about 27 ms per frame (1 step). A second step costs 10 ms without
  ControlNet and 20 ms with it.
- **Through the real server** (`python -m server`, WebSocket kid client on the same PC, one
  frame in flight, the default config with ControlNet and 2 steps), a growing drawing
  streamed 30 frames with a mean round trip of 128 ms (about 8 pictures/s). The first frame
  took 733 ms, because the first frame after start-up primes the pipeline and the prompt is
  encoded for the first time. The extra ~40 ms over the bench comes from PNG encode/decode
  and the socket.
- **Start-up** with cached engines: backend ready in 7.7 s. The first-ever start builds the
  engines: the plain set (UNet, VAE encoder/decoder, safety checker) took about 10 minutes.
  The ControlNet set (ControlNet engine about 4 minutes, plus a ControlNet-capable UNet
  about 10 minutes) took longer.
- **Live step changes** between 1, 2 and 3 steps from the admin API applied without an engine
  rebuild. The engines are built for UNet batch 1 to 4, so 1 to 4 steps all fit.

## Picture quality and settings

- **Scribble polarity: `white_on_black` (the default) is right.** With the same sketch and
  settings, white lines on black make the ControlNet keep the child's lines (house, cat
  outline, flower shape). Black on white follows them clearly less. The model card's example
  image shows a black-on-white drawing, but that is a display image, not the conditioning
  format. See [img/gpu-quality-1step.png](img/gpu-quality-1step.png).
- **The sketch is the outline, not part of the picture.** With the first settings, the child's
  pen strokes showed up directly in the picture. Three reasons: img2img started from the raw
  sketch, the ControlNet was strong (0.8), and the prompt began with "a child's drawing
  turned into a picture". The defaults are now:
  - With ControlNet on, img2img starts from a **blurred sketch** (radius = size/32, so 16 px
    at 512). The child's colours and rough shapes stay, and the lines reach the model only
    through the ControlNet. With ControlNet off, the raw sketch is still used, because it is
    then the only source of structure.
  - **`controlnet_scale: 0.55`** (was 0.8). At 0.35 the drawing is lost. At 0.8 the lines come
    back as thick traced outlines.
  - **`t_index_list: [6, 26]`** (creativity 88 on the admin page, which spaces later steps 20
    apart). The first default `[30, 42]` returned the sketch almost unchanged: mean pixel
    distance 2.5/255.
  - **Prompt head "a colorful illustration"** instead of "a child's drawing turned into a
    picture".

  Results: in the Nature + Cartoon comparison, no row keeps pen strokes. The lines become
  objects, such as a house frame or a tree, and the blurred start reinterprets them most
  freely. See [img/gpu-no-pen-lines.png](img/gpu-no-pen-lines.png). Through the live server
  with the default Ocean topic, the house became an underwater hut, the face a round
  character, and the flower a fish. See [img/gpu-live-ocean.png](img/gpu-live-ocean.png).
  If a parent wants the picture to stick closer to the drawing, "How closely the picture
  follows the lines" on the admin page raises the ControlNet strength live.
- Earlier pictures with the first fixes:
  [img/gpu-quality-2step.png](img/gpu-quality-2step.png) and
  [img/gpu-live-settle.png](img/gpu-live-settle.png). These were made before the change
  above, so the lines are still visible in them.

## Update: iPad-shaped pictures, paper colour, up to 8 steps

Measured later the same day, after these changes: pictures are 640×448 (portrait 448×640)
instead of 512×512, the engines are built for 1 to 8 steps (`model.max_steps: 8`), the start
image is `init_mode: colors` on the paper colour, and the defaults are ControlNet 0.55 and
t_index `[6, 26]`. Backend time per picture with ControlNet, from `tools/make_help_images.py`:

| steps | ms | pictures/s |
|---|---|---|
| 1 | 49 | 20 |
| 2 (default) | 67 | 15 |
| 4 | 108 | 9 |
| 8 | 186 | 5 |

- **GPU memory** with the batch 1-8 engine was about 9.6 GB of the RTX 3080's 10 GB (including
  the desktop's ~0.85 GB), against ~7.9 GB with 1 to 4 steps at 512×512. It stayed flat over
  repeated orientation switches, so there is no leak. The engine is now built for batch 1-4
  only (see below): 8.4 GB.
- **Orientation switch** (a landscape drawing after a portrait one, or the other way round):
  the wrapper is rebuilt at the new size in about 8 s. The engines have dynamic shapes, so they
  are reused, not rebuilt. Tested landscape → portrait → landscape → portrait → landscape.
- **Paper colour carries into the picture:** black paper gives a night scene, blue paper the
  deep sea, yellow paper a sunny one. See the help page (`/static/help.html`) for every control,
  rendered on this PC.

### Steps: up to 4 together, more one after another

The fork runs a frame's steps together as one UNet batch, so the TensorRT engine had to be
built for batch 1-8 for 8 steps, which used ~9.6 GB. Now the engine is built for batch 1-4
(`model.max_batch_steps: 4`), and 5-8 steps switch the stream to the pipeline's
one-after-another loop at batch 1. The wrapper refuses that mode for img2img at construction,
but the pipeline and its parameter updater support it, and switching happens live on a
step-count change. Measured with ControlNet at 640×448:

| steps | how | ms | pictures/s |
|---|---|---|---|
| 1 | together | 51 | 20 |
| 2 (default) | together | 73 | 14 |
| 4 | together | 117 | 9 |
| 5 | one by one | 174 | 6 |
| 6 | one by one | 205 | 5 |
| 8 | one by one | 267 | 4 (batch 1-8 engine: 186 ms) |

GPU memory: 8.4 GB in use, down from 9.6 GB. One-by-one steps also have no frame lag. Pitfall
fixed: that loop re-noises between steps with `torch.randn_like` on the global CUDA
generator, so an unchanged drawing flickered by ~16/255 per frame. The backend re-seeds the
generator with the surprise number before each such frame, which brings it to 0.00/255.

## Update: draw fast, then refine; upscaler; resolution; memory per engine

**How a picture is made now.** While the child draws, each new sketch is answered with the
"draw reaction" steps (1 to `model.max_batch_steps`, run together as one UNet batch). As soon
as the drawing pauses, the worker runs refinement passes at 4, 8, 12, 16 and 20 steps (up to
the "refine" setting; steps run one after another at batch 1). Each pass is a complete picture
of the current drawing and replaces the one shown, so the picture visibly refines; with the AI
upscale on, refined pictures are 2x (1280×896). A new stroke stops refinement after the pass in
progress.

Measured on the live server (640×448, ControlNet, refine 20, upscale x2): the drawing picture
arrives first, then refined pictures at +1.6, +2.0, +2.5, +3.2 and +4.0 s after the stroke; a
new stroke during refinement gets its picture within ~0.2-0.3 s. Backend times per pass:

| | ms |
|---|---|
| refine 4 steps | 153 |
| refine 8 steps | 286 |
| refine 12 steps | 418 |
| refine 20 steps | 681 |
| AI upscale x2 (realesr-animevideov3, 640×448 → 1280×896) | 31 |

The upscaler (Real-ESRGAN compact, BSD-3-Clause, 2.5 MB, `models/`) adds ~0.23 GB.

**GPU memory per engine batch size** (`tools/measure_vram.py`, clean processes, above the
desktop's ~0.8 GB; `static/vram.json` has the full table the admin page estimates from):

| engine built for | 512 px | 640 px | 1024 px | draw speed at 640 px |
|---|---|---|---|---|
| 1 step (`max_batch_steps: 1`) | 5.1 GB | 5.1 GB | 5.2 GB | 47 ms (1 step) |
| up to 2 steps | 8.1 GB | 8.1 GB | 8.2 GB | 68 ms (2 steps) |
| up to 4 steps (default) | 7.3 GB | 7.4 GB | 7.7 GB | 117 ms (4 steps), 53 ms (1 step) |

Memory hardly depends on the picture size (the engines are dynamic up to 1024 px) and is not
monotonic in the batch size: the 1-2 engine needs more than the 1-4 engine (measured twice).
The fork's ControlNet TensorRT engine does not build for a batch range of 1-3 on TensorRT 10.12
("Invalid Engine", reproduced three times, also after deleting the half-built files), and after
that failure the fork carried on and later crashed with a CUDA illegal memory access when the
pipeline was rebuilt. So `model.max_batch_steps` accepts 1, 2, 4 or 8, and the backend now
refuses to start (with a clear message) when the fork reports a failed ControlNet engine.

**Resolution** (long side; backend time with 2 steps): 512 → 56 ms, 640 → 79 ms, 768 → 102 ms,
1024 → 186 ms. Changing it rebuilds the pipeline (a few seconds), not the engines.

## Problems found and fixed

1. **`diffusers_ipadapter` missing.** The fork's TensorRT export imports it even without
   IP-Adapter, but only its `[ipadapter]` extra installs it (along with insightface, which
   does not build on Windows without MSVC). The first engine build failed with
   "Acceleration has failed". setup.ps1 / setup.sh now install just that package, at the
   fork's pin.
2. **ControlNet silently ignored.** The fork's UNet engine path has no ControlNet in its key.
   After a run without ControlNet, a ControlNet run loaded that UNet: it logged
   "Including ControlNet support", ran the ControlNet engine (about 27 ms per frame) and
   threw its output away. A mismatched or all-black control image gave bit-identical
   pictures. The backend now builds into `engines\plain\` and `engines\controlnet\`.
3. **Pictures one frame late with 2+ steps.** StreamDiffusion's denoising batch returns the
   picture for the sketch sent `steps - 1` frames earlier, and the first frame came back
   black. So the picture shown after the child stopped drawing was always for the
   previous sketch. The backend now reports `latency_frames`. The worker primes a fresh
   pipeline (at start, after a blank canvas or after an error) with that many extra runs,
   and once no new sketch has come for 150 ms it re-runs the last sketch and re-sends the
   picture with the same seq. The kid page accepts a same-seq picture as a replacement.
   While drawing, this costs nothing extra.
4. **`run.ps1` died on the server's first log line** whenever its output was redirected (a
   log file, a scheduled task) in Windows PowerShell 5.1. It now lets the exit code decide,
   as setup.ps1 already did.
5. **Upload screening called the safety checker wrongly.** The real checker is
   `NSFWDetectorEngine(tensor, threshold) -> bool` (TensorRT only). Turning the safety filter
   on live, when the server started without it, would crash every frame because the wrapper
   then has no checker. That change is now refused with an error (and reverted) instead.
6. Two gallery tests failed on Windows. One was a real bug: on a PC booted less than the
   10 s throttle ago, the first draft autosave waited, because `time.monotonic()` counts from
   boot. The other was a race in the test.

## Tests

`.venv\Scripts\python.exe -m pytest -q`: 228 passed (Windows, mock backend). The Playwright
e2e script (`tests/e2e/run.mjs`) was not run, because Playwright is not installed on this PC.

## Not done / open

- One CUDA "illegal address" crash happened once, during a long probe that changed settings
  many times. It did not reproduce: two re-runs of the same probe, plus 45 rounds and 300
  frames of live changes (1 to 3 steps, ControlNet on and off, strength 0.55 to 0.9), all
  passed. If it happens in the server, restart Dreamer: a CUDA context error cannot be
  recovered inside the process.
- Not tested on a real iPad yet (only from the PC: WebSocket client and desktop browser).
- Windows Firewall was not changed. To let the iPad connect, run this once from an
  **administrator** PowerShell:

  ```powershell
  New-NetFirewallRule -DisplayName "Dreamer" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow -Profile Private
  ```
