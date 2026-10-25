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
- **Default creativity changed from `[30, 42]` to `[10, 30]`.** At t_index 30 and above, SD-Turbo
  returns the sketch almost unchanged (mean pixel distance 2.5/255 without ControlNet).
  At 10 it makes a real illustration that still follows the lines. Among the 2-step options,
  `[6, 26]` drifts too far from the drawing, `[10, 30]` is the best balance, and `[14, 34]` and
  above barely change the sketch. See [img/gpu-quality-2step.png](img/gpu-quality-2step.png).
  The admin page's creativity slider spaces later steps 20 apart, so the default (creativity
  80, 2 steps) maps to `[10, 30]`.
- With the Ocean topic, a drawn circle became a fishbowl full of fish, and its outline was
  kept exactly. See [img/gpu-live-settle.png](img/gpu-live-settle.png).

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

- Not tested on a real iPad yet (only from the PC: WebSocket client and desktop browser).
- Windows Firewall was not changed. To let the iPad connect, run this once from an
  **administrator** PowerShell:

  ```powershell
  New-NetFirewallRule -DisplayName "Dreamer" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow -Profile Private
  ```
