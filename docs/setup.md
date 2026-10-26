# Dreamer: setup and technical guide

Dreamer is a drawing game for young kids. A child draws on an iPad in Safari, and the
drawing streams over the home Wi-Fi to a Windows PC with an NVIDIA GPU. The PC runs
[StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion) (SD-Turbo + a scribble
ControlNet) and sends a picture back while the child draws. A parent picks topics and
styles from a phone on a PIN-protected admin page. Children never type prompts: every
prompt is built on the server from parent-approved topics and styles, with a fixed
kid-friendly safety tail.

Everything runs on your LAN from one Python process. Drawings and pictures never leave
the PC. Models are downloaded once, on the first start; after that Dreamer runs without
contacting the internet (once the models are cached, it sets `HF_HUB_OFFLINE`).

Back to the [product page](../README.md). How the pieces fit together:
[showcase/architecture.md](showcase/architecture.md).

## Quick start (Windows PC with an RTX GPU)

1. Install [Python 3.10](https://www.python.org/downloads/windows/) (64-bit, keep the *py
   launcher* option), [Git for Windows](https://git-scm.com/download/win) and a current
   NVIDIA driver (CUDA 12.8 or newer). Keep about 15 GB of disk space free.
2. In PowerShell, from the Dreamer folder, run the setup script once:

   ```powershell
   .\setup.ps1
   ```

   It creates `.venv` and installs everything into it (server packages, CUDA PyTorch, the
   StreamDiffusion fork and TensorRT), checks that PyTorch sees the GPU, downloads the
   small upscaler model (`models/realesr-animevideov3.pth`, 2.5 MB) and creates
   `config.yaml`. The picture models themselves are not downloaded yet; that happens on
   the first start. Nothing is installed into your system Python. Options: `-Mock` (no GPU
   stack), `-Force` (recreate `.venv`), `-Python <path or "py -3.10">`, `-Cuda cu128`,
   `-StreamDiffusionRef <ref>` (another fork commit or branch).
   [requirements-gpu.md](../requirements-gpu.md) lists the same steps for a manual install.

   If PowerShell says *running scripts is disabled*, allow local scripts once with
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. If you downloaded the repo as a
   ZIP, Windows also marks the files as "from the internet"; unblock them from the repo
   folder with `Get-ChildItem -Recurse | Unblock-File`. Alternatively, run
   `powershell -ExecutionPolicy Bypass -File .\setup.ps1` (and the same for `run.ps1`).
3. Set `admin_pin` in `config.yaml`.
4. Start it:

   ```powershell
   .\run.ps1
   ```

   The first run downloads the picture models from Hugging Face (a few GB, cached in
   `%USERPROFILE%\.cache\huggingface`) and builds TensorRT engines into `engines\`. That
   takes roughly 10 to 30 minutes, once; the kid page shows "busy" meanwhile. Later starts
   take about 8-10 seconds (7.7 s in [gpu-results.md](gpu-results.md), 9-10 s in
   `static/vram.json`) and run offline.
5. On the iPad, open `http://<pc-ip>:7860/` in Safari and choose *Add to Home Screen*.
   The parent page is `http://<pc-ip>:7860/admin`. `run.ps1` prints both URLs at startup.
   Allow the port through the firewall once (see [LAN access and firewall](#lan-access-and-firewall)).

The optional AI upscaler (off by default) needs `models/realesr-animevideov3.pth`
(Real-ESRGAN compact, BSD-3-Clause, 2.5 MB), which the setup script downloads. If that
download failed, AI upscale stays unavailable; put the file in `models/` by hand.

## Mock mode (development, no GPU)

The `mock` backend needs no torch or GPU. It turns each sketch into a tinted, posterized
picture, so you can work on the whole app on any machine:

```bash
./setup.sh --mock        # Linux / WSL: .venv + server packages, config.yaml with backend: mock
./run.sh                 # http://localhost:7860/
```

On Windows use `.\setup.ps1 -Mock` and `.\run.ps1`. Mock mode accepts Python 3.10 or newer;
the GPU install needs exactly 3.10. `./setup.sh` without `--mock` installs the GPU stack on
Linux/WSL with the same flags as `setup.ps1` (`--force`, `--python PATH`, `--cuda TAG`).

## Configuration

Every key is optional. The keys are documented in
[`config.example.yaml`](../config.example.yaml). The server reads `config.yaml` from the
current directory, or the file named by the `SKETCH_CONFIG` environment variable. An
unknown key logs a warning. An invalid value stops the server with exit code 2.

Most picture settings can also be changed live on the admin page; those are saved in
`data/params.json` and override the `model:` defaults from `config.yaml`.

## LAN access and firewall

The server listens on `0.0.0.0:7860`. Windows blocks inbound connections by default, so
allow the port on private networks once, from an **administrator** PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Dreamer" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow -Profile Private
```

On the first start Windows may also ask whether Python may use the network: click
**Allow** for private networks. If that prompt was dismissed, Windows creates a *Block*
rule for `python.exe`, which wins over the Allow rule above and the iPad cannot connect.
Open `wf.msc`, go to *Inbound Rules*, and delete any blocking rules named "python".

Make sure your home Wi-Fi is set to the *Private* network profile. The app uses plain
HTTP and has no accounts, so don't expose it to the internet.

## Admin page

Open `/admin` on your phone and enter the PIN (4 to 12 digits; write it in quotes in
`config.yaml`). On the iPad, tapping the Dreamer icon in the toolbar opens the same PIN
pad and then the admin page. Five wrong tries lock the PIN pad for 30 seconds; each
further wrong try doubles the wait (up to 15 minutes) until the right PIN is entered. The
login lasts 30 days in that browser; changing the PIN logs every device out. On the admin
page you can:

- **Now:** watch the live picture, the engine state and speed, and change the current
  topic, style, the iPad's starting view and the drawing page look (dark or light).
  Changes show on the iPad immediately.
- **Picture controls:** creativity, the draw reaction (steps while drawing, 1 to
  `model.max_batch_steps`, default 4; 2 steps by default), "refine when they stop" (once
  the drawing has paused for 150 ms, the picture keeps improving step by step, 0 to 20;
  default 8), the refinement model (how those steps are made; default "D: Anchored detail
  ladder": up to 6 single-step rungs of about 44 ms each, so roughly 0.3 s of refinement;
  see [research/refinement-models.md](research/refinement-models.md)), AI upscale of
  refined pictures (2x, Real-ESRGAN compact model in `models/`; off by default), resolution (512 to 1024 px long
  side), how closely the picture follows the lines, the "surprise number" (seed), the
  safety filter and its strictness, and under *More settings* line detection, the
  starting image and picture quality. Changes apply live without a restart and are saved
  in `data/params.json`. If the engine cannot apply a change, the previous setting comes
  back and the page shows the error. With the safety filter off, the page shows a red
  warning. Each control has an **ⓘ** button. It opens `/static/help.html` in a new tab,
  showing that control with real example pictures rendered on your GPU. Re-render them
  after changing models or defaults with `.venv\Scripts\python.exe tools\make_help_images.py`.
  A memory meter shows the estimated graphics memory (from `static/vram.json`, measured
  with `tools/measure_vram.py`) against 10 GB; settings that would not fit are disabled.
- **Styles:** hide or show, reorder, rename or re-emoji any style. You can also add your
  own style with a short description, up to 120 characters, such as "rainbow colors,
  shiny stickers", and delete your own styles with one tap (and a confirm). The safety
  tail is always added after it.
- **Style from a picture:** upload a photo whose look you like. While the safety filter
  is on, each upload is screened by the same checker first; with the filter off, uploads
  are stored unscreened. See the known limits below.
- **Topics:** hide or show and reorder, or add your own topic with a short description of
  what the pictures are about (up to 120 characters, such as "a pirate ship on a sunny
  sea"). Your own topics can be deleted again.
- **Gallery:** see below.

Topics and styles are saved in `data/styles.json`.

## Drawing on the iPad

- **Paper:** every drawing has a background colour, black by default, kept as its own
  layer. **New** asks for a paper colour first (the very first drawing starts on black).
  **Paper** changes it at any time and
  keeps the strokes. The eraser only removes strokes. The paper colour also sets the mood
  of the picture: black paper gives night scenes.
- **Shape:** drawings have the iPad's shape (640×448), not a square, and take the
  orientation the iPad has when the drawing is started (portrait drawings are 448×640).
  Turning the iPad mid-drawing keeps the drawing upright and fits it to the screen.
  Switching between a landscape and a portrait drawing makes the PC rebuild its picture
  pipeline once, which takes about 8 seconds.
- **Views:** *Auto* (the sketch while drawing, then the picture), *Sketch*, *Picture* and
  *Both* (the drawing and the picture side by side).
- **Bigger drawing:** the minimise button shrinks the toolbar to just the colours and the
  brush size.
- **Dark or light:** the drawing page is dark by default; switch it under "Drawing page
  look" on the admin page (or `kid_theme` in `config.yaml`).
- **Live, then polished:** while drawing, the picture follows every stroke quickly; when
  your child stops, it refines for a moment, within about a second with the defaults (and
  gets upscaled, if that is on). A new stroke stops the refinement after the current step.
- **Connection dot:** the small dot on the Dreamer icon shows whether the PC is reachable
  and ready.

## Gallery

Your child can keep pictures they like:

- **Save** (the yellow star) keeps the picture on screen together with the drawing it
  came from. The picture flies into the **Gallery** button. Saving again after more
  drawing updates the same picture instead of making a copy.
- **Gallery** shows every saved picture. Tap one to see the picture and the drawing side
  by side, then **Keep drawing** to put that drawing back on the canvas and carry on.
- **New** starts an empty canvas. It asks first if the current drawing is not saved
  (**Clear** does the same).
- **Drafts:** the drawing in progress is saved automatically (at most every 10 seconds,
  and when the page closes). When the iPad page is reopened, the drawing comes back by
  itself, with a **Start fresh** button for a clean canvas. Clear and New drop the draft.

Children cannot delete pictures. On the admin page, the **Gallery** section lists the
saved pictures. Tap one to rename it, download the picture (`dreamer-<name>-picture.jpg`)
or the drawing (`dreamer-<name>-sketch.png`), or move it to the **Trash**. From the Trash you can put a picture back,
or tap **Empty the trash** to delete those pictures for good.

Storage: the list of pictures is kept in `data/dreamer.db` (SQLite). The images are kept
in `data/artworks/<year-month>/<id>/`: the picture (JPEG), the drawing's stroke layer
(PNG, lossless, so a continued drawing keeps clean lines) and a thumbnail. File names
carry a version number (`picture.<v>.jpg`, `sketch.<v>.png`, `thumb.<v>.jpg`) so an update
never mixes old and new files. To back up the gallery, copy `dreamer.db` and the
`artworks` folder while Dreamer is stopped.

## Tests

```bash
./setup.sh --mock                      # once: creates .venv with the server + test packages
.venv/bin/python -m pytest -q          # unit + API + WebSocket tests (mock backend)
.venv/bin/python -m ruff check server tools tests   # lint (settings in ruff.toml)
node tests/e2e/run.mjs                 # browser end-to-end test (needs the playwright npm package)
```

On Windows, run the unit tests with `.venv\Scripts\python.exe -m pytest -q`. All automated
tests use the mock backend, so they need no GPU.

If `playwright` is installed globally, run the e2e script as
`NODE_PATH=$(npm root -g) node tests/e2e/run.mjs`.

The e2e script starts `.venv`'s `python -m server` with a temporary config (mock
backend, a free port, a temporary data dir, PIN 4321). It draws on the kid page in an
iPad-sized Chromium window, checks that a picture arrives, and tests the view modes and
Clear. It logs into the admin page in a phone-sized window, adds a style, and checks that
the style appears on the kid page without a reload. It then saves a picture, opens it in
the gallery, uses *Keep drawing*, checks draft restore and *Start fresh* after a reload,
and renames, deletes and purges the picture from the admin page while the kid gallery is
open. Screenshots are saved to `tests/e2e/screenshots/`.

GPU checks and tools (Windows, real backend): `tools\bench_gpu.py` (speed),
`tools\measure_vram.py` (graphics memory table), `tools\refine_lab.py` and
`tools\controlnet_lab.py` (research sheets), `tools\make_showcase.py` (all screenshots).
See [showcase/README.md](showcase/README.md).

## Troubleshooting

- **`setup.ps1` says Python was not found, or reports the wrong version:** install
  Python 3.10 (64-bit) or point the script at it with `-Python "C:\path\to\python.exe"`.
  If the existing `.venv` uses another Python, rerun with `-Force`.
- **`run.ps1` says "run .\setup.ps1 first":** `.venv` is missing; run the setup script.
- **The iPad cannot connect:** use the PC's address that `run.ps1` prints, check that the
  Wi-Fi is on the *Private* profile, and see [LAN access and firewall](#lan-access-and-firewall)
  (a *Block* rule for `python.exe` is the usual cause).
- **The admin PIN pad is locked:** wait for the countdown; each further wrong try doubles
  the wait. If the PIN is forgotten, change `admin_pin` in `config.yaml` and restart.
- **No picture, or the connection dot on the logo shows a problem:** look at the server console.
  Mock mode (`backend: mock`) shows whether the problem is the network or the GPU stack.
- **GPU problems** (CUDA not available, StreamDiffusion missing, TensorRT engine errors,
  a step count the engine was not built for): see the troubleshooting list in
  [requirements-gpu.md](../requirements-gpu.md#troubleshooting).
- **A CUDA "illegal address" error in the console:** restart Dreamer; a CUDA context error
  cannot be recovered inside the process. It was seen once during a long settings probe
  and did not reproduce ([gpu-results.md](gpu-results.md#not-done--open)).
- **Save says "Draw something first!":** there is no picture for the current drawing yet
  (or the canvas is empty). Draw a line and wait for the picture.

## Known limits

- **Style pictures are phase 2.** Uploaded style pictures are screened and stored, but
  they don't change the output yet. Applying them needs SD 1.5 + IP-Adapter, while phase 1
  uses SD-Turbo. Until then, a picture style uses its name as the prompt text.
- **The GPU backend has run on one PC so far** (Windows 11, RTX 3080 10 GB). At the
  default 640×448 with ControlNet, the backend makes about 12-15 pictures/s with 2 steps
  (67-80 ms) and about 20 with 1 step; through the real server the round trip is about
  0.1-0.2 s.
  See [gpu-results.md](gpu-results.md) for numbers, settings and the fixes it needed.
- **Refinement is a trade-off.** SD-Turbo is a 1-4 step model: continuing its own path
  beyond about 6 steps repaints the scene instead of refining it. The default refinement
  model (anchored detail ladder) pins the broad layout to the first picture; it had the
  lowest drift and the most detail of the models measured, but passes the full "don't
  change, only add detail" rule on only some drawings ([research/refinement-models.md](research/refinement-models.md#results-2026-10-25-night-rtx-3080)).
- One child session at a time. A second iPad that opens the page takes over, and the
  first one shows a "Draw here" button.
- The kid page has been used on one real iPad at home (Safari and the home-screen app) and
  tested in desktop Chromium/Edge with touch emulation; *Keep drawing* and draft restore,
  which paint the saved PNG back onto the canvas, are covered by the automated browser test
  more than by real use so far.
- `setup.ps1` and `run.ps1` have run on Windows 11 (PowerShell 5.1); `setup.sh` has not
  run the GPU install on Linux/WSL yet.
- Gallery files are replaced atomically, with a short retry on Windows when a file is
  being read at that moment. The gallery tests pass on Windows, but that retry path (a
  file locked by a reader during the replace) has not been exercised there.
