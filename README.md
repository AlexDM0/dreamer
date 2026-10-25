<p align="center"><img src="static/logo.svg" alt="Dreamer" width="320"></p>

# Dreamer

Dreamer is a drawing game for young kids. A child draws on an iPad in Safari, and the
drawing streams over the home Wi-Fi to a Windows PC with an NVIDIA GPU. The PC runs
[StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion) (SD-Turbo + a scribble
ControlNet) and sends a picture back while the child draws. A parent picks topics and
styles from a phone on a PIN-protected admin page. Children never type prompts: every
prompt is built on the server from parent-approved topics and styles, with a fixed
kid-friendly safety tail.

Everything runs on your LAN from one Python process. Nothing goes to the cloud.

<!-- Screenshots: add docs/kid.png (iPad) and docs/admin.png (phone) here.
     `node tests/e2e/run.mjs` saves fresh ones to tests/e2e/screenshots/. -->

## Quick start (Windows PC with an RTX GPU)

1. Install [Python 3.10](https://www.python.org/downloads/windows/) (64-bit, keep the *py
   launcher* option), [Git for Windows](https://git-scm.com/download/win) and a current
   NVIDIA driver (CUDA 12.8 or newer).
2. In PowerShell, from the Dreamer folder, run the setup script once:

   ```powershell
   .\setup.ps1
   ```

   It creates `.venv` and installs everything into it (server packages, CUDA PyTorch, the
   StreamDiffusion fork and TensorRT), checks that PyTorch sees the GPU, and creates
   `config.yaml`. Nothing is installed into your system Python. Options: `-Mock` (no GPU
   stack), `-Force` (recreate `.venv`), `-Python <path or "py -3.10">`, `-Cuda cu128`.
   [requirements-gpu.md](requirements-gpu.md) lists the same steps for a manual install.

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

   The first run downloads the models and builds TensorRT engines into `engines/`,
   which takes a while. Later starts are quick.
5. On the iPad, open `http://<pc-ip>:7860/` in Safari and choose *Add to Home Screen*.
   The parent page is `http://<pc-ip>:7860/admin`. `run.ps1` prints both URLs at startup.
   Allow the port through the firewall once (see [LAN access and firewall](#lan-access-and-firewall)).

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
[`config.example.yaml`](config.example.yaml). The server reads `config.yaml` from the
current directory, or the file named by the `SKETCH_CONFIG` environment variable. An
unknown key logs a warning. An invalid value stops the server with exit code 2.

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
`config.yaml`). Five wrong tries lock the PIN pad for 30 seconds; each further wrong try
doubles the wait (up to 15 minutes) until the right PIN is entered. The login lasts 30
days in that browser; changing the PIN logs every device out. On the admin page you can:

- **Now:** watch the live picture, the engine state and speed, and change the current
  topic, style and the iPad's starting view. Changes show on the iPad immediately.
- **Picture controls:** creativity, steps, how closely the picture follows the lines,
  the "surprise number" (seed), the safety filter, line detection and picture quality.
  Changes apply live without a restart and are saved in `data/params.json`. If the
  engine cannot apply a change (for example a step count the TensorRT engine was not
  built for), the previous setting comes back and the page shows the error.
- **Styles:** hide or show, reorder, rename or re-emoji any style. You can also add your
  own style with a short description, up to 120 characters, such as "rainbow colors,
  shiny stickers". The safety tail is always added after it.
- **Style from a picture:** upload a photo whose look you like. Each upload is screened
  first. See the known limits below.
- **Topics:** hide or show and reorder.

Settings are saved in `data/styles.json`.

## Gallery

Your child can keep pictures they like:

- **Save** (the yellow star) keeps the picture on screen together with the drawing it
  came from. The picture flies into the **Gallery** button. Saving again after more
  drawing updates the same picture instead of making a copy.
- **Gallery** shows every saved picture. Tap one to see the picture and the drawing side
  by side, then **Keep drawing** to put that drawing back on the canvas and carry on.
- **New** starts an empty canvas. It asks first if the current drawing is not saved.
- **Drafts:** the drawing in progress is saved automatically (at most every 10 seconds,
  and when the page closes). When the iPad page is reopened, the drawing comes back by
  itself, with a **Start fresh** button for a clean canvas. Clear and New drop the draft.

Children cannot delete pictures. On the admin page, the **Gallery** section lists the
saved pictures. Tap one to rename it, download the picture (`picture.jpg`) or the drawing
(`sketch.png`), or move it to the **Trash**. From the Trash you can put a picture back,
or tap **Empty the trash** to delete those pictures for good.

Storage: the list of pictures is kept in `data/dreamer.db` (SQLite). The images are kept
in `data/artworks/<year-month>/<id>/`: `picture.jpg`, `sketch.png` (lossless, so a
continued drawing keeps clean lines) and `thumb.jpg`. To back up the gallery, copy
`dreamer.db` and the `artworks` folder while Dreamer is stopped.

## Tests

```bash
./setup.sh --mock                      # once: creates .venv with the server + test packages
.venv/bin/python -m pytest -q          # unit + API + WebSocket tests (mock backend)
node tests/e2e/run.mjs                 # browser end-to-end test (needs the playwright npm package)
```

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
  [requirements-gpu.md](requirements-gpu.md#troubleshooting).
- **Save says "Draw something first!":** there is no picture for the current drawing yet
  (or the canvas is empty). Draw a line and wait for the picture.

## Known limits

- **Style pictures are phase 2.** Uploaded style pictures are screened and stored, but
  they don't change the output yet. Applying them needs SD 1.5 + IP-Adapter, while phase 1
  uses SD-Turbo. Until then, a picture style uses its name as the prompt text.
- **The GPU backend is untested.** The StreamDiffusion backend was written against the
  fork's documented API but has not run on a real GPU; that code is marked `# VERIFY:`.
  All automated tests use the mock backend.
- One child session at a time. A second iPad that opens the page takes over, and the
  first one shows a "Draw here" button.
- The kid page was tested in desktop Chromium with touch emulation, not yet on a real
  iPad. That includes *Keep drawing* and draft restore, which paint the saved PNG back
  onto the canvas.
- **The setup scripts have not run on real Windows.** `setup.ps1` and `run.ps1` were
  parse-checked for PowerShell 5.1 and run under PowerShell 7 on Linux; the GPU install
  steps (CUDA PyTorch, StreamDiffusion, TensorRT) have not run anywhere yet.
- Gallery files are replaced atomically, with a short retry on Windows when a file is
  being read at that moment. This retry has not been tested on Windows.
