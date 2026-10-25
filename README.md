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

1. Install the GPU stack into `.venv` by following **[requirements-gpu.md](requirements-gpu.md)**
   (Python 3.10, CUDA PyTorch, the StreamDiffusion fork, TensorRT).
2. Copy `config.example.yaml` to `config.yaml` and set `admin_pin`.
3. Start it from PowerShell:

   ```powershell
   .\run.ps1
   ```

   If PowerShell says *running scripts is disabled*, allow local scripts once with
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. If you downloaded the repo as a
   ZIP, Windows also marks the files as "from the internet"; unblock them from the repo
   folder with `Get-ChildItem -Recurse | Unblock-File`. Alternatively, run
   `powershell -ExecutionPolicy Bypass -File .\run.ps1`.

   The first run downloads the models and builds TensorRT engines into `engines/`,
   which takes a while. Later starts are quick.
4. On the iPad, open `http://<pc-ip>:7860/` in Safari and choose *Add to Home Screen*.
   The parent page is `http://<pc-ip>:7860/admin`. The server prints both URLs at startup.

## Mock mode (development, no GPU)

The `mock` backend needs no torch or GPU. It turns each sketch into a tinted, posterized
picture, so you can work on the whole app on any machine:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip ...
printf 'backend: mock\nadmin_pin: "4321"\n' > config.yaml
.venv/bin/python -m server                     # http://localhost:7860/
```

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

## Tests

```bash
.venv/bin/python -m pytest -q          # unit + API + WebSocket tests (mock backend)
node tests/e2e/run.mjs                 # browser end-to-end test (needs the playwright npm package)
```

The e2e script starts `python -m server` with a temporary config (mock backend, a free
port, a temporary data dir, PIN 4321). It draws on the kid page in an iPad-sized
Chromium window, checks that a picture arrives, and tests the view modes and Clear. Then
it logs into the admin page in a phone-sized window, adds a style, and checks that the
style appears on the kid page without a reload. Screenshots are saved to
`tests/e2e/screenshots/`.

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
  iPad.
