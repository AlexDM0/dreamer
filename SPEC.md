# Dreamer: implementation spec (phase 1)

_Updated 2026-10-26 to match the code; docs/setup.md is the user guide._ This started as the phase 1 spec; later changes live in the `SPEC-addendum-*.md` files (params, gallery, canvas, setup) and are folded in below where they replaced a phase 1 decision.

**Dreamer** is a kids' drawing game (the product name; use it in page titles, README, logs). A child draws on an iPad in Safari; the sketch streams over a
WebSocket to a Windows PC with an RTX 3080, which runs StreamDiffusion and streams back a
generated picture. A parent manages styles from a phone via a PIN-protected admin page.
Everything runs on the home LAN from one Python process.

Design doc (background, not required reading): see README and docs/.

## Hard rules

- One Python process: FastAPI + uvicorn, host `0.0.0.0`, port from config (default 7860).
- Exactly ONE thread touches the diffusion backend / CUDA: the GPU worker thread.
- "Latest frame wins": never queue sketches. A single-slot mailbox is overwritten.
- The kid's page never sends prompt text. The server has no code path that accepts prompt
  text from the kid socket. Free-text styles come only from the authenticated admin API.
- One child session at a time: a new kid WebSocket replaces (closes) the previous one.
- Frontend: plain HTML/CSS/ES modules, no build step, no frameworks, no CDN. Must work in
  iOS Safari 15+ and desktop Chromium.
- Python 3.10+ (the PC runs 3.10; tests here run on 3.13). No GPU in CI/tests: the
  `mock` backend must make the whole app runnable and testable without torch.
- Don't import torch/streamdiffusion at module import time anywhere except inside the
  `streamdiffusion` backend's constructor, so tests and the mock backend never need them.

## Repository layout

```
dreamer/
  server/
    __init__.py
    __main__.py        # `python -m server` -> loads config, runs uvicorn
    app.py             # create_app(config, backend=None) -> FastAPI; routes, static, ws
    config.py          # load_config(path) -> Config dataclass; defaults; canvas_dims()
    session.py         # KidSession manager (single active kid socket), admin watchers
    mailbox.py         # Mailbox (single slot, thread-safe)
    worker.py          # GpuWorker thread (frames, refinement, uploads, picture-model switch)
    protocol.py        # binary header pack/unpack, message type constants
    styles.py          # StyleStore: topics + styles (presets + styles.json), prompt builder
    auth.py            # admin PIN check, signed session cookie, lockout
    imaging.py         # decode PNG, make init image + line map, encode JPEG
    params.py          # GenParams, limits, validation, ParamsStore (data/params.json)
    gallery.py         # GalleryStore: SQLite metadata + artwork files, drafts
    refine.py          # refinement models: what happens once the child stops drawing
    upscale.py         # optional AI upscaler (Real-ESRGAN compact, needs torch)
    offline.py         # sets HF_HUB_OFFLINE once the models are cached
    fsutil.py          # atomic-replace helper shared by the stores (Windows retry)
    backends/
      __init__.py      # make_backend(config, picture_model) -> Backend
      base.py          # Backend protocol (abstract)
      mock.py          # MockBackend: no GPU, deterministic stylised output
      streamdiffusion.py  # StreamDiffusionBackend (Daydream fork wrapper, SD-Turbo)
      common.py        # small torch-free helpers shared by the GPU backends
      tcd.py           # TcdBackend: SD 1.5 + TCD picture model (plain diffusers)
  static/
    index.html  style.css  theme.css  app.js  toolbar.js  canvas.js  paper.js  net.js
    view.js  gallery.js  gate.js          # kid page (gate.js = parent PIN gate)
    admin.html  admin.css  admin.js  admin.webmanifest   # parent page
    help.html + help/    # parent help page with example pictures
    vram.json            # measured GPU memory/speed table (tools/measure_vram.py)
    logo.svg  favicon.svg  icon-*.png  admin-icon-*  dreamer.ico  manifest.webmanifest
  tests/
    test_*.py          # pytest, uses mock backend
    e2e/run.mjs        # playwright (node) end-to-end test against mock backend
  tools/               # bench_gpu, controlnet_lab, refine_lab, measure_vram, make_* scripts
  docs/                # setup.md (user guide), showcase, research, reviews
  config.example.yaml
  requirements.txt      # runtime + test deps for the server WITHOUT torch
  requirements-gpu.md   # manual reference for the GPU install steps setup.ps1 runs
  setup.ps1  setup.sh   # create .venv and install everything (see SPEC-addendum-setup.md)
  run.ps1  run.sh       # launchers (.venv only)
  Dreamer.cmd           # double-click launcher: runs run.ps1 and pauses on error
  README.md
```

## Config (`config.yaml`, all keys optional; `config.example.yaml` documents them)

```yaml
host: 0.0.0.0
port: 7860
backend: streamdiffusion      # or "mock"
admin_pin: "1234"             # REQUIRED to change; 4-12 digits, quoted; server logs a warning if default
secret_key: ""                # cookie signing; auto-generated and persisted to data/secret if empty
data_dir: data                # styles.json, params.json, secret, style pictures, gallery (dreamer.db + artworks/)
jpeg_quality: 85              # 1..100
size: 640                     # long side in px, multiple of 64 (64..2048); short side is 0.7x (canvas_dims)
default_view_mode: auto       # auto | sketch | picture | split
kid_theme: dark               # dark | light; the drawing page's look, also switchable on /admin
mock_delay_ms: 30             # mock backend only: simulated GPU time per frame
model:
  base: stabilityai/sd-turbo
  t_index_list: [6, 26]
  max_steps: 8                # most denoising steps the admin page may pick
  max_batch_steps: 4          # 1, 2, 4 or 8; steps run together as one UNet batch (sets engine memory; change rebuilds once)
  refine_steps: 8             # after drawing pauses, refine up to this many steps in total (0 = off, live max 20)
  upscaler: "off"             # "off" | "x2"; model file models/realesr-animevideov3.pth (upscaler_model)
  controlnet: thibaud/controlnet-sd21-scribble-diffusers   # null disables ControlNet
  controlnet_scale: 0.55
  line_polarity: white_on_black   # or black_on_white; what the ControlNet expects
  engine_dir: engines
  safety_checker: true
  safety_threshold: 0.5
  # "SD 1.5 + TCD" picture model (selectable live on /admin):
  tcd_base: stable-diffusion-v1-5/stable-diffusion-v1-5
  tcd_lora: h1t/TCD-SD15-LoRA            # tcd_lora_file defaults to pytorch_lora_weights.safetensors
  tcd_controlnet: lllyasviel/control_v11p_sd15_scribble
  tcd_strength: 1.0
  tcd_gamma: 0.3
```

Unknown keys are ignored with a warning; invalid values raise ValueError at startup.
Many of these (t_index_list, controlnet_scale, safety_*, line_polarity, jpeg_quality,
refine_steps, upscaler, size) are only the *defaults* of the live params, see
SPEC-addendum-params.md.

`Config` is a dataclass (nested `ModelConfig`). Env var `SKETCH_CONFIG` points at the
config file; default `config.yaml` in CWD; missing file -> defaults. `canvas_dims(cfg, long_side)`
gives the landscape/portrait (width, height) of drawings and pictures (see SPEC-addendum-canvas.md).

## Backend interface (`server/backends/base.py`)

```python
class Backend(Protocol):
    def warmup(self) -> None: ...                          # called once on worker thread
    def set_prompt(self, prompt: str) -> None: ...         # worker thread only
    def set_params(self, params: GenParamsLike) -> None: ...  # live params; raise on failure, keep old ones
    def generate(self, init_image: PIL.Image.Image, line_map: PIL.Image.Image | None) -> PIL.Image.Image: ...
    def check_image_safe(self, image: PIL.Image.Image) -> bool: ...  # used for admin uploads
    @property
    def name(self) -> str: ...
```

Optional members, looked up with `getattr` by the worker (the mock has none of them):

- `latency_frames: int` (default 0): how many `generate()` calls later a sketch's picture
  comes out (denoising batch, steps - 1); the worker primes a fresh pipeline and re-runs the
  last sketch once drawing pauses.
- `set_size(width, height)`: rebuild at another drawing size/orientation (see the canvas addendum).
- `close()`: free GPU memory; called before another picture model is loaded.
- `refine_steps(init, line_map, t_index, first_shown, noise)` and
  `latent_ladder(first, line_map, t_index, ...)` (StreamDiffusionBackend): step-by-step
  refinement generators used by `server/refine.py`; `refine_own(...)` (TcdBackend): refine by
  re-rendering with more steps from the same seed.

Phase 1 had only the first five members; the rest came with the params, canvas and refinement work.

All methods are called ONLY from the GPU worker thread (the admin-upload safety check is
routed through the worker as a job, see Worker). `make_backend(config, picture_model)` returns
the mock, `StreamDiffusionBackend` (SD-Turbo) or, for picture model `tcd`, `TcdBackend`.

- `MockBackend`: no torch. `generate` returns a deterministic stylised version of the
  init image (e.g. posterize + blur + a tint derived from a hash of the prompt) at the same
  size, optionally sleeping `mock_delay_ms` (constructor arg, default 30) to simulate GPU
  time. `check_image_safe` returns True unless the image is pure magenta (255,0,255) —
  a test hook. `warmup` sleeps briefly.
- `StreamDiffusionBackend`: wraps `streamdiffusion.StreamDiffusionWrapper` from the
  Daydream fork (github.com/daydreamlive/StreamDiffusion). Use ONLY these verified
  constructor args: `model_id_or_path, t_index_list, mode="img2img", width, height,
  frame_buffer_size=1, use_denoising_batch=True, cfg_type="none", use_tiny_vae=True,
  acceleration="tensorrt", engine_dir, use_controlnet, controlnet_config=[{"model_id",
  "preprocessor": "passthrough", "conditioning_scale"}], use_safety_checker,
  safety_checker_fallback_type="previous", safety_checker_threshold, output_type="pil"`.
  Then `prepare(prompt=..., num_inference_steps=50)`. Per frame:
  `update_control_image(0, line_map)` (if ControlNet on) then `wrapper(image=init_image)`.
  Prompt change: `wrapper(image=..., prompt=new_prompt)` handles `update_prompt`
  internally; simplest is to store the pending prompt and pass it on the next generate.
  `check_image_safe` for uploads: if the wrapper exposes a loaded `safety_checker`, use it
  with care; otherwise return True and log that upload screening is unavailable. Mark any
  uncertain API use with a `# VERIFY:` comment — this backend cannot be tested here.
  Since phase 1 it has been run on the target PC (see docs/gpu-results.md): it also
  implements `set_params`, `set_size`, `close`, `latency_frames`, `refine_steps` and
  `latent_ladder`, and uses the pinned fork commit installed by setup.ps1.
- `TcdBackend` (`backends/tcd.py`): the "SD 1.5 + TCD" picture model, plain diffusers (no
  TensorRT), selected live with the `picture_model` param; the worker unloads the SD-Turbo
  backend first. Implements `refine_own` instead of the StreamDiffusion refinement members.

## Imaging (`server/imaging.py`)

- `decode_sketch(png_bytes, size, layered=False) -> PIL.Image`: `size` is an int or
  `(width, height)` (phase 1 was square). PNG only; reject oversized payloads and non-image
  data (raise ValueError). Unlayered sketches are composited onto white (RGB); `layered`
  keeps the RGBA stroke layer (see the canvas addendum).
- `make_line_map(sketch, polarity, ...) -> PIL.Image (RGB)`: pixels that count as ink
  (distance from white, or alpha for a stroke layer, above `line_threshold`, default 40) are
  lines; white lines on black for `white_on_black` (default) or the inverse, with
  `line_dilate` px of dilation (default 1).
- `make_init_image(...)`: the img2img start image per `init_mode` (colors | sketch | blank),
  see the params addendum.
- `encode_jpeg(img, quality) -> bytes`.
- `is_blank(sketch) -> bool`: nothing drawn. For a blank sketch the worker skips the
  backend and answers with a picture in the background colour (cheap clear).

## Styles (`server/styles.py`)

Two lists: topics and styles. Each item:
`{"id": str, "label": str, "emoji": str, "prompt": str, "kind": "preset"|"custom"|"image", "hidden": bool, "image": str|None}`.

- Built-in preset topics (at least): ocean, nature, space, dinosaurs, castle, city, jungle,
  winter. Preset styles (at least): cartoon, crayon, watercolor, clay, storybook,
  pixel art. Prompt fragments are short and kid-friendly.
- Custom styles and topics (admin-typed text) and image styles (store the image, kind
  "image"; not applied to pictures yet, see the style-image endpoint) persist in
  `data/styles.json`. Presets can be hidden/reordered via the same
  file (store order + hidden flags). Writes are atomic (temp file + replace).
- `build_prompt(topic_id, style_id) -> str` =
  `"a colorful illustration, {topic}, {style}, for children, friendly, bright colors, wholesome"`
  (head `a colorful illustration`, tail `for children, friendly, bright colors, wholesome`).
  Unknown ids fall back to defaults (first visible). The safety tail is always last.
- Admin custom style text: strip, collapse whitespace, prompt max 120 chars (label 40), reject control
  characters. (Parent-only, so no blocklist required, but keep the safety tail.)
- `ids` for custom items: slug of label + short suffix; unique.
- StyleStore is thread-safe (lock) and exposes `public_list()` (visible items, no file paths)
  for the kid page and `admin_list()` (everything).

## Mailbox (`server/mailbox.py`)

Exactly as below (single slot, overwrite, event):

```python
class Mailbox:
    def put(self, item) -> None
    def take(self, timeout: float | None = None)  # returns item or None on timeout
    def clear(self) -> None
```

## Worker (`server/worker.py`)

`GpuWorker(backend, mailbox, on_result, on_status, config)` — a `threading.Thread`
(daemon). Loop: warmup (status "busy" -> "ready"), then `take(timeout=0.5)`; on a job:

- Job kinds (dataclasses): `FrameJob(seq, png_bytes, prompt)` and
  `ScreenUploadJob(image, future)` for admin upload safety checks. Uploads must not be
  dropped by "latest wins": keep a separate small `queue.Queue` for upload jobs that the
  worker drains before taking a frame.
- For FrameJob: if prompt changed since last frame -> backend.set_prompt. Decode, blank
  check, line map, generate, encode JPEG, call `on_result(seq, jpeg_bytes)`.
- `on_result`/`on_status` are thread-safe callables supplied by app.py that use
  `loop.call_soon_threadsafe`. The worker never touches sockets.
- Errors: log, `on_status("error", msg)`, continue. After 3 consecutive failures, call
  backend factory again to rebuild (factory passed in, optional).
- Beyond phase 1: live params are applied before the next frame (SPEC-addendum-params.md);
  once drawing pauses it runs the refinement model (`server/refine.py`, optional upscale via
  `server/upscale.py`) and sends more pictures with the same seq; it can switch the picture
  model (`turbo` <-> `tcd`) and the drawing size/orientation by rebuilding the backend; after
  the first successful warmup `offline.mark_ready` lets later starts run without internet.
  `FrameJob` also carries the canvas background (see the canvas addendum).
- `stop()` for clean shutdown (FastAPI lifespan).
- Track simple stats: last frame ms, fps (EMA). Expose via `stats()`.

## Protocol (`server/protocol.py`) — kid WebSocket `GET /ws`

Binary messages: 5-byte header then payload.

| byte 0 | type |
|---|---|
| 0x01 | SKETCH (client->server, PNG) |
| 0x02 | PICTURE (server->client, JPEG) |

bytes 1..4: `seq`, uint32 big-endian. PICTURE echoes the seq of the sketch it was made from.

JSON text messages, field `t`:

client -> server
- `{"t":"hello","v":1}` (later also `"continuing": <artwork id>|null`)
- `{"t":"topic","id":"ocean"}`
- `{"t":"style","id":"crayon"}`
- `{"t":"clear"}` — server clears the mailbox; client follows with a blank SKETCH.
- Added later (see the addenda): `canvas`, `save`, `new`, `continue`, `continuing`, `restored`.
- `{"t":"ping"}` -> server replies `{"t":"pong"}`

server -> client
- `{"t":"ready","v":1,"topics":[...],"styles":[...],"topic":"ocean","style":"cartoon","view_mode":"auto","backend":"mock"}` (reply to hello; also re-sent as `{"t":"styles",...}` same shape minus v when admin changes lists or defaults)
- `{"t":"status","state":"busy"|"ready"|"error","msg":"..."}`
- `{"t":"replaced"}` sent to an old socket right before it is closed because a new kid socket connected.
- `{"t":"pong"}`
- Added later: `ready` also carries `kid_theme`, `canvas` and `draft`; `saved`, `continue` and
  `gallery` messages (gallery addendum). The selection fields come from `session.current()`.

Server ignores unknown `t` values and any field named `prompt`/`text` from the kid
socket. Malformed binary (short header, wrong type) is ignored with a log line.
Max message size 2 MB. The current topic/style selection lives server-side in the session
(survives reconnects).

Admin watchers: `GET /ws/admin` (requires admin cookie) receives the same PICTURE binary
frames and `status` messages, plus `{"t":"stats",...}` once a second (also `params` and
`gallery` change notices). Read-only.

## Admin HTTP API (cookie auth, JSON)

- `POST /api/admin/login` `{"pin":"1234"}` -> sets HttpOnly signed cookie `sg_admin`
  (HMAC-SHA256 of an expiry timestamp, key derived from secret_key and the PIN, so a new
  PIN logs everyone out; 30 days, or 15 minutes when the body has `"short": true`). 401 on
  wrong PIN. Rate limit: after 5 wrong attempts a 30 s lockout, doubling per further wrong
  PIN up to 15 minutes (429 with Retry-After), plus a global budget of 20 failures (in-memory).
- `POST /api/admin/logout`
- `GET /api/admin/state` -> `{"topics":[...all],"styles":[...all],"topic","style","view_mode","kid_theme","backend","stats":{...}}`
- `POST /api/admin/styles` `{"label","emoji","prompt"}` -> creates custom style; `POST /api/admin/topics` the same for a custom topic.
- `PATCH /api/admin/items/{kind}/{id}` `{"hidden"?:bool, "label"?, "emoji"?, "prompt"?}` (kind = topics|styles; prompt editable only for custom)
- `DELETE /api/admin/items/{kind}/{id}` (custom/image items only)
- `POST /api/admin/order` `{"kind":"topics"|"styles","ids":[...]}`
- `POST /api/admin/select` `{"topic"?:id,"style"?:id,"view_mode"?:mode,"kid_theme"?:"dark"|"light"}` -> sets current
  selection / default view mode and pushes to the kid page.
- `POST /api/admin/style-image` multipart `file`, `label`, `emoji` -> screens via worker
  (`ScreenUploadJob`), stores in `data/style_images/` (PNG, long side capped at 1024), creates kind "image" style. Respond
  `{"ok":true,"style":{...},"note":"Style pictures need phase 2 (SD 1.5 + IP-Adapter); stored but not applied yet."}` when the backend doesn't support image styles. Max 8 MB, image types only.
- Later endpoints, documented in the addenda: `GET|PATCH /api/admin/params`, `POST /api/admin/params/reset`
  (params; the GET also returns `limits`, `defaults` and `gpu` memory info), the artwork
  endpoints under `/api/admin/artworks` and the public `/api/artworks` (gallery).
- Every successful mutation broadcasts `{"t":"styles",...}` to the kid socket.
- `GET /healthz` -> `{"ok":true,"backend":..,"state":..}` (no auth).

Static: `/` -> `static/index.html`, `/admin` -> `static/admin.html`, `/static/*`, `/favicon.ico` -> `favicon.svg`.
Disable caching on HTML (Cache-Control: no-cache) so the iPad picks up updates.

## Kid page (`static/index.html` + modules)

Landscape-first iPad layout, also usable in portrait. Big touch targets (>= 48px).

- Toolbar along the top: colour swatches (8–10 big round swatches incl. black, white as
  "eraser") + native `<input type="color">` as "more colours"; brush width slider
  `<input type="range" min=4 max=40>` with a live dot preview; Clear button (big, with an
  icon); view-mode toggle (Auto / Sketch / Picture / Split icons); topic and style buttons
  (emoji + label, horizontally scrollable rows, current one highlighted). Connection dot
  (green/yellow/red).
- `canvas.js`: phase 1 had a 512x512 white canvas; now the drawing is a transparent stroke
  layer over a background colour, with the iPad's shape (SPEC-addendum-canvas.md), scaled
  to fit via CSS; Pointer Events (`pointerdown/move/up/cancel`), `setPointerCapture`,
  `getCoalescedEvents()` when available; round caps/joins; draw line segments between
  points; palm rejection: if a `pen` pointer has been seen, ignore `touch` pointers while
  a pen is down. Exposes `isDirty()`, `clearDirty()`, `toBlob()`, `clear()`, events
  `strokestart`, `strokeend`.
- `net.js`: WebSocket to `ws(s)://<host>/ws`; `binaryType="arraybuffer"`; hello on open;
  reconnect with backoff 0.5s doubling to 5s; one-frame-in-flight: send a SKETCH only when
  canvas dirty and no frame in flight; in-flight is cleared on PICTURE or after a 2 s
  timeout; `sendNow()` used on strokeend (bypasses dirty check but still respects
  in-flight by queueing exactly one "pending final" send); tracks `finalSeq`. Pings every
  10 s.
- `view.js`: two stacked canvases (sketch, picture) in one container; picture drawn via
  `createImageBitmap(new Blob([jpeg],{type:"image/jpeg"}))` with fallback to Image +
  object URL; modes via container class `mode-auto|mode-sketch|mode-picture|mode-split`.
  Auto: show sketch on strokestart instantly; after strokeend, keep sketch until a
  PICTURE with `seq >= finalSeq` arrives, then cross-fade (~150 ms CSS opacity). Split:
  side-by-side in landscape, stacked in portrait (CSS media query). Picture only: picture
  visible; the drawing canvas still receives pointer events (opacity 0, on top). Mode
  persisted in localStorage (try/catch); server `view_mode` is the default when nothing is
  stored. Keep last picture visible during reconnects.
- Safari hardening: `<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">`, `touch-action: none` on the
  drawing surface, `-webkit-user-select:none; -webkit-touch-callout:none;` on body,
  prevent `gesturestart` and double-tap zoom, `apple-mobile-web-app-capable` meta + a
  simple inline SVG/PNG icon for "Add to Home Screen".
- No prompt text box anywhere on the kid page.
- Added since phase 1: `gallery.js` (Save / Gallery / continue, see the gallery addendum),
  `gate.js` (tapping the Dreamer icon asks for the admin PIN, then opens `/admin`),
  `paper.js` (background colour picker) and `theme.css` (dark/light look, `kid_theme`).

## Admin page (`static/admin.html` + admin.js)

Phone-first layout. If not logged in: big PIN pad (digits) -> login. Then sections:

1. Now: live picture (from `/ws/admin`), backend + state + fps, current topic/style
   pickers, default view mode select, kid theme select.
   Then "Picture controls" (live diffusion params, SPEC-addendum-params.md).
2. Styles: list of all styles (presets + custom + image) with hide/show toggle, up/down
   reorder, edit label/emoji (and prompt for custom), delete (custom/image).
   "Add a style" form: label, emoji, prompt text.
3. Style picture: file input (camera/library), label, emoji, upload; show the server's
   note that style pictures are stored but not applied to pictures yet.
4. Topics: hide/show + reorder.
5. Gallery (saved artworks, trash; gallery addendum).
6. Log out.
Help for parents is in `static/help.html`.

## Testing

- pytest (`tests/`), using `create_app(config, backend=MockBackend(mock_delay_ms=1))` and
  FastAPI `TestClient` (incl. websocket_connect). Cover: protocol pack/unpack, mailbox
  overwrite semantics, imaging (line map polarity, blank detection, oversized payload
  rejection), styles (presets, custom CRUD, persistence, prompt building, safety tail last,
  kid can't inject text), auth (login, wrong pin, lockout, protected endpoints 401),
  kid ws end-to-end (hello->ready, sketch->picture with echoed seq, latest-wins under burst,
  replaced old socket), admin select pushes `styles` to kid socket, upload screening reject
  (magenta) and accept.
- Playwright e2e in `tests/e2e/` (node, `playwright` is preinstalled globally: use
  `require('playwright')`, chromium.launch() needs no options; do NOT run playwright
  install): start server with mock backend on a free port, open the kid page with an iPad
  viewport + hasTouch, draw strokes with mouse/touch, assert a picture frame is rendered,
  view-mode switching works, clear works; open admin, log in with PIN, add a custom style,
  assert it appears on the kid page without reload.
- `python -m server` with `backend: mock` must start and serve.
- Tests now also cover config, params, gallery (+API), canvas, refine, offline and the worker;
  run them with `.venv/bin/python -m pytest -q` (Windows: `.venv\Scripts\python.exe -m pytest -q`).
  The e2e is the single script `tests/e2e/run.mjs` (`node tests/e2e/run.mjs`).

## Branding

- Name: **Dreamer**. Tone: playful, warm, simple; for young kids.
- Logo: `static/logo.svg` — a hand-drawn-looking (crayon/scribble feel) cloud with a small
  sparkle/star, plus the wordmark "Dreamer" in a rounded friendly font stack (no web font
  downloads; use system rounded fonts like "ui-rounded", "SF Pro Rounded", "Nunito",
  "Comic Sans MS" fallback chain) — or wordmark drawn as SVG paths. Works on light and
  dark backgrounds. Also `static/icon-180.png` (apple-touch-icon, 180x180, logo mark on a
  soft gradient background, rendered from SVG with Playwright/Chromium screenshot or
  Pillow) and `static/favicon.svg` (mark only).
- Use the logo in the kid page toolbar (small mark), the admin page header, and README.
- Colour palette: soft sky blue, sunny yellow, coral pink, mint; define as CSS variables
  shared by both pages (`theme.css`; docs/design-system.md describes the current look).
