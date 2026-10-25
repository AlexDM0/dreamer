# Dreamer: implementation spec (phase 1)

**Dreamer** is a kids' drawing game (the product name; use it in page titles, README, logs). A child draws on an iPad in Safari; the sketch streams over a
WebSocket to a Windows PC with an RTX 3080, which runs StreamDiffusion and streams back a
generated picture. A parent manages styles from a phone via a PIN-protected admin page.
Everything runs on the home LAN from one Python process.

Design doc (background, not required reading): see README.

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
    config.py          # load_config(path) -> Config dataclass; defaults
    session.py         # KidSession manager (single active kid socket), admin watchers
    mailbox.py         # Mailbox (single slot, thread-safe)
    worker.py          # GpuWorker thread
    protocol.py        # binary header pack/unpack, message type constants
    styles.py          # StyleStore: topics + styles (presets + styles.json), prompt builder
    auth.py            # admin PIN check, signed session cookie
    imaging.py         # decode PNG, make init image + line map, encode JPEG
    backends/
      __init__.py      # make_backend(config) -> Backend
      base.py          # Backend protocol (abstract)
      mock.py          # MockBackend: no GPU, deterministic stylised output
      streamdiffusion.py  # StreamDiffusionBackend (Daydream fork wrapper)
  static/
    index.html  style.css  app.js  toolbar.js  canvas.js  net.js  view.js
    admin.html  admin.css  admin.js
  tests/
    test_*.py          # pytest, uses mock backend
    e2e/               # playwright (node) end-to-end tests against mock backend
  config.example.yaml
  requirements.txt      # runtime deps for the server WITHOUT torch (fastapi etc.)
  requirements-gpu.md   # (or section in README) exact Windows GPU install steps
  run.ps1               # Windows launcher
  README.md
```

## Config (`config.yaml`, all keys optional; `config.example.yaml` documents them)

```yaml
host: 0.0.0.0
port: 7860
backend: streamdiffusion      # or "mock"
admin_pin: "1234"             # REQUIRED to change; server logs a warning if default
secret_key: ""                # cookie signing; auto-generated and persisted to data/secret if empty
data_dir: data                # styles.json, uploaded style images, secret
jpeg_quality: 85
size: 512
default_view_mode: auto       # auto | sketch | picture | split
model:
  base: stabilityai/sd-turbo
  t_index_list: [6, 26]
  controlnet: thibaud/controlnet-sd21-scribble-diffusers   # null disables ControlNet
  controlnet_scale: 0.55
  line_polarity: white_on_black   # or black_on_white; what the ControlNet expects
  engine_dir: engines
  safety_checker: true
  safety_threshold: 0.5
```

`Config` is a dataclass (nested `ModelConfig`). Env var `SKETCH_CONFIG` points at the
config file; default `config.yaml` in CWD; missing file -> defaults.

## Backend interface (`server/backends/base.py`)

```python
class Backend(Protocol):
    def warmup(self) -> None: ...                          # called once on worker thread
    def set_prompt(self, prompt: str) -> None: ...         # worker thread only
    def generate(self, init_image: PIL.Image.Image, line_map: PIL.Image.Image | None) -> PIL.Image.Image: ...
    def check_image_safe(self, image: PIL.Image.Image) -> bool: ...  # used for admin uploads
    @property
    def name(self) -> str: ...
```

All methods are called ONLY from the GPU worker thread (the admin-upload safety check is
routed through the worker as a job, see Worker).

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

## Imaging (`server/imaging.py`)

- `decode_sketch(png_bytes, size) -> PIL.Image (RGB, size x size)`: composite alpha onto
  white, resize if needed. Reject >2 MB payloads and non-image data (raise ValueError).
- `make_line_map(sketch, polarity) -> PIL.Image (RGB)`: any pixel noticeably non-white
  (max channel distance from white > ~40) is a line; output white lines on black for
  `white_on_black` (default) or the inverse. Slight dilation (1px) is fine.
- `encode_jpeg(img, quality) -> bytes`.
- `is_blank(sketch) -> bool`: all (near) white. For a blank sketch the worker skips the
  backend and returns a white image (cheap clear).

## Styles (`server/styles.py`)

Two lists: topics and styles. Each item:
`{"id": str, "label": str, "emoji": str, "prompt": str, "kind": "preset"|"custom"|"image", "hidden": bool, "image": str|None}`.

- Built-in preset topics (at least): ocean, nature, space, dinosaurs, castle, city, jungle,
  winter. Preset styles (at least): cartoon, crayon, watercolor, clay, storybook,
  pixel art. Prompt fragments are short and kid-friendly.
- Custom styles (admin-typed text) and image styles (phase 2; store the image, kind
  "image") persist in `data/styles.json`. Presets can be hidden/reordered via the same
  file (store order + hidden flags). Writes are atomic (temp file + replace).
- `build_prompt(topic_id, style_id) -> str` =
  `"a colorful illustration, {topic}, {style}, for children, friendly, bright colors, wholesome"`.
  Unknown ids fall back to defaults (first visible). The safety tail is always last.
- Admin custom style text: strip, collapse whitespace, max 120 chars, reject control
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
- `{"t":"hello","v":1}`
- `{"t":"topic","id":"ocean"}`
- `{"t":"style","id":"crayon"}`
- `{"t":"clear"}` — server clears the mailbox; client follows with a blank SKETCH.
- `{"t":"ping"}` -> server replies `{"t":"pong"}`

server -> client
- `{"t":"ready","v":1,"topics":[...],"styles":[...],"topic":"ocean","style":"cartoon","view_mode":"auto","backend":"mock"}` (reply to hello; also re-sent as `{"t":"styles",...}` same shape minus v when admin changes lists or defaults)
- `{"t":"status","state":"busy"|"ready"|"error","msg":"..."}`
- `{"t":"replaced"}` sent to an old socket right before it is closed because a new kid socket connected.
- `{"t":"pong"}`

Server ignores unknown `t` values and any field named `prompt`/`text` from the kid
socket. Malformed binary (short header, wrong type) is ignored with a log line.
Max message size 2 MB. The current topic/style selection lives server-side in the session
(survives reconnects).

Admin watchers: `GET /ws/admin` (requires admin cookie) receives the same PICTURE binary
frames and `status` messages, plus `{"t":"stats",...}` once a second. Read-only.

## Admin HTTP API (cookie auth, JSON)

- `POST /api/admin/login` `{"pin":"1234"}` -> sets HttpOnly signed cookie `sg_admin`
  (HMAC-SHA256 of an expiry timestamp with secret_key, 30 days). 401 on wrong PIN.
  Basic rate limit: after 5 wrong attempts, 30 s lockout (in-memory).
- `POST /api/admin/logout`
- `GET /api/admin/state` -> `{"topics":[...all],"styles":[...all],"topic","style","view_mode","backend","stats":{...}}`
- `POST /api/admin/styles` `{"label","emoji","prompt"}` -> creates custom style.
- `PATCH /api/admin/items/{kind}/{id}` `{"hidden"?:bool, "label"?, "emoji"?, "prompt"?}` (kind = topics|styles; prompt editable only for custom)
- `DELETE /api/admin/items/styles/{id}` (custom/image only)
- `POST /api/admin/order` `{"kind":"topics"|"styles","ids":[...]}`
- `POST /api/admin/select` `{"topic"?:id,"style"?:id,"view_mode"?:mode}` -> sets current
  selection / default view mode and pushes to the kid page.
- `POST /api/admin/style-image` multipart `file`, `label`, `emoji` -> screens via worker
  (`ScreenUploadJob`), stores in `data/style_images/`, creates kind "image" style. Respond
  `{"ok":true,"style":{...},"note":"Style pictures need phase 2 (SD 1.5 + IP-Adapter); stored but not applied yet."}` when the backend doesn't support image styles. Max 8 MB, image types only.
- Every successful mutation broadcasts `{"t":"styles",...}` to the kid socket.
- `GET /healthz` -> `{"ok":true,"backend":..,"state":..}` (no auth).

Static: `/` -> `static/index.html`, `/admin` -> `static/admin.html`, `/static/*`.
Disable caching on HTML (Cache-Control: no-cache) so the iPad picks up updates.

## Kid page (`static/index.html` + modules)

Landscape-first iPad layout, also usable in portrait. Big touch targets (>= 48px).

- Toolbar along the top: colour swatches (8–10 big round swatches incl. black, white as
  "eraser") + native `<input type="color">` as "more colours"; brush width slider
  `<input type="range" min=4 max=40>` with a live dot preview; Clear button (big, with an
  icon); view-mode toggle (Auto / Sketch / Picture / Split icons); topic and style buttons
  (emoji + label, horizontally scrollable rows, current one highlighted). Connection dot
  (green/yellow/red).
- `canvas.js`: 512x512 backing canvas, white background, scaled to fit via CSS keeping
  square aspect; Pointer Events (`pointerdown/move/up/cancel`), `setPointerCapture`,
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

## Admin page (`static/admin.html` + admin.js)

Phone-first layout. If not logged in: big PIN pad (digits) -> login. Then sections:

1. Now: live picture (from `/ws/admin`), backend + state + fps, current topic/style
   pickers, default view mode select.
2. Styles: list of all styles (presets + custom + image) with hide/show toggle, up/down
   reorder, edit label/emoji (and prompt for custom), delete (custom/image).
   "Add a style" form: label, emoji, prompt text.
3. Style picture: file input (camera/library), label, emoji, upload; show the server's
   note about phase 2.
4. Topics: hide/show + reorder.
5. Log out.

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
  shared by both pages.
