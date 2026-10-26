# Codebase review: quality, performance, consistency (2026-10-26)

Reviewed at main `af8b02a` (after the TCD picture model, the one-button picker, pinch and
wheel brush size). Scope: `server/` (all), `static/` (kid page JS/CSS, admin page skimmed),
`tools/` and `tests/` for lint only. Method: read the code; ran `ruff` (E, F, W, B, UP, SIM,
PERF, RUF) over Python; timed the per-frame CPU work on the cloud machine (4 cores; the PC's
desktop CPU is faster, so treat the milliseconds as relative). No GPU was used.

Overall: the code is in good shape. Clear module boundaries (one GPU thread, latest-wins
mailbox, per-socket send queues), careful input handling on the unauthenticated kid socket,
consistent design tokens (no stray colours in the CSS except one), docstrings that explain
*why*, and 265 tests. The findings below are mostly performance headroom on the GPU thread
and a few consistency gaps from the fast pace of the last two days.

## High

1. **Per-frame CPU work on the GPU thread is larger than it needs to be.** Every sketch
   goes through `decode_sketch` → `make_line_map` → `make_init_image` on the GPU worker
   thread before the GPU runs (`server/worker.py:_frame_inputs`). Timed on a 640×448
   sketch with 40 coloured strokes (cloud CPU):

   | step | now | cheaper equivalent |
   |---|---|---|
   | PNG decode | 4.8 ms | (keep) |
   | line map (`MaxFilter(3)`) | 12.5 ms | numpy 3×3 dilation: **0.3 ms**, pixel-identical |
   | init image "colors" (2 Gaussian blurs r=24 + `max/min(axis=2)` + float math) | **62 ms** | per-channel `np.maximum`/`np.minimum` instead of `axis=2` reductions (the top cost in the profile), and the blur at ¼ size then upscaled (mean diff 0.9/255) |
   | JPEG encode | 1.2 ms | (keep) |

   That is ~80 ms of CPU per frame here; on the PC probably a third to half of that, while
   the GPU frame itself is ~50 ms. The GPU sits idle during it. Fix: the two cheaper
   equivalents above, then measure `last_ms` on the PC before/after.
2. **Refinement and settle frames redo that CPU work for the same sketch.**
   `_refine_frame` decodes the PNG and rebuilds the init image and line map that
   `_process_frame` just made for the same `FrameJob` (`worker.py:386-389`); settle frames
   do too. Cache `(seq, init, line_map)` from the last processed job and reuse it: the first
   refine rung shows up that much sooner.
3. **Offline mode doesn't know the TCD models.** `offline.mark_ready` records only
   `base`/`controlnet`/`safety_checker` and is called only for the StreamDiffusion backend
   (`server/offline.py:_models`, `worker.py:219`). On a PC that has never switched to
   SD 1.5 + TCD, the next start goes offline and the first switch to TCD fails (its models
   are not in the cache), then falls back to SD-Turbo with an error. Fix: list the TCD
   model ids in the marker only after TCD has loaded once, and keep `apply()` from going
   offline while `picture_model: tcd` has never been marked; or have setup download them.

## Medium

4. **Send queues drop the newest picture, not the oldest.** `Peer._enqueue` drops the item
   being added when the 32-slot queue is full (`server/session.py:50-53`). For pictures
   that is the wrong way round: a phone on weak Wi-Fi watching /admin falls up to 32
   pictures behind and then stops getting new ones. Coalesce: keep at most one queued
   PICTURE frame per peer and replace it.
5. **TCD safety check only looks at the middle of the picture.** `TcdBackend._nsfw` resizes
   to 448 on the short side, then centre-crops 448×448 (`server/backends/tcd.py:166-169`),
   so on a 640×448 landscape picture the outer 96 px on each side are never checked (in
   portrait, the top and bottom). Check the whole picture (pad to square instead of crop,
   or run two overlapping crops).
6. **One sketch in flight limits the frame rate to one round trip.** `static/net.js` sends
   the next sketch only when the previous picture arrives, so iPad PNG encode + Wi-Fi up +
   CPU prep + GPU + Wi-Fi down are strictly serial. The server's mailbox is latest-wins
   already, so letting the page send the next dirty sketch as soon as the previous one has
   *arrived at the server* (or allowing 2 in flight) would overlap encode/upload with GPU
   time. Worth trying after 1–2.
7. **Two constants and two helpers are defined twice.** `T_INDEX_MAX = 49` in both
   `server/params.py:21` and `server/refine.py:23`; `DEFAULT_SEED = 2` and `_model_attr`
   in both backends (`streamdiffusion.py`, `tcd.py`). Import one from the other.
8. **Every finger stroke now copies the whole canvas at pen-down.** Since pinch,
   `_touchSnapshot` is taken for every touch stroke, not only once a pencil was seen
   (`static/canvas.js:332`). `getImageData` of 640×448 forces a GPU read-back on iPad
   Safari at the moment the finger lands. It is only needed while a second finger can
   still turn the stroke into a pinch (the first ~300 ms) or a pencil is in use; a cheap
   alternative is to start drawing the first finger's stroke after a short delay or after
   it moves a few pixels.

## Low

9. **Stale code and comments.**
   - `worker.py:22` imports `T_INDEX_MAX` unused.
   - `worker.py:35-36` keeps `refine_path` "for callers of the old name"; only a test
     uses it, so the test can call `refine.continue_path` directly.
   - `worker.py:109-112`: two comments for `_refine_plan`, the first is outdated (talks
     about step lists and the upscaler).
   - `session.py:184` `current_prompt()` is unused by the app (only tests).
   - ruff: 3 unused imports in tools, 32 `raise … from` missing inside `except` (B904,
     mostly `HTTPException` in app.py: harmless, but the traceback context is lost in
     logs), 11 `typing` imports that are `collections.abc` now.
10. **Store calls on the event loop.** Gallery calls consistently go through
    `asyncio.to_thread`; the style store's JSON writes (`add_custom_style`, `update_item`,
    `set_order`, `delete_item` in `server/app.py:664-727`) run on the event loop. They are
    tiny files, so this is consistency, not a measurable stall.
11. **TCD warmup renders at a fixed 640×448** (`tcd.py:97`) instead of the configured
    resolution; the first real picture at another size compiles again.
12. **One hard-coded colour.** `.pinch-ring` uses `#fff` and `rgba(0,0,0,.45)`
    (`static/style.css:1266`) where every other rule uses theme tokens.

## Checked and fine

Latest-wins mailbox and epoch handling (stale pictures never become the saved frame);
AdminGate body caps before parsing; PNG-only decoding with a pixel cap on the kid socket;
same-origin check on the kid WebSocket; parent-gate cookie (15 min on the iPad); gallery
SQLite in WAL with fsynced temp files and atomic replace; idle behaviour (the worker blocks
on a condition, ~17–18 W measured idle on the PC); no unused CSS classes in either page;
design tokens used throughout.

## Suggested order

1 and 2 (an afternoon, measurable on the PC), 3 (before anyone relies on offline + TCD),
4 and 5 (small), then try 6. The rest can ride along with the next change in those files.
