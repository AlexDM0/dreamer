# Code review, 2026-10-26

Full read-only review in four parts (server core, GPU backends and tools, frontend, tests /
scripts / docs). Every finding was checked against the code; the top ones were re-checked by
hand. Test suite: 265 passed (mock backend), about 55 s.

## Bugs to fix (ranked)

| # | Severity | Where | What goes wrong | Fix |
|---|---|---|---|---|
| 1 | High | `server/backends/tcd.py` `_nsfw` | The TCD safety check scales the short side to 448 and centre-crops: about 30% of every picture (left/right edges, or top/bottom in portrait) is never checked. The TensorRT path squashes the whole picture to 448×448. | Resize to (448, 448) like the fork, no crop. |
| 2 | High | `static/canvas.js` `_down` | Pinch tracking runs before palm rejection: a resting palm plus another touch can start a pinch and change the brush size while the pencil draws. | Palm check first; clear fingers on pen down; no pinch while the pen is active. |
| 3 | High | `Dreamer.cmd` | No `-ExecutionPolicy`: on a fresh Windows PC (policy Restricted) the desktop launcher fails. Works on this PC only because the policy here is Unrestricted. | `-ExecutionPolicy Bypass` on the launcher line. |
| 4 | Medium | `server/params.py` resolution limits | Live resolution accepts 64..1024, the engines handle 384..1024. A low value fails on every frame, triggers rebuild loops, and is saved in params.json, so it survives restarts. `config.size` up to 2048 is copied in unclamped. | Minimum 384 (or the offered options only); clamp the config value. |
| 5 | Medium | `server/worker.py` `_maybe_rebuild` | Rebuilding after repeated failures drops the backend without `close()`, so the new one loads while the old one still holds GPU memory. The failures that cause a rebuild are often out-of-memory errors, so it fails the same way. | Use `_close_backend()`. |
| 6 | Medium | `server/gallery.py` draft writer + `app.py` save | Draw, pause 10 s, Save, close the app: the draft never records which artwork it continues, so the next Save makes a duplicate. | Treat a changed source id as dirty. |
| 7 | Medium | `server/worker.py` start-up params | One saved setting failing at start-up (e.g. the safety filter turned off in config) resets all saved admin settings to defaults; during a model switch it can desync the picture model. | Revert per field; keep `picture_model` = the loaded backend. |
| 8 | Medium | `server/backends/streamdiffusion.py` `latent_ladder` | Model E ("+ outlines") crashes when ControlNet is disabled in config, and every refine then counts as a failure (rebuild loop). | Only push control images when a ControlNet is loaded and on. |
| 9 | Medium | `server/backends/streamdiffusion.py` `check_image_safe` | With the live safety filter switched off, admin uploads are stored unscreened although the checker is loaded (TCD screens regardless). | Screen uploads whenever a checker is loaded. |
| 10 | Medium | `static/net.js` | A connection attempt that hangs (iPad waking up) never times out, and "Try again" does nothing meanwhile. | Close a socket still connecting after ~5 s. |
| 11 | Low-medium | `server/app.py` clear/new | Clear/New doesn't stop the worker settling and refining the old drawing; the first new picture waits behind it. | Reset settle/refine state on clear. |
| 12 | Low-medium | `static/canvas.js` pinch | A pinch counts as an unsaved stroke ("not saved yet" after only resizing). | Separate event, no `unsaved`. |
| 13 | Low-medium | `server/refine.py` hires tiles | Tiles are checked one by one, not the finished picture; a flagged tile is replaced by the previous tile from another region. | Check the composite; fall back to the first picture. |
| 14 | Low | various | Gallery paging stops at 100 items; the 8 s Save timer is never cleared; a late-loading draft can overwrite a fresh drawing; Reset on the admin page can leave a slider stale; the admin socket has no liveness check; offline mode plus uncached TCD models gives an unclear error; fallback pictures can have the wrong size after an orientation change. | Small, local fixes each. |
| 15 | Low (design) | `server/auth.py` | 20 wrong PINs from the iPad gate lock the parent out too (global lock, up to 15 min). | Per-client limit, or let a correct PIN through a global lock. |

## Performance

- Every finger stroke reads the whole stroke layer back with `getImageData`, and `isBlank` scans
  pixels on every Save/Clear/New. Use an offscreen copy and a "has ink" flag.
- `worker._frame_inputs` composes an init image every frame and throws it away when a line map is used.
- Loading the upscaler runs a full upscale to throw away; hires tiles re-encode the same anchor tile on every rung; the low-pass kernel is rebuilt on each call; inputs are converted twice per frame. All minor.
- `styles._save` (fsync plus retries, up to ~0.75 s) runs on the event loop.
- The tests take about 55 s, mostly real sleeps; about 12 tests take most of it, and the fixed sleeps can flake on a busy PC.

## Consistency / cleanups

- Dead code: `refine._img2img`, `refine._spread`, `noise="fixed"`, the `refine_path` alias, `Config.data_path`, `StyleStore.delete_style`, `DraftAutosaver.dirty`, an unused import.
- Duplicates: `POLARITIES`/`INIT_MODES` (3 places), `T_INDEX_MAX` (2), `_model_attr`/`DEFAULT_SEED` (both backends), retry logic (gallery and styles), `loadImage`, `isLight`, PIN-pad rendering, the colour regex, `#000000` instead of `DEFAULT_BACKGROUND`.
- Stale text: `SPEC.md` and the SPEC addenda describe phase 1 (sizes, layout, controls); refine.py's header and its "2x" comment; worker comments about "step lists"; `kid_continue`'s docstring; the trackpad-pinch comment (Safari sends gesture events, which are blocked).
- `tools/tcd_lab.py` uses the slow VAE processor and strength 0.88, so its timings overstate TCD's cost.
- Tests: no tests for the upscaler, the TCD backend, or config.example.yaml vs. the defaults; several tests assert internals.
- Accessibility: dialogs don't move or trap focus; the connection status isn't announced; radio groups have no arrow keys.

Checked and fine: gallery and upload path safety, cookie HMAC and Origin check, sketch decode limits, thread hand-off, memory release on a model switch, safety checks on every SD-Turbo picture path, pinned versions consistent across the setup scripts.
