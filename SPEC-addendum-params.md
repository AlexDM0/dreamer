# Addendum: live diffusion controls on the admin page

_Updated 2026-10-26 to match the code; docs/setup.md is the user guide._ The table below lists every field of `GenParams` in `server/params.py` (the source of truth for ranges); the original request covered the first twelve, refinement, upscale, resolution and picture-model fields were added later.

Alex's request (2026-10-25): the admin page has all relevant parametric controls for the diffusion.

All runtime-changeable without restart. Backend gets a new method:
`set_params(params: GenParams) -> None` (worker thread only; applied before the next frame).

`GenParams` (dataclass, persisted in data/params.json, defaults from config.model):
| field | type / range | applied via (StreamDiffusionBackend) |
|---|---|---|
| t_index_list | list of 1..8 ints (`T_INDEX_MAX_STEPS`), each 0..49, strictly increasing | wrapper.update_stream_params(t_index_list=...). Phase 1 limited this to 1..4 steps and warned about engine rebuilds. Now up to `model.max_batch_steps` (1, 2, 4 or 8; default 4) steps run together as one UNet batch (the TensorRT engine is built for that range), more run one after another at batch 1, so any count up to `model.max_steps` (default 8) works without a rebuild; changing `max_batch_steps` in config rebuilds the engines once. The admin UI has a "Creativity" slider (maps to the first index) and a "Draw reaction" steps slider (1..max_batch_steps); on an apply failure the params revert and the error is reported. |
| seed | int 0..2^31-1, plus "new random seed" button | update_stream_params(seed=...) |
| controlnet_enabled | bool | update_stream_params(controlnet_config=[{..., "enabled": bool}]) |
| controlnet_scale | float 0.0..2.0 | update_stream_params(controlnet_config=[{..., "conditioning_scale": x}]) |
| safety_checker | bool | update_stream_params(use_safety_checker=...) — admin UI must show a red warning when off; switching it ON fails (and reverts) if the backend started with `model.safety_checker: false` |
| safety_threshold | float 0.05..0.95 | update_stream_params(safety_checker_threshold=...) |
| line_threshold | int 10..120 (pixel distance from white that counts as a line) | server-side imaging.make_line_map |
| line_polarity | white_on_black / black_on_white | server-side imaging |
| line_dilate | int 0..3 px | server-side imaging |
| init_mode | colors / sketch / blank (default colors) | server-side imaging: what img2img starts from. `colors` drops the strokes and keeps soft colour clouds of the coloured strokes (dark outlines add none), so the sketch guides the picture without being traced; the line map carries the outline. Ignored (sketch used) when ControlNet is off. |
| init_blur | int 4..64 px (default 24) | Gaussian radius of the colour clouds in `colors` mode |
| jpeg_quality | int 50..95 | server-side encode |
| refine_steps | int 0..20 (default 8 from `model.refine_steps`; 0 = off) | after the child stops drawing, the drawing's path continues step by step up to this many steps in total (`server/refine.py`) |
| refine_model | path / path_locked / sdedit / anchored / anchored_edges / hires_tiles / anchored_hires (default `anchored`) | which refinement model runs (`REFINE_MODELS`); SD-Turbo only, the TCD model re-renders with more steps (`refine_own`) |
| upscale | off / x2 (default from `model.upscaler`) | AI upscale of refined pictures, never while drawing (`server/upscale.py`; needs `models/realesr-animevideov3.pth`) |
| resolution | int, multiple of 64, 64..1024; admin offers 512, 640, 768, 896, 1024 (default `config.size`) | long side of drawings and pictures; the short side is 0.7x (canvas addendum). The TensorRT engines handle 384..1024. The admin page estimates GPU memory from `static/vram.json` and disables options over the 10 GB budget. |
| picture_model | turbo / tcd (default turbo) | `turbo` = SD-Turbo via StreamDiffusion, `tcd` = SD 1.5 + TCD (`backends/tcd.py`); switching makes the worker unload one backend and load the other |
| tcd_steps | int 1..8 (default 4) | steps per picture while drawing with the TCD picture model |

The verified update_stream_params signature (Daydream fork wrapper.py) includes: num_inference_steps, guidance_scale, delta, t_index_list, seed, prompt_list, negative_prompt, controlnet_config, ipadapter_config, use_safety_checker, safety_checker_threshold, ... SD-Turbo runs with cfg_type="none", so guidance_scale/negative prompt are NOT exposed (no effect).

API: `GET /api/admin/params` -> {params, limits (min/max/step/options per field), defaults, gpu (max_batch_steps, memory budget, the measured vram table), refine_models (id -> label)}; `PATCH /api/admin/params` partial update, validated + clamped server-side (400 on invalid type), returns applied params; `POST /api/admin/params/reset` -> defaults. Changes go into the worker via a thread-safe "pending params" slot (latest wins) applied before the next frame; apply errors are reported through on_status("error", ...) and the params revert.
Admin UI: a "Picture controls" section (picture model, creativity, draw reaction or TCD drawing steps, refine when they stop, refinement model, AI upscale, resolution with a GPU memory meter, follow the lines, surprise number, safety filter, plus an advanced group: line sensitivity, line thickening, line map, starting image, colour softness, picture quality) with sliders + numeric readouts, debounced (150 ms, `PARAMS_DEBOUNCE_MS`) PATCH while dragging so the parent sees the live picture change, a "Reset to defaults" button, short plain-language help text per control (with an info link to `static/help.html`) (e.g. "Creativity: how far the picture may stray from the drawing"). The stats line (fps, last frame ms) helps tune.
MockBackend implements set_params by varying its stylisation (creativity -> blur/tint strength, seed -> tint hue, line strength) so the e2e test can assert a visible change.
Tests: params validation/clamping, persistence, admin auth required, worker applies params before next frame, mock output changes with params; e2e: change creativity on admin, picture on kid page changes.
