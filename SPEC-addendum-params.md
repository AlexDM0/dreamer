# Addendum: live diffusion controls on the admin page

Alex's request (2026-10-25): the admin page has all relevant parametric controls for the diffusion.

All runtime-changeable without restart. Backend gets a new method:
`set_params(params: GenParams) -> None` (worker thread only; applied before the next frame).

`GenParams` (dataclass, persisted in data/params.json, defaults from config.model):
| field | type / range | applied via (StreamDiffusionBackend) |
|---|---|---|
| t_index_list | list of 1..4 ints, each 0..49, strictly increasing | wrapper.update_stream_params(t_index_list=...) — VERIFY: changing the list LENGTH may exceed the TensorRT engine batch range; admin UI offers "steps" 1..4 and a "creativity" slider that maps to the first index; warn in UI that step-count changes may trigger an engine rebuild or fail, and on failure revert to previous params and report error |
| seed | int 0..2^31-1, plus "new random seed" button | update_stream_params(seed=...) |
| controlnet_enabled | bool | update_stream_params(controlnet_config=[{..., "enabled": bool}]) |
| controlnet_scale | float 0.0..2.0 | update_stream_params(controlnet_config=[{..., "conditioning_scale": x}]) |
| safety_checker | bool | update_stream_params(use_safety_checker=...) — admin UI must show a red warning when off |
| safety_threshold | float 0.05..0.95 | update_stream_params(safety_checker_threshold=...) |
| line_threshold | int 10..120 (pixel distance from white that counts as a line) | server-side imaging.make_line_map |
| line_polarity | white_on_black / black_on_white | server-side imaging |
| line_dilate | int 0..3 px | server-side imaging |
| jpeg_quality | int 50..95 | server-side encode |

The verified update_stream_params signature (Daydream fork wrapper.py) includes: num_inference_steps, guidance_scale, delta, t_index_list, seed, prompt_list, negative_prompt, controlnet_config, ipadapter_config, use_safety_checker, safety_checker_threshold, ... SD-Turbo runs with cfg_type="none", so guidance_scale/negative prompt are NOT exposed (no effect).

API: `GET /api/admin/params` -> {params, limits (min/max/step per field), defaults}; `PATCH /api/admin/params` partial update, validated + clamped server-side (400 on invalid type), returns applied params; `POST /api/admin/params/reset` -> defaults. Changes go into the worker via a thread-safe "pending params" slot (latest wins) applied before the next frame; apply errors are reported through on_status("error", ...) and the params revert.
Admin UI: a "Picture controls" section with sliders + numeric readouts, debounced (~150 ms) PATCH while dragging so the parent sees the live picture change, a "Reset to defaults" button, short plain-language help text per control (e.g. "Creativity: how far the picture may stray from the drawing"). The stats line (fps, last frame ms) helps tune.
MockBackend implements set_params by varying its stylisation (e.g. creativity -> blur/tint strength, seed -> tint hue) so the e2e test can assert a visible change.
Tests: params validation/clamping, persistence, admin auth required, worker applies params before next frame, mock output changes with params; e2e: change creativity on admin, picture on kid page changes.
