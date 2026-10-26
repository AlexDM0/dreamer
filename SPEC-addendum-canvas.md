# Addendum: background layer, drawing shape, minimisable toolbar (Alex, 2026-10-25)

_Updated 2026-10-26 to match the code; docs/setup.md is the user guide._ This replaced the phase 1 square 512x512 white canvas. The toolbar minimise button lives in `static/app.js`, the background colour picker in `static/paper.js`.

"I want to be able to set the background color, keep that as 2 layers so I can always change
it later." / "background should be selected when a new image is started" / "default black" /
"The ui should be minimisable, leaving only the color and size slider. Max space for the
drawing. The iPad aspect ratio and "up" direction should be respected."

## Drawing model

- A drawing has a **background colour** (`#rrggbb`, default `#000000`) and an **orientation**
  (`landscape` | `portrait`), fixed when the drawing starts (New, Start fresh, first load
  without a draft) from the device orientation at that moment.
- The kid page keeps **two layers**: the background (a solid colour, changeable any time) and
  a transparent **stroke layer**. The eraser and Clear only touch strokes.
- Drawings and pictures have the iPad's shape, not a square. `config.size` is the long side
  (default 640, a multiple of 64); the admin "Resolution" param (`resolution`, 512 to 1024) overrides it live and `canvas_dims(cfg, long_side)` computes the shapes. The short side is 0.7× rounded to 64: landscape 640×448,
  portrait 448×640 (`config.canvas_dims`). When the iPad rotates mid-drawing, the drawing
  stays upright and is letterboxed. Only a new drawing takes the new orientation.

## Protocol

- `ready` gains `canvas: {"landscape": [w, h], "portrait": [w, h]}`.
- Kid → server `{"t":"canvas","background":"#rrggbb","orientation":"landscape"|"portrait"}`:
  sent after `ready`, and whenever background or orientation changes. Invalid values are
  ignored. Once a page has sent it, its sketch PNGs are **stroke layers**: RGBA with a
  transparent background, at its orientation's size. A page that never sends it is treated as
  before: an opaque sketch on white.
- Pictures come back at the drawing's size.
- `continue` and `ready.draft` gain `background` and `orientation`.

## Server

- Ink is opaque pixels of the layer (`line_threshold` applies to alpha), so white strokes on
  a dark background are lines like any other. A blank layer is answered with a picture in
  the background colour.
- The img2img start image (`init_mode`) uses the background colour as its ground: `colors` =
  soft clouds of the coloured strokes over the background, `blank` = the background,
  `sketch` = strokes over the background. With ControlNet off, strokes over the background.
  So a black background gives a night-time starting point.
- Switching orientation (or the resolution param) rebuilds the StreamDiffusion wrapper at the other size (`set_size`). The TensorRT
  engines have dynamic shapes (384 to 1024), so they are reused. The worker reports "busy"
  meanwhile.

## Gallery

- `sketch.png` stores the stroke layer as sent, never merged with the background. The
  background and orientation are stored in the artwork's `params_json` under `canvas`, and the
  artwork JSON (kid and admin) has `background` and `orientation`. Artworks from before layers
  report `#ffffff` / `square`: their sketch is an opaque drawing on white.
- Pages show a sketch over its background colour.
