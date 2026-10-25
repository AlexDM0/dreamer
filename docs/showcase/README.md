# Dreamer showcase kit

Everything needed to put together a short presentation (or a social post) about Dreamer.
All pictures in the kit are real: they were made on the development PC (RTX 3080) by the
actual app, driven by scripts. Nothing is mocked up by hand.

**Settings used for the screenshots:** AI upscale x2 was switched on, so the refined
pictures are 1280×896. The app's default is upscale off; say "with AI upscale on" when
showing them. The help-page shots (`*-help-refine.png`) still show the earlier "continue
the path" refinement strip (4/8/12/20 steps, with cumulative times), not the current
anchored model; caption them that way until they are regenerated.

## What is in it

| File | What it is for |
|---|---|
| [slides.md](slides.md) | An 11-slide outline: headline, talking points, the image to show and speaker notes per slide |
| [captions.md](captions.md) | One-line and longer captions for the 12 strongest screenshots |
| [facts.md](facts.md) | The fact sheet: every number with its source. Use only these numbers on slides |
| [architecture.md](architecture.md) | How it works, as Mermaid diagrams (big picture, drawing → picture, draw fast then refine) plus "things we learned" |
| [../../README.md](../../README.md) | The product page (App Store style), with the screenshot gallery and before/after table |
| [../../screenshots/](../../screenshots/) | All screenshots; [manifest.json](../../screenshots/manifest.json) has a title, caption, device, theme and features for each |
| [../../screenshots/examples/](../../screenshots/examples/) | Six raw before/after pairs (`*-sketch.png` = the drawing on its paper, `*-picture.jpg` = the refined picture, upscaled 2x) |
| [../research/img/](../research/img/) | Refinement research sheets (`refine-<drawing>.png`: rows = models, columns = first picture then each step; `-zoom` = centre crops) |
| [../img/](../img/) | Early GPU bring-up pictures (scribble polarity, pen lines showing through, live ocean test) for the "behind the scenes" slide |
| [../gpu-results.md](../gpu-results.md), [../research/refinement-models.md](../research/refinement-models.md) | The measurements behind the numbers |

Screenshot naming: `<device>-<orientation>-<artwork or screen>[-<variant>].<ext>`, for
example `ipad-landscape-cat-split.jpg` or `phone-admin-now-light.png`. Phone shots of the
parent page come in a `-dark` and a `-light` version. UI shots are at 2x device pixel ratio
(iPad 2360×1640, phone 780×1688).

Rendering the Mermaid diagrams: GitHub renders them in [architecture.md](architecture.md)
directly. For slides, paste a block into [mermaid.live](https://mermaid.live) and export a
PNG or SVG, or use any Mermaid CLI.

## Regenerating the visuals

All commands run on the Windows PC from the repo folder, with the GPU backend set up
(see [../setup.md](../setup.md)). Stop a running Dreamer first, so the GPU is free.

| What | Command | Writes |
|---|---|---|
| All screenshots + example pairs | `.venv\Scripts\python.exe tools\make_showcase.py` | `screenshots/*`, `screenshots/examples/*`, `screenshots/manifest.json` |
| Only some groups | `.venv\Scripts\python.exe tools\make_showcase.py --only kid` (groups: `kid`, `admin`, `help`, `examples`) | the files of those groups |
| Help-page example pictures | `.venv\Scripts\python.exe tools\make_help_images.py` | `static/help/*.jpg`, `static/help/data.json` |
| Refinement research sheets | `.venv\Scripts\python.exe tools\refine_lab.py` | `docs/research/refinement-results.md`, `refinement-results.json`, `docs/research/img/refine-*.png` |
| ControlNet comparison sheets | `.venv\Scripts\python.exe tools\controlnet_lab.py` | `docs/research/img/controlnet-*.png` and `controlnet-results.json` in `docs/research/` (not generated yet) |
| App icons | `.venv\Scripts\python.exe tools\make_icons.py` | `static/icon-*.png`, `static/dreamer.ico` |

Notes:

- `make_showcase.py` starts its own Dreamer with a temporary config (test PIN, free port,
  temporary data folder; your `data/` is never touched) and drives headless Microsoft Edge
  with real touch input. `--mock` runs it without a GPU, but then the pictures are only
  tinted sketches, so don't use those for a presentation.
- `refine_lab.py` and `controlnet_lab.py` use the drawings in `screenshots/examples/`, so
  run `make_showcase.py` (at least `--only examples`) first if those changed.
- Run `make_help_images.py` after the screenshots if the help page shots
  (`*-help-*.png`) should show the new example pictures, or run it first and then
  `make_showcase.py --only help`.
- `make_icons.py` renders from `static/favicon.svg` with headless Edge.

**Any change to the UI only needs these scripts re-run.** The file names stay the same, so
the README, the slides and the captions keep pointing at the right images. Only if a
screen is added or removed do `slides.md`, `captions.md` and the README gallery need a
manual edit. If a number changes (a new GPU, new defaults), update [facts.md](facts.md)
first and then the slides that cite it.
