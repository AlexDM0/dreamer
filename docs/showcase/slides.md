# Dreamer: presentation outline (11 slides, about 10 minutes)

Each slide has a headline, talking points, the image(s) to show and speaker notes. All
paths are relative to the repo root. Every number comes from [facts.md](facts.md),
[../gpu-results.md](../gpu-results.md) or
[../research/refinement-models.md](../research/refinement-models.md); the source is named
next to it. Don't add numbers that are not in those files.

Diagrams: the Mermaid blocks live in [architecture.md](architecture.md). Export them to
PNG/SVG (for example via mermaid.live) before building the deck.

---

## 1. Title: Dreamer

**Headline:** Dreamer: a child draws, and the picture appears while they draw.

- A drawing game for ages 4-7 on the iPad.
- Powered by a gaming PC with an NVIDIA RTX graphics card in your own home; no account.
- Parents choose what the pictures can be about.

**Image:** `static/logo.svg` above `screenshots/ipad-landscape-cat-split.jpg`

**Speaker notes:** Open with the picture, not the tech. "On the left is a cat drawn with a
finger; on the right is what Dreamer made of it, under the sea, while the cat was being drawn."
Let the image sit for a moment before talking.

---

## 2. The idea

**Headline:** Kids love to draw. What if the drawing could answer back?

- Small children draw in shapes and scribbles; they can't type, and many can't read yet.
- Image AI usually needs a typed prompt and a cloud account: not for a 4-year-old.
- Idea: the drawing *is* the prompt. The parent sets the theme, the child just draws.

**Image:** `screenshots/examples/rocket-sketch.png` next to `screenshots/examples/rocket-picture.jpg`

**Speaker notes:** Keep the problem human. The design brief was: nothing to type, nothing
to read, nothing leaves the house, and the child's own lines must stay recognisable.
Those four constraints drive every later slide.

---

## 3. Demo flow: draw

**Headline:** Pick a paper, pick a world, and draw.

- "New" asks for a paper colour first; black paper gives night scenes.
- Big colour swatches, one brush slider, worlds and styles as picture buttons.
- The picture follows every line within a fraction of a second.

**Images:** `screenshots/ipad-landscape-paper-picker.png`, then
`screenshots/ipad-landscape-rocket-live-split.png`

**Speaker notes:** Walk through it as a child would: tap a paper colour, tap Space, tap
Watercolor, draw a rocket. In the live split view the picture is already there while the
rocket is half drawn. Backend speed while drawing is about 12-15 pictures per second
(67-80 ms) at 640×448 (facts.md, "The experience"); through the real server the round trip is about
0.1-0.2 s (facts.md).

---

## 4. Demo flow: pause, keep, look back

**Headline:** It polishes when they pause, and keeps what they love.

- Stop drawing, and within half a second the picture gains detail (and, with AI upscale on, doubles in size).
- The yellow star saves picture and drawing together; the picture flies into the gallery.
- "My pictures": a big grid; tap one, see its drawing, and keep drawing.

**Images:** `screenshots/ipad-landscape-house-saving.jpg`, `screenshots/ipad-landscape-gallery.jpg`,
`screenshots/ipad-landscape-gallery-detail.png`

**Speaker notes:** Point out the little star burst on save. Children cannot delete
pictures; parents can, via a Trash. Drafts are saved automatically, so closing the iPad
doesn't lose a drawing. The screenshots were made with AI upscale on (2x, 31 ms, facts.md);
the app's default is off.

---

## 5. Demo flow: the parent page

**Headline:** Parents set the boundaries from their phone.

- PIN-protected, with a growing lock-out after wrong tries.
- Live view of the iPad's picture; change topic, style and look on the fly.
- Picture controls with a help page that shows real example pictures for every setting.

**Images:** `screenshots/phone-admin-login-dark.png`, `screenshots/phone-admin-picture-controls-dark.png`
(optionally `screenshots/phone-admin-now-dark.png`) (or the `-light` versions on a light deck)

**Speaker notes:** Topics and styles can be hidden, reordered, or added in a few words
("a pirate ship on a sunny sea"). The memory meter
(`screenshots/phone-admin-memory-meter-dark.png`) greys out settings that would not fit
in the graphics card's 10 GB, which is a nice detail if there is time. If you show the
"Now" card (`phone-admin-now-dark.png`), explain its stats row first: "Engine", "Speed" and
"Last picture" are technical readouts , not something parents need to watch.

---

## 6. How it works

**Headline:** One PC, one iPad, the home Wi-Fi.

- iPad sends the stroke layer + paper colour over a WebSocket; the PC sends a JPEG back.
- On the PC: one Python process (FastAPI), one GPU worker with StreamDiffusion + TensorRT.
- Line map → ControlNet keeps the shapes; colour clouds → start image; prompt from parent choices.

**Images:** diagrams "The big picture" and "From a drawing to a picture" from
[architecture.md](architecture.md)

**Speaker notes:** Model: SD-Turbo with a scribble ControlNet for SD 2.1, accelerated with
TensorRT engines (facts.md, "Under the hood"). The start image is soft colour clouds of
the child's strokes on the paper colour, so the lines guide the picture without being
traced. "Latest frame wins": sketches are never queued, so the picture never lags behind
a backlog.

---

## 7. Draw fast, then refine

**Headline:** Two speeds: keep up while drawing, polish while looking.

- While drawing: 2 steps run together as one batch, about 67-80 ms per picture (facts.md).
- After a 150 ms pause: up to 6 "anchored" rungs of about 44 ms each replace the picture, roughly 0.3 s in all (facts.md).
- A new stroke stops refining after the current rung and goes back to the fast mode.

**Image:** diagram "Draw fast, then refine" from [architecture.md](architecture.md), plus
`screenshots/desktop-help-refine.png`

**Speaker notes:** Caption the help strip truthfully: it shows the *earlier* "continue the
path" refinement, the same drawing while drawing and at 4, 8, 12 and 20 steps, with the
cumulative time for each. It is not the current model. Use it to set up the next slide:
look how at 8+ steps the scene starts to *change*. That was the problem the anchored
model solves.

---

## 8. The refinement story

**Headline:** "The first thing you see should not change, but detail is added every step."

- SD-Turbo is a 1-4 step model; continuing its path repaints the scene (max drift 35.8 of 255).
- Turned the rule into numbers: structure drift (stay under ~6), detail gain (rise), step size (shrink).
- Seven refinement models measured on six real drawings; "anchored detail ladder" won: drift 5.6, detail 1.75x, 44 ms per step.

**Images:** `docs/research/img/refine-cat-zoom.png` (centre crops: first picture vs last,
one row per model); optionally `docs/research/img/refine-cat.png` for the full strip

**Speaker notes:** All numbers from the Results table in refinement-models.md. In the zoom
crop the top row ("path") redraws the cat's face; the "anchored" row keeps the face and
sharpens it. Anchored = single-step rungs from late noise levels, chained, with the
low frequencies of the latent pinned to the first picture. Be honest: it passes the full
rule on 2 of 6 drawings, mostly missing by a hair on "detail must keep rising". Tiled
hires adds real resolution but costs about 295 ms per step.

---

## 9. Safety and privacy

**Headline:** Built for a child, checked every time, kept at home.

- Children never type: prompts are built on the PC from parent-approved topics and styles, ending in a fixed child-friendly phrase.
- An image safety classifier checks every picture before it is shown (about 4 ms); if flagged, the previous picture stays.
- No account. Drawings and pictures never leave the PC; models are downloaded once, on the first start, then Dreamer runs without contacting the internet.

**Image:** `screenshots/phone-admin-login-light.png` next to `screenshots/ipad-landscape-start.png`

**Speaker notes:** Sources: facts.md "Safety and privacy". Wording matters: the classifier
"checks every picture"; it can make mistakes, so never say "guaranteed safe". Parent
style uploads go through the same checker while the filter is on (with it off they are
stored unscreened). The parent page shows a red warning if the filter is switched off. Also
say plainly: pictures are made by AI and can surprise you; sitting with a child the first
few times helps.

---

## 10. Behind the scenes: what we learned

**Headline:** Five bugs that taught us how the pipeline really works.

- The model ignored the drawing: an engine cache mixed up UNets with and without ControlNet.
- Pictures one frame late: the batched pipeline returns the previous frame; fixed by prime and settle.
- Pen strokes showed in the picture: fixed with colour-cloud start images and a softer ControlNet (0.55).

**Images:** `docs/img/gpu-quality-2step.png` (lines still visible, before the fix) next to
`docs/img/gpu-no-pen-lines.png` (after)

**Speaker notes:** Full list in architecture.md "Things we learned" and gpu-results.md
"Problems found and fixed". The other two: flicker in step-by-step mode (fresh noise
every frame, fixed by re-seeding) and a ControlNet engine that cannot be built for batch
1-3 on TensorRT 10.12, which crashed later; now only batch sizes that build are allowed.
The black-control-image test (identical pictures with and without a drawing) is a good
story: always test that your conditioning actually conditions.

---

## 11. Numbers, and what's next

**Headline:** Measured on one RTX 3080 (10 GB).

- About 12-15 pictures/s while drawing; refinement done within half a second of the pause (about 0.3 s, facts.md).
- 7.4 GB graphics memory for the default engines; start-up about 8-10 s; first-ever engine build roughly 10-25 min, once (facts.md).
- 256 automated tests on the mock backend (facts.md).

**What's next:**

- Test with more children and iPads (so far one iPad at home).
- Style from a photo (needs SD 1.5 + IP-Adapter; uploads are already stored, and screened while the safety filter is on).
- Refinement: try SD 1.5 + TCD, and lineart/HED ControlNets (refinement-models.md, "Still to try").

**Image:** `screenshots/ipad-landscape-gallery.jpg` as a closing backdrop, or the six
before/after pairs from `screenshots/examples/` as a grid

**Speaker notes:** End on the children's pictures, not on the GPU. Limits to say out loud:
one child at a time, needs a Windows PC with an NVIDIA RTX card. Don't quote the old "about
4 s" refinement from gpu-results.md: it belongs to the superseded 4-to-20-step schedule.
