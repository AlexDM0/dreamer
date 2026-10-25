# Independent adversarial review: explainer, README, showcase

Second review, done separately from the authoring session as Alex asked: two reviewers with
only the repo, checking every claim against the code and the measurement docs, opening the
explainer in a browser at phone, iPad and desktop sizes, and reading the README as a
sceptical parent. Reviewed at main `72375e3`. No files were changed by the review; this list
is for the fixes.

Verified as correct (no action): anchored model default with 8 steps (6 rungs, 44 ms),
engines for batch 1–4, upscaler only on refined pictures and off by default (31 ms,
+0.23 GB), 5.1 / 7.4 GB, 56 → 186 ms for 512 → 1024 px, ControlNet 0.55, dark mode default,
custom topics, the Trash, no price or buying wording, every relative link and image path in
README and docs/showcase, the explainer's settings, drift arrays and averages against
`refinement-results.json`, every interactive control, no page-level horizontal scroll, dark
mode, no console errors. slides.md works as a deck.

## High

1. **Parent gate: one PIN entry on the iPad unlocks the parent page for 30 days.**
   (code: `server/auth.py:26` `COOKIE_MAX_AGE = 30 * 24 * 3600`, `static/gate.js` skips the
   PIN when the cookie is valid; docs: README.md:87-88, slides.md:87, facts.md:36.) After a
   parent logs in once from the iPad, a child tapping the Dreamer icon gets the whole parent
   page, including delete and the safety switch. **Code fix** (the gate was added from the
   cloud session; it is my bug): the gate always asks for the PIN, even with a valid cookie;
   a login from the gate gets a short cookie (for example 15 minutes, `{"pin":…, "short":
   true}` → `max_age=900`), and "Back to drawing" on the admin page logs out first. The
   phone keeps its 30-day login. Then the README line can stay as it is.
2. **The safety filter is described as broader than it is.** (README.md:101-104 and 143,
   slides.md:166, captions.md:179.) It is an NSFW detector (`NSFWDetectorEngine`,
   streamdiffusion.py), and the strictness slider can loosen it. Parents will read "checks
   every picture" as general screening. Fix (README:101): "**A filter checks every picture
   for nudity and adult content.** … It does not judge whether a picture is scary; hiding
   topics and sitting with your child covers that. You can make it stricter or looser on
   the parent page." facts.md:33 names "Freepik/nsfw_image_detector": no repo file names
   the model, so check or remove.
3. **The hardware line leaves out graphics memory.** (README.md:5, 17, 154-155.) Owners of
   6–8 GB RTX cards would be misled: the default engines use 7.4 GB plus ~0.8 GB for the
   desktop, and the meter budgets 10 GB. Fix (README:17): "**Needs:** a Windows 11 PC with
   an NVIDIA RTX graphics card with 10 GB of graphics memory (tested on an RTX 3080), about
   15 GB of free disk space, internet for the first start, and an iPad on the same Wi-Fi."
4. **Explainer: keyboard focus ring invisible on most controls.** (how-it-works.html CSS
   line 99 is overridden by later `box-shadow` rules at lines 139/141/145/162/202/243.)
   Fix: `.stage button:focus-visible, button.primary:focus-visible, .seg
   button:focus-visible, .pick:focus-visible, .pad:focus-visible { box-shadow:
   var(--focus-ring); }`.

## Medium

5. **Refinement timing is stated four different ways.** The real sequence (worker.py
   `SETTLE_DELAY = 0.15`): 150 ms quiet, one settle frame (~70 ms), 150 ms again, then the
   rungs; the last rung lands ~0.65–0.72 s after the last stroke, more with the upscaler on
   (31 ms × 6). Fix: how-it-works.html §3 line 474 ("When the child stops for 150 ms…") →
   "About a third of a second after the child stops (150 ms quiet, one settle frame,
   150 ms again), the polishing starts…"; captions.md:260 "within half a second" → "within
   about a second"; captions.md:289, slides.md:134 and showcase/README.md:10-12 → "about
   0.1, 0.2 and 0.3 s after refinement starts (measured with a 1-step draw reaction)";
   architecture.md:126-129 sequence diagram: add the settle frame and the second 150 ms;
   explainer simulator and "0.65–0.72 s": note the upscaler adds ~31 ms per rung.
6. **README never says Dreamer is not an App Store app, and setup sounds easier than it
   is.** (README.md:3-5, 159.) Fix: README:5 "Dreamer is not an App Store app: it runs in
   Safari, served by your PC." README:159 "Setup is technical (installing Python, running
   PowerShell scripts, editing a settings file); about half an hour if you're comfortable
   with that."
7. **Explainer §3 note (line 551) misreads the results.** The meadow's drift creeps up to
   7.9 at the last rung (not "mostly at the first rung"), and its detail rises; the cat
   fails on drift and detail, the house and dinosaur on detail only. Fix: "the cat (8.3 →
   7.1) and the meadow (7.4–7.9, creeping up) go over the drift target; the house and
   dinosaur stay under it but their detail dips after the first rung."
8. **Explainer table of contents is a hidden horizontal drag strip.** (nav.toc lines
   118-121.) At 390 px items 4–6 are off-screen with no hint; at 820 px item 6 is clipped.
   Alex has said horizontal dragging is not ideal. Fix: `flex-wrap: wrap` below ~860 px, or
   a "Sections" button with a menu.
9. **Explainer on phones: the network diagram labels render at ~6 px.** (§1 SVG, viewBox
   640.) Screenshot: review scratch `phone-sheet0.png`. Fix: a stacked SVG variant under
   560 px or HTML labels; `.lat-row { grid-template-columns: 1fr }` on phones.
10. **GB units disagree between explainer and fact sheet.** The explainer divides by 1,024
    ("7.2 GB"), facts.md and gpu-results.md say 7.4 GB (7,418 MB). Also "147 ms, 4 steps"
    (vram.json run) vs "117 ms" (gpu-results). Fix: use the docs' decimal GB, and label the
    4-step time with its measurement run.
11. **Explainer misdescribes when sketches are sent.** (§1 line 371, stage 1 line 714:
    "With every stroke it sends…".) `static/net.js` sends whenever the canvas is dirty and
    nothing is in flight. Fix: "While the child draws, it keeps sending the stroke layer: a
    new one as soon as the previous picture comes back."
12. **Privacy wording overpromises slightly.** (README.md:110-111, 134-135,
    architecture.md:93-94, slides.md:167.) `server/offline.py`: changing the model or
    switching the safety checker on can download again. Fix: "...after that Dreamer runs
    without contacting the internet, unless you change the picture model in the settings
    file." facts.md:35 "the setup script only fetches the 2.5 MB upscaler" → "of the
    models, the setup script only fetches the 2.5 MB upscaler" (it also installs PyTorch
    and packages).
13. **Stale plans and notes.** slides.md:212 "What's next" lists SD 1.5 + TCD and
    lineart/HED, which were tested and rejected → "try a lower ControlNet strength for
    refinement only (SD 1.5 + TCD and lineart/HED were tested and lost to the current
    setup)". showcase/README.md:49 "(not generated yet)" for the ControlNet sheets: they
    exist, delete it.

## Low

14. Caption mismatches: captions.md:189 "keeps its pose and its tail" (the tail became a
    paw) → "keeps its ears, face and round body"; captions.md:242 "a butterfly" → "in the
    drawing" (it became a blue blob); captions.md:277-279 promise resolution and ⓘ buttons
    the cropped screenshot doesn't show.
15. Every iPad screenshot shows the old horizontal topic/style rows with cut-off labels.
    After the one-button picker lands, re-run `tools/make_showcase.py` and reword slides.md
    slide 3.
16. First-start time: "10–25 min" (facts.md:26, slides.md:205) vs "10 to 30" (setup.md:52):
    use 10–30. Test count "256" (facts.md:53, slides.md:206): 259 today; say "about 260".
    Round trip 0.1–0.2 s (facts.md:17, slides.md:60) is from the 512 px run: label it.
17. Explainer details: the engine picker should say batch 3 can't be built and 8 is
    allowed (`config.py BATCH_STEPS`); the "ControlNet ≈ 27 ms of a 49 ms frame" chip mixes
    512 px and 640×448 runs; line 507 "repaints the scene from about 6 steps" → "drifts from
    about 6 steps; at 12–20 it paints a different scene"; §6 picker lacks the meadow sheet
    that exists; Crayon emoji ✏️ vs 🖍 in styles.py; line 581 "the iPad keeps the previous
    one" → "the PC sends the previous picture again"; legend: "whole round trip" and
    "polishing rung" share a colour in light mode, settle block unlabeled.
18. README.md:41 numbers only one of two screenshots ("Upright works too · 7. The parent
    page"); slides.md:99 stray space before a comma; README:129-131 could add that anyone on
    the Wi-Fi can open the drawing page and take over the iPad's session (the kid socket
    has no PIN). facts.md:3 driver "591.59" is in gpu-results.md, fine.
19. The explainer is standalone only inside the repo: it loads images from
    `../screenshots/examples/` and `research/img/`. If it is ever shared on its own, inline
    or bundle them.
