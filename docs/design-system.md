# Dreamer design system

Dreamer has two faces: a **kid page** (`static/index.html`, iPad, a 4-6-year-old's
fingers) and an **admin page** (`static/admin.html`, a parent's phone). Both are built from
one token layer, `static/theme.css`. This document covers those tokens, the components
built from them, and the rules that keep the two pages consistent.

**Direction:** calm, warm and grown-up, like a well-made creative tool. The surfaces are
warm off-white "paper", there is one confident accent (Iris, a dusky blue-violet), the
elevation is soft, and everything uses the system typeface. The playfulness lives in the
logo, the micro-interactions (spring presses, the save "pop", sparkles) and the child's own
content (colours, emoji, pictures). It does not come from candy colours or chunky
sticker shadows.

## 1. Tokens (`static/theme.css`)

The tokens come in three layers. **Only `theme.css` may contain raw colour values.** Page
CSS uses semantic tokens and scales only. The `--sand-*`, `--iris-*` and other primitives
must never appear in `style.css` or `admin.css`.

### 1.1 Primitives

| Ramp | Use | Key values |
|---|---|---|
| `--sand-0 … --sand-1000` | warm neutrals (surfaces, text, borders) | 0 `#ffffff`, 50 `#f7f5f2`, 100 `#f0ede8`, 200 `#e2ddd5`, 400 `#9a9388`, 600 `#5e5850`, 900 `#1c1a18`, 950 `#141311` |
| `--iris-50 … --iris-950` | the accent | 50 `#f3f2fe`, 300 `#aba3f7`, 500 `#6c5fe4`, 600 `#5848d6`, 700 `#4b3bc6`, 900 `#2a2550` |
| `--moon-100/300` | logo crescent cream | `#fff8ee`, `#fde9c4` |
| `--green-*`, `--amber-*`, `--red-*` | status | green 600 `#1e8556`, amber 600 `#c27a00`, red 600 `#cc3a31` |

### 1.2 Semantic colours

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | sand-50 | sand-950 | page background |
| `--surface` | white | sand-900 | cards, toolbars, inputs, chips |
| `--surface-raised` | white | sand-850 | dialogs, sheets |
| `--surface-sunken` | sand-100 | sand-1000 | wells: slider tracks, segmented controls, stat tiles, pick rows |
| `--surface-hover` / `--surface-pressed` | sand-25 / sand-150 | sand-850 / sand-800 | interaction states of neutral controls |
| `--surface-inverse` | sand-900 | sand-50 | toasts and hints |
| `--glass` | white 84% | sand-900 80% | translucent labels over pictures, the sticky header |
| `--scrim` | ink 38% | black 58% | behind dialogs and sheets |
| `--canvas` | white | white | drawing paper and picture frames (never themed) |
| `--text` | sand-900 | sand-50 | body text |
| `--text-muted` | sand-600 | sand-300 | labels, secondary copy (at least 4.5:1) |
| `--text-subtle` | sand-500 | sand-400 | placeholders, hints (at least 4.5:1 on `--surface`) |
| `--text-inverse` | sand-50 | sand-900 | text on `--surface-inverse` |
| `--text-on-accent`, `--text-on-danger` | white | white | text on filled accent/danger |
| `--border` | sand-200 | sand-800 | dividers, card edges |
| `--border-strong` | sand-400 | sand-600 | input/control outlines (at least 3:1) |
| `--accent` | iris-600 | iris-500 | primary buttons, selected chips, switches (white text at least 4.5:1) |
| `--accent-hover` / `--accent-pressed` | iris-700 / iris-800 | iris-600 / iris-700 | states |
| `--accent-text` | iris-700 | iris-300 | accent-coloured text and icons on surfaces |
| `--accent-soft` / `--accent-soft-strong` | iris-50 / iris-100 | iris-950 / iris-900 | tinted wells, selected backgrounds, focus halo on inputs |
| `--focus-color` | iris-500 | iris-400 | focus ring |
| `--success`, `--warning`, `--danger` (+ `-soft`, `-text`, `--danger-hover`) | | | status dots, alerts, destructive actions |
| `--celebrate` | amber-400 | amber-400 | the saved star, sparkles |
| `--brand-night`, `--brand-moon` | | | logo colours, for brand moments only |
| `--spectrum` | | | rainbow on the "more colours" swatch (content, not themed) |

**Theme switching.** Light is the default. Dark follows `prefers-color-scheme` unless the
page pins `<html data-theme="light|dark">`. The dark values are written twice, once inside
the media query (`:root:not([data-theme="light"])`) and once for `:root[data-theme="dark"]`.
Keep the two blocks identical. The kid page pins **light**, because the drawing paper is
white and a dark frame around it would fight the picture. The admin page follows the phone.

### 1.3 Scales

| Group | Tokens |
|---|---|
| Type families | `--font-sans` (system-ui / -apple-system / SF Pro / Segoe UI Variable / Segoe UI / Inter / Roboto), `--font-rounded` (ui-rounded / SF Pro Rounded, then sans; kid page only), `--font-mono`. No web fonts: the app is LAN-only and must work offline. |
| Type sizes | `--text-xs` 12, `--text-sm` 13, `--text-md` 15, `--text-base` 17, `--text-lg` 20, `--text-xl` 24, `--text-2xl` 30 (px). Inputs are at least 16px so iOS doesn't zoom. |
| Weights | `--weight-regular` 400, `-medium` 500, `-semibold` 600, `-bold` 700. Default to semibold for UI labels and bold only for page titles. Never use 800. |
| Leading / tracking | `--leading-tight` 1.2, `--leading-normal` 1.45; `--tracking-tight` -0.01em (headings, buttons), `--tracking-wide` 0.04em (uppercase tags) |
| Spacing | `--space-1` 4, `-2` 8, `-3` 12, `-4` 16, `-5` 20, `-6` 24, `-8` 32, `-10` 40, `-12` 48 (4px grid) |
| Radii | `--radius-xs` 6, `-sm` 10, `-md` 14, `-lg` 20, `-xl` 28, `-pill` 999 |
| Elevation | `--shadow-1` (resting chips, cards), `--shadow-2` (toolbar, buttons, selected states), `--shadow-3` (canvas, hero button), `--shadow-4` (dialogs, sheets, toasts), `--inner-edge` (hairline inside colour fills). Pair raised surfaces with a 1px `--border` ring (`box-shadow: 0 0 0 1px var(--border), var(--shadow-n)`). |
| Motion | `--dur-instant` 80ms, `-fast` 140ms, `-base` 200ms, `-slow` 320ms; `--ease-out`, `--ease-in-out`, `--ease-spring` (overshoot, for playful presses); `--press-scale` 0.96. With reduced motion, durations collapse to 1ms and both pages switch animations off. |
| Focus | `--focus-ring`: a 2px surface gap, then a 2px `--focus-color` ring. Use it on `:focus-visible` only. |
| Icons | `--icon-stroke` 2 (admin, small icons), `--icon-stroke-bold` 2.25 (kid page) |
| Touch | `--touch-min` 44px (admin), `--touch-kid` 48px (any kid control), `--touch-kid-lg` 56px (main kid tools) |

## 2. Components

The components below are listed with the classes JS relies on. **Never rename an id, a
`data-*` attribute or a class that JS or `tests/e2e/run.mjs` queries.** Restyle them
instead.

### Buttons
| Variant | Kid page | Admin page | Look |
|---|---|---|---|
| Primary | `.btn.big` | `.btn.primary` | `--accent` fill, `--text-on-accent`, `--shadow-1/2`, pill. Pressed: `--accent-pressed` plus press scale. |
| Secondary | `.btn.ghost`, `.btn.more` | `.btn.ghost` | `--surface` fill with a 1px `--border` / `--border-strong` ring. |
| Ghost / destructive | n/a | `.btn.danger`, `.logout` | Transparent, `--danger-text`. Hover/pressed fills `--danger-soft`. |
| Icon | `.tile` (icon over a label) | `.icon` (44px square) | Neutral, `--text-muted`. Active: `--accent-soft` with `--accent-text`. |

Heights: 56-60px on the kid page and 48px on the admin page. Labels are semibold with
tight tracking. Only one primary button may appear in a group.

### Hero action (kid page)
`.side-btn.save .disc` is the only filled accent disc on the kid page (76px, `--shadow-3`).
Gallery and New use neutral `--surface` discs. The saved state turns the star
`--celebrate` with a spring pop. In portrait the rail becomes pill buttons.

### Chips (kid topic/style picks)
`.pick` is a `--surface` chip with a hairline border and the parent's emoji (user content)
plus a label. `.pick.active` is filled `--accent` with white text: that's unmissable for a
child with no reading needed. Rows (`.pick-row`) sit in a `--surface-sunken` well and
start with a line icon (`.row-icon`: globe = worlds, palette = styles).

### Colour swatches
`.swatch` is a plain colour disc with an `--inner-edge` hairline. `.active` adds a
`--surface` gap ring and a `--text` ring. The eraser is a sunken disc with a line icon. The
"more" swatch is `--spectrum` with a drawn "+" chip. The drawing colours themselves live in
`toolbar.js` (content, not chrome).

### Segmented control
The kid `.modes` / `.mode` control and the admin `.tabs` / `.tab` control share one
pattern: a `--surface-sunken` track, transparent segments in `--text-muted`, and the active
segment on `--surface` with `--shadow-1/2` and a hairline border. On the kid page the
active icon turns `--accent-text`.

### Cards
`.card` (admin) is `--surface`, `--radius-lg`, a 1px border plus `--shadow-1`, and
20/16px padding. Its title is `--text-lg` semibold, followed by a `.muted` description.
There are no coloured bars or ornaments.

### Sliders
The kid range has an 8px pill track filled with `--accent` up to `--fill` (set by
`toolbar.js`) and a 30px white thumb with an accent core and `--shadow-2`. Admin ranges use
native `accent-color: var(--accent)`, which looks right on iOS and is accessible.

### Switch (admin)
`.switch` is a checkbox: 48×28 with a 44px hit area, `--border-strong` when off and
`--accent` when on, and a spring knob.

### Form fields (admin)
`.field` has a `--text-sm` semibold muted label, a 48px input, an inset 1px
`--border-strong` outline, and `--radius-sm`. On focus it shows a 2px `--accent` inset
plus a 4px `--accent-soft-strong` halo. The select chevron is drawn with gradients in
`--text-muted`, so it follows the theme.

### Toasts and hints
Admin `.toast` and kid `.hint` use `--surface-inverse` / `--text-inverse`, pill-shaped,
with `--shadow-3/4` and a spring entrance. Errors use `--danger`. A hint action button
(`.hint-btn`) is accent.

### Dialogs
The kid `.overlay .card` and the admin `.sheet` / `.sheet-card` sit on `--scrim` with a
light blur, on `--surface-raised`, with `--shadow-4` and `--radius-xl`. Kid dialogs lead
with a round `.card-icon` (an `--accent-soft` circle with a line icon) and use big,
plainly worded buttons. The admin sheet is a bottom sheet with a grabber.

### PIN pad (admin)
`.pin-pad` buttons are 64px+ `--surface` keys with a hairline border, tabular `--text-2xl`
medium digits, and a pressed state of `--surface-pressed` plus press scale. Backspace
(`.key-soft`) is transparent; OK (`.key-go`) is accent. The dots (`.pin-dots span`) are
outlined in `--border-strong` and filled `--accent` when entered. A wrong PIN shakes.

### Status dot
The kid `#conn` (a badge on the logo mark, with a `--surface` ring) and the admin
`.pill i` share one mapping: ready → `--success`, connecting/busy/error → `--warning`
(pulsing), off/offline → `--danger`, replaced → `--border-strong`. The admin pill is a
`--glass` label over the live picture.

### Paper picker and minimised toolbar
The paper picker (`static/paper.js`) is a raised sheet with 8 big round swatches; the
current paper has the accent ring. The paper colours, like the pen colours, are content and
live in JS, not in `theme.css`. Minimised, the kid toolbar keeps only the colour swatches,
the brush slider and the expand button; the drawing then takes all free space at its own
aspect ratio.

### Help page
`static/help.html` explains each admin picture control with real rendered examples
(`static/help/`, made by `tools/make_help_images.py`). The default value's figure has the
accent border and a "default" badge. The admin page's ⓘ links (`.params .info`) open it
at the control's anchor in a new tab.

### Empty states
`.empty-icon` is an 88px `--accent-soft` circle with a line icon, followed by an
`--text-xl` semibold line and an optional muted sub-line.

## 3. Icons

- Inline SVG, `viewBox="0 0 24 24"`, `fill="none"`, `stroke="currentColor"`, round caps and
  joins. Stroke widths come from `--icon-stroke-bold` (kid) and `--icon-stroke` (admin),
  set in CSS, not in the markup.
- Icons inherit colour from their control, so they never carry their own hex.
- Masked icons (CSS `mask` with a data-URI SVG, as in the split-view pane badges) use
  `stroke='black'`. A mask only reads alpha, so this is not a colour.
- **Emoji are content, not chrome.** Parent-defined topic/style emoji are shown as given.
  Emoji are allowed in playful copy and the save sparkle burst, but never as UI icons.

## 4. Kid page vs admin page

| | Kid page | Admin page |
|---|---|---|
| User | 4-6-year-old, iPad, landscape and portrait | Parent, phone |
| Theme | pinned light (`data-theme="light"`) | follows the system (light/dark) |
| Type | `--font-rounded`, medium base, labels optional | `--font-sans`, regular base |
| Touch | at least 48px everywhere, 56px+ for main tools, 76px for Save | at least 44px, buttons 48px |
| Reading | none needed for core actions: every action has an icon, and labels are a bonus | normal copy, helper text under controls |
| Accent use | Save, the selected chip, the active mode, slider fill | primary buttons, switches, values, links |
| Motion | spring presses, save pop/gulp, star burst, fly-to-gallery | quiet: fades, sheet slide, small spring on the toast and switch |
| Density | generous; one toolbar, one stage, one rail | compact cards in a single column (max 560px) |

## 5. Rules

1. Page CSS uses semantic tokens only. If a value is missing, add a semantic token to
   `theme.css` (with a dark value) rather than reaching for a primitive or a hex.
2. Elevation is a soft shadow plus a hairline border. Never use an offset "sticker" shadow
   or a `0 4px 0` bottom bevel.
3. Use one filled accent per view region. Secondary actions are neutral.
4. Text meets AA: at least 4.5:1 for body (the `--text-muted` and `--text-subtle` pairs
   are chosen for this) and at least 3:1 for control outlines (`--border-strong`).
5. Every interactive element shows `--focus-ring` on `:focus-visible`.
6. Respect `prefers-reduced-motion`: there are token-level collapses plus a global
   animation kill in both pages.
7. Touch targets never drop below the page's touch token, even in compact media queries
   (the iPad-mini landscape query keeps 48px; only phone-landscape fallbacks go smaller).

## 6. Logo

`static/favicon.svg` (the mark) and `static/logo.svg` (the mark plus the wordmark) show
**the dreamer**: a sleeping child (closed eyes, rosy cheeks) resting in a cream crescent
moon that holds them like a cradle. A thought trail of two dots rises to an amber sparkle:
the dream that turns into a picture. It sits on an iris "night sky" tile, so it reads on
light and dark backgrounds and down to 16px (tile, crescent, head and star survive).

The wordmark is "Dreamer" in the system sans at weight 650 with tight tracking. Its fill
switches to `#f7f5f2` under `prefers-color-scheme: dark` (inside the SVG). Use the mark
alone wherever the space is square (kid toolbar, favicon, the "another screen" dialog) and
the full logo in the admin header and the README. `static/icon-180.png` is the mark
rendered full-bleed on the same gradient, because iOS rounds the corners itself.
