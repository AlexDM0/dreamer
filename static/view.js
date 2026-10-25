// Stage: the sketch canvas stacked over the picture canvas, plus the view modes.
//
// Modes (container class): mode-auto | mode-sketch | mode-picture | mode-split.
// Auto: the sketch shows instantly when a stroke starts; after the stroke ends it stays
// until the PICTURE made from the final frame (net "final" flag, i.e. seq >= finalSeq)
// arrives, then the picture cross-fades in (CSS opacity transition on .show-picture).

export const MODES = ["auto", "sketch", "picture", "split"];
const STORE_KEY = "dreamer.viewMode";
const GAP = 16; // px between panes in split mode; keep in sync with style.css --gap
const RING = 12; // px of pane frame/shadow that must stay inside the wrap

function readStoredMode() {
  try {
    const m = localStorage.getItem(STORE_KEY);
    return MODES.includes(m) ? m : null;
  } catch {
    return null;
  }
}

function storeMode(mode) {
  try { localStorage.setItem(STORE_KEY, mode); } catch { /* private mode etc. */ }
}

export class View extends EventTarget {
  /**
   * @param {HTMLElement} wrap  the area the stage must fit in
   * @param {HTMLElement} stage container holding both panes
   * @param {HTMLCanvasElement} picture
   */
  constructor(wrap, stage, picture) {
    super();
    this.wrap = wrap;
    this.stage = stage;
    this.picture = picture;
    this.pctx = picture.getContext("2d");
    this.pctx.fillStyle = "#ffffff";
    this.pctx.fillRect(0, 0, picture.width, picture.height);
    this.mode = "auto";
    this.userChose = false;   // a stored/explicit choice beats the server default
    this.hasPicture = false;
    this.drawing = false;
    this.pictures = 0;        // frames rendered, exposed as data-pictures for tests
    this._token = 0;          // newest decode wins

    this._portrait = matchMedia("(orientation: portrait)");
    const stored = readStoredMode();
    if (stored) {
      this.userChose = true;
      this._apply(stored);
    } else {
      this._apply("auto");
    }

    const relayout = () => this.layout();
    if ("ResizeObserver" in window) new ResizeObserver(relayout).observe(wrap);
    window.addEventListener("resize", relayout);
    this.layout();
  }

  /** Server default (ready / styles messages); ignored once the kid picked a mode. */
  setDefaultMode(mode) {
    if (!this.userChose && MODES.includes(mode)) this._apply(mode);
  }

  /** Kid tapped a mode button. */
  setMode(mode) {
    if (!MODES.includes(mode)) return;
    this.userChose = true;
    storeMode(mode);
    this._apply(mode);
  }

  strokeStart() {
    this.drawing = true;
    // Instant switch back to the sketch: suspend the fade for one frame.
    if (this.stage.classList.contains("show-picture")) {
      this.stage.classList.add("no-fade");
      this.stage.classList.remove("show-picture");
      void this.stage.offsetWidth; // flush styles so the removal is not animated
      requestAnimationFrame(() => this.stage.classList.remove("no-fade"));
    }
  }

  strokeEnd() {
    this.drawing = false;
  }

  /**
   * Draw a JPEG frame from the server.
   * @param {ArrayBuffer} jpeg
   * @param {boolean} final made from the newest final sketch
   */
  async showPicture(jpeg, final) {
    const token = ++this._token;
    const blob = new Blob([jpeg], { type: "image/jpeg" });
    let img = null;
    let cleanup = () => {};
    try {
      if ("createImageBitmap" in window) {
        img = await createImageBitmap(blob);
        cleanup = () => img.close && img.close();
      } else {
        img = await loadImage(blob);
        cleanup = () => URL.revokeObjectURL(img.src);
      }
    } catch {
      try {
        img = await loadImage(blob); // createImageBitmap(blob) can fail on older Safari
        cleanup = () => URL.revokeObjectURL(img.src);
      } catch {
        return;
      }
    }
    if (token !== this._token) { cleanup(); return; } // a newer frame overtook this one
    const { width, height } = this.picture;
    this.pctx.drawImage(img, 0, 0, width, height);
    cleanup();
    this.hasPicture = true;
    this.pictures += 1;
    this.stage.dataset.pictures = String(this.pictures);
    if (final && !this.drawing) this.stage.classList.add("show-picture");
    this.dispatchEvent(new Event("picture"));
  }

  /** Fit the stage into the wrap: one square, or two squares side by side / stacked. */
  layout() {
    const w = this.wrap.clientWidth;
    const h = this.wrap.clientHeight;
    if (!w || !h) return;
    let side;
    if (this.mode === "split") {
      side = this._portrait.matches ? Math.min(w, (h - GAP) / 2) : Math.min((w - GAP) / 2, h);
    } else {
      side = Math.min(w, h);
    }
    side -= RING; // leave room for the white frame drawn around each pane
    this.stage.style.setProperty("--side", `${Math.max(64, Math.floor(side))}px`);
    this.dispatchEvent(new Event("layout"));
  }

  _apply(mode) {
    this.mode = mode;
    for (const m of MODES) this.stage.classList.toggle(`mode-${m}`, m === mode);
    if (this.wrap) this.layout();
    this.dispatchEvent(new CustomEvent("mode", { detail: { mode } }));
  }
}

function loadImage(blob) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => { URL.revokeObjectURL(img.src); reject(new Error("bad image")); };
    img.src = URL.createObjectURL(blob);
  });
}
