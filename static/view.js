// Stage: the sketch canvas stacked over the picture canvas, plus the view modes.
//
// Modes (container class): mode-auto | mode-sketch | mode-picture | mode-split.
// Auto: the sketch shows instantly when a stroke starts; after the stroke ends it stays
// until the PICTURE made from the final frame (net "final" flag, i.e. seq >= finalSeq)
// arrives, then the picture cross-fades in (CSS opacity transition on .show-picture).
//
// The panes always have the drawing's own aspect (setDrawing): they are fitted (contain)
// into the free area and never stretched. Split shows them side by side or stacked,
// whichever gives the bigger panes (class split-stacked). The drawing's background colour
// is the panes' background (--drawing-bg), behind the transparent stroke layer.

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
    this.aspect = picture.width / picture.height;
    this.background = "#000000";
    this._blankPicture();
    this.mode = "auto";
    this.userChose = false;   // a stored/explicit choice beats the server default
    this.hasPicture = false;
    this.drawing = false;
    this.pictures = 0;        // frames rendered, exposed as data-pictures for tests
    this._token = 0;          // newest decode wins

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

  /**
   * The drawing's size and paper changed. A new size starts an empty picture at that size
   * (pictures come back at the drawing's size); a new colour only recolours the frame.
   * @param {{width:number, height:number, background:string}} drawing
   */
  setDrawing({ width, height, background }) {
    this.background = background;
    this.stage.style.setProperty("--drawing-bg", background);
    // Compared with the drawing's own size: the picture canvas may hold a bigger
    // (upscaled) picture of the same drawing.
    const previous = this._drawingDims || [this.picture.width, this.picture.height];
    this._drawingDims = [width, height];
    if (width !== previous[0] || height !== previous[1]) {
      this.picture.width = width;
      this.picture.height = height;
      this.aspect = width / height;
      this._token += 1; // a picture still decoding belongs to the old drawing
      this.hasPicture = false;
      this.stage.classList.remove("show-picture");
      this._blankPicture();
      this.layout();
    } else if (!this.hasPicture) {
      this._blankPicture();
    }
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
    // A picture of the drawing's shape at another resolution (an upscaled refinement, or a
    // new resolution setting) is shown at its own resolution, not squeezed into the old one.
    if (Math.abs(img.width / img.height - this.aspect) < 0.01 &&
        (img.width !== this.picture.width || img.height !== this.picture.height)) {
      this.picture.width = img.width;
      this.picture.height = img.height;
    }
    // Contain: a picture of another shape is letterboxed on the paper, never stretched.
    const { width, height } = this.picture;
    const scale = Math.min(width / img.width, height / img.height);
    const drawWidth = img.width * scale;
    const drawHeight = img.height * scale;
    if (Math.abs(drawWidth - width) > 0.5 || Math.abs(drawHeight - height) > 0.5) this._blankPicture();
    this.pctx.drawImage(img, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
    cleanup();
    this.hasPicture = true;
    this.pictures += 1;
    this.stage.dataset.pictures = String(this.pictures);
    if (final && !this.drawing) this.stage.classList.add("show-picture");
    this.dispatchEvent(new Event("picture"));
  }

  /** Fit the stage into the wrap at the drawing's aspect: one pane, or two panes side by
   * side / stacked (whichever is bigger). */
  layout() {
    const wrapWidth = this.wrap.clientWidth;
    const wrapHeight = this.wrap.clientHeight;
    if (!wrapWidth || !wrapHeight) return;
    const aspect = this.aspect;
    // Largest pane of this aspect inside a box, leaving room for the frame and shadow.
    const fit = (boxWidth, boxHeight) => {
      const paneWidth = Math.max(0, Math.min(boxWidth - RING, (boxHeight - RING) * aspect));
      return [paneWidth, paneWidth / aspect];
    };
    let pane;
    let stacked = false;
    if (this.mode === "split") {
      const sideBySide = fit((wrapWidth - GAP) / 2, wrapHeight);
      const onTop = fit(wrapWidth, (wrapHeight - GAP) / 2);
      stacked = onTop[0] > sideBySide[0];
      pane = stacked ? onTop : sideBySide;
    } else {
      pane = fit(wrapWidth, wrapHeight);
    }
    const paneWidth = Math.max(64, Math.floor(pane[0]));
    const paneHeight = Math.max(Math.round(64 / aspect), Math.floor(paneWidth / aspect));
    this.stage.style.setProperty("--pw", `${paneWidth}px`);
    this.stage.style.setProperty("--ph", `${paneHeight}px`);
    this.stage.classList.toggle("split-stacked", stacked);
    this.dispatchEvent(new Event("layout"));
  }

  _blankPicture() {
    this.pctx.fillStyle = this.background;
    this.pctx.fillRect(0, 0, this.picture.width, this.picture.height);
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
