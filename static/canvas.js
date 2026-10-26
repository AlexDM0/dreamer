// Drawing surface, in two layers:
//   - the background layer: one solid colour (`background`), shown behind the strokes by
//     CSS (the pane's background). It can change any time without touching the strokes.
//   - the stroke layer: a transparent backing canvas at the drawing's own size (what is
//     sent to the server, e.g. 640x448). The eraser cuts holes in it (destination-out).
// Every stroke is mirrored onto the visible canvas, whose pixel size follows its CSS size
// x devicePixelRatio so lines stay crisp on Retina screens.
//
// A drawing has an orientation ("landscape" | "portrait"), fixed when it is started. Its
// aspect never changes afterwards: rotating the iPad only re-fits it on screen.
//
// Events: "strokestart", "strokeend", "strokecancel" (a finger stroke was undone because
// it turned into a pinch: the drawing is back as it was, nothing new to save), "drawing"
// (detail: {width, height, background, orientation}) when the size or the background
// changes, and "pinchstart" / "pinch" / "pinchend" (see _startPinch).

export const DEFAULT_SIZES = { landscape: [640, 448], portrait: [448, 640] };
export const DEFAULT_BACKGROUND = "#000000";
const COLOR_RE = /^#[0-9a-f]{6}$/i;
const MAX_DISPLAY_PX = 2048;   // cap the visible canvas's backing store (longest side)
const PEN_GRACE_MS = 800;      // ignore touches this long after the pencil lifts (palm)
const KEY_OUT_DISTANCE = 40;   // legacy opaque sketches: this close to the paper = no ink

/** The orientation a new drawing gets on this screen right now. */
export function deviceOrientation() {
  return window.innerWidth >= window.innerHeight ? "landscape" : "portrait";
}

/** A valid "#rrggbb" (lower case), or the fallback. */
export function normaliseColor(color, fallback = DEFAULT_BACKGROUND) {
  return typeof color === "string" && COLOR_RE.test(color) ? color.toLowerCase() : fallback;
}

export class DrawingCanvas extends EventTarget {
  /** @param {HTMLCanvasElement} el */
  constructor(el) {
    super();
    this.el = el;
    this.sizes = { ...DEFAULT_SIZES };
    this.orientation = deviceOrientation();
    this.background = DEFAULT_BACKGROUND;
    this.src = document.createElement("canvas"); // the stroke layer, source of truth
    this.sctx = this.src.getContext("2d");
    this._setSourceSize();
    this.ctx = el.getContext("2d");
    this.color = "#ffffff";
    this.width = 12;         // brush width in drawing pixels
    this.eraser = false;
    this._dirty = false;
    this._active = null;     // pointerId of the stroke in progress
    this._activeType = null; // its pointerType
    this._last = null;       // last point {x, y} in drawing pixels
    this._penSeen = false;
    this._penDown = false;
    this._penUpAt = 0;
    this._touchSnapshot = null; // drawing before a touch stroke, undone if a pen takes over
    this._snapshotCanvas = document.createElement("canvas"); // reused for _touchSnapshot
    // Whether the stroke layer has ink: "none", "some" or "unknown" (the eraser was used,
    // or a picture was loaded). isBlank only scans pixels when it is "unknown".
    this._ink = "none";
    // Pinch to resize the brush: fingers on the canvas (pointerId -> client point), and the
    // pinch in progress ({distance} at its start), or null.
    this._fingers = new Map();
    this._pinch = null;
    this._bind();
    this._fitDisplay(true);
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => this._fitDisplay()).observe(el);
    }
    window.addEventListener("resize", () => this._fitDisplay());
  }

  get drawingWidth() { return this.src.width; }
  get drawingHeight() { return this.src.height; }

  isDirty() { return this._dirty; }
  clearDirty() { this._dirty = false; }

  /** What the server needs to know about this drawing ({"t":"canvas"}). */
  info() { return { background: this.background, orientation: this.orientation }; }

  /** @returns {Promise<Blob>} PNG of the stroke layer only (transparent where empty). */
  toBlob() {
    return new Promise((resolve, reject) => {
      this.src.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("toBlob failed"))), "image/png");
    });
  }

  /** Server sizes per orientation (the ready message's `canvas`). Strokes already drawn
   * are scaled along if the current drawing's size changes. */
  setSizes(sizes) {
    if (!sizes || typeof sizes !== "object") return;
    for (const name of ["landscape", "portrait"]) {
      const dims = sizes[name];
      if (Array.isArray(dims) && dims.length === 2
          && dims.every((n) => Number.isInteger(n) && n > 0 && n <= 4096)) {
        this.sizes[name] = [dims[0], dims[1]];
      }
    }
    const [width, height] = this.sizes[this.orientation];
    if (width === this.src.width && height === this.src.height) return;
    const old = document.createElement("canvas");
    old.width = this.src.width;
    old.height = this.src.height;
    old.getContext("2d").drawImage(this.src, 0, 0);
    this._endStroke(false);
    this._setSourceSize();
    this.sctx.drawImage(old, 0, 0, this.src.width, this.src.height);
    if (this._ink === "some") this._ink = "unknown"; // scaling down can lose thin lines
    this._changed();
  }

  /** Start a new, empty drawing with this orientation and background. */
  startDrawing({ orientation = deviceOrientation(), background = DEFAULT_BACKGROUND } = {}) {
    this._endStroke(false);
    this.orientation = orientation === "portrait" ? "portrait" : "landscape";
    this.background = normaliseColor(background);
    this._setSourceSize(); // also empties the stroke layer
    this._ink = "none";
    this._changed();
  }

  /** Change the background layer; the strokes stay as they are. */
  setBackground(color) {
    const background = normaliseColor(color, this.background);
    if (background === this.background) return;
    this.background = background;
    this._dirty = true;
    this._emitDrawing();
  }

  /** Empty the stroke layer (the background stays). */
  clear() {
    this._endStroke(false);
    this.sctx.clearRect(0, 0, this.src.width, this.src.height);
    this._ink = "none";
    this._repaint();
    this._dirty = true;
  }

  /**
   * Replace the drawing with a saved sketch (a transparent stroke layer). An orientation
   * of "square" or none (artworks from before layers) is taken from the image's shape; a
   * square image is letterboxed into a landscape drawing. A fully opaque image (an old
   * sketch on white paper) gets its paper keyed out, so its strokes become a layer.
   * @param {CanvasImageSource & {width:number, height:number}} img
   * @param {{orientation?: string, background?: string}} meta
   */
  loadDrawing(img, { orientation, background } = {}) {
    let next = orientation;
    if (next !== "landscape" && next !== "portrait") {
      next = img.height > img.width ? "portrait" : "landscape";
    }
    this.startDrawing({ orientation: next, background: normaliseColor(background, "#ffffff") });
    const fullWidth = this.src.width;
    const fullHeight = this.src.height;
    const scale = Math.min(fullWidth / img.width, fullHeight / img.height); // contain
    const drawWidth = Math.max(1, Math.round(img.width * scale));
    const drawHeight = Math.max(1, Math.round(img.height * scale));
    const left = Math.round((fullWidth - drawWidth) / 2);
    const top = Math.round((fullHeight - drawHeight) / 2);
    this.sctx.drawImage(img, left, top, drawWidth, drawHeight);
    this._keyOutIfOpaque(left, top, drawWidth, drawHeight);
    this._ink = "unknown";
    this._repaint();
    this._dirty = true;
  }

  /** True when the stroke layer is empty (no visible ink). Scans the pixels only when the
   * ink tracking cannot tell (after erasing or loading a picture), and remembers the answer. */
  isBlank() {
    if (this._ink !== "unknown") return this._ink === "none";
    const data = this.sctx.getImageData(0, 0, this.src.width, this.src.height).data;
    this._ink = "none";
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] > 8) { this._ink = "some"; break; }
    }
    return this._ink === "none";
  }

  setColor(color) { this.color = color; }
  setEraser(on) { this.eraser = !!on; }
  setWidth(width) { this.width = Math.max(1, Number(width) || 1); }

  /** CSS pixels per drawing pixel, for previews. */
  displayScale() {
    const rect = this.el.getBoundingClientRect();
    return rect.width ? rect.width / this.src.width : 1;
  }

  // ---- internals ----

  _setSourceSize() {
    const [width, height] = this.sizes[this.orientation];
    this.src.width = width;   // resizing also clears it
    this.src.height = height;
  }

  /** Size or content was replaced: refit the display and tell the page. */
  _changed() {
    this._dirty = true;
    this._fitDisplay(true);
    this._emitDrawing();
  }

  _emitDrawing() {
    this.dispatchEvent(new CustomEvent("drawing", {
      detail: {
        width: this.src.width, height: this.src.height,
        background: this.background, orientation: this.orientation,
      },
    }));
  }

  /** Old sketches are opaque (strokes on white paper): make the paper transparent. */
  _keyOutIfOpaque(left, top, width, height) {
    const image = this.sctx.getImageData(left, top, width, height);
    const data = image.data;
    for (let i = 3; i < data.length; i += 4) if (data[i] < 250) return; // already a layer
    const bg = parseInt(this.background.slice(1), 16);
    const bgRed = bg >> 16;
    const bgGreen = (bg >> 8) & 255;
    const bgBlue = bg & 255;
    for (let i = 0; i < data.length; i += 4) {
      const distance = Math.max(Math.abs(data[i] - bgRed), Math.abs(data[i + 1] - bgGreen),
                                Math.abs(data[i + 2] - bgBlue));
      if (distance <= KEY_OUT_DISTANCE) data[i + 3] = 0;
    }
    this.sctx.putImageData(image, left, top);
  }

  /** Size the visible canvas to its CSS box x DPR and repaint it from the stroke layer. */
  _fitDisplay(force = false) {
    const rect = this.el.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      if (force) {
        this.el.width = this.src.width;
        this.el.height = this.src.height;
        this._repaint();
      }
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    let pxWidth = rect.width * dpr;
    let pxHeight = rect.height * dpr;
    const cap = MAX_DISPLAY_PX / Math.max(pxWidth, pxHeight);
    if (cap < 1) { pxWidth *= cap; pxHeight *= cap; }
    pxWidth = Math.max(1, Math.round(pxWidth));
    pxHeight = Math.max(1, Math.round(pxHeight));
    if (!force && pxWidth === this.el.width && pxHeight === this.el.height) return;
    this.el.width = pxWidth;
    this.el.height = pxHeight;
    this._repaint();
  }

  _repaint() {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, this.el.width, this.el.height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(this.src, 0, 0, this.el.width, this.el.height);
    // From now on draw in drawing pixels; the transform scales to display pixels.
    ctx.setTransform(this.el.width / this.src.width, 0, 0, this.el.height / this.src.height, 0, 0);
  }

  /** Run a drawing function on the stroke layer and the visible canvas alike: the pen
   * paints, the eraser cuts the strokes away (revealing the background). */
  _both(fn) {
    if (this.eraser) {
      if (this._ink === "some") this._ink = "unknown"; // it may have erased the last ink
    } else {
      this._ink = "some";
    }
    for (const ctx of [this.sctx, this.ctx]) {
      ctx.save();
      ctx.globalCompositeOperation = this.eraser ? "destination-out" : "source-over";
      fn(ctx);
      ctx.restore();
    }
  }

  _bind() {
    const el = this.el;
    el.addEventListener("pointerdown", (e) => this._down(e));
    el.addEventListener("pointermove", (e) => this._move(e));
    el.addEventListener("pointerup", (e) => this._up(e));
    el.addEventListener("pointercancel", (e) => this._up(e));
    // iOS: stop the page from scrolling / magnifying while drawing.
    el.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });
    el.addEventListener("touchmove", (e) => e.preventDefault(), { passive: false });
    el.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  /** Pointer position in drawing pixels (the CSS box can be any size). */
  _point(e) {
    const rect = this.el.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * this.src.width,
      y: ((e.clientY - rect.top) / rect.height) * this.src.height,
    };
  }

  /** Distance between the first two fingers and their midpoint, in CSS pixels. */
  _fingerSpan() {
    const [first, second] = [...this._fingers.values()];
    return { distance: Math.hypot(second.x - first.x, second.y - first.y),
             x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
  }

  /** A second finger came down: undo the stroke the first finger started and pinch. Emits
   * "pinchstart", then "pinch" with {scale, x, y} (scale vs. the start), then "pinchend". */
  _startPinch() {
    if (this._activeType === "pen") return; // never resize the brush mid-line
    if (this._active !== null && this._activeType === "touch") {
      this._restoreTouchSnapshot();
      this._endStroke(false);
      // The drawing is back as it was: the page re-sends it, but nothing new to save.
      this.dispatchEvent(new Event("strokecancel"));
    }
    const span = this._fingerSpan();
    this._pinch = { distance: Math.max(1, span.distance) };
    this.dispatchEvent(new CustomEvent("pinchstart", { detail: { x: span.x, y: span.y } }));
  }

  /** A pencil came down: forget the fingers and end a pinch in progress. */
  _dropFingers() {
    this._fingers.clear();
    if (!this._pinch) return;
    this._pinch = null;
    this.dispatchEvent(new Event("pinchend"));
  }

  _down(e) {
    if (e.pointerType === "touch" && this._penSeen
        && (this._penDown || Date.now() - this._penUpAt < PEN_GRACE_MS)) {
      return; // palm rejection: the hand rests on the glass while / just after writing
    }
    if (e.pointerType === "touch") {
      this._fingers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this._pinch) { e.preventDefault(); return; } // a third finger changes nothing
      if (this._fingers.size === 2) {
        e.preventDefault();
        this._startPinch();
        return;
      }
    }
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (e.pointerType === "pen") {
      this._penSeen = true;
      this._penDown = true;
      this._dropFingers();
      // Pen wins over a palm/finger stroke that started first: undo what it drew.
      if (this._active !== null && this._activeType === "touch") {
        this._restoreTouchSnapshot();
        this._endStroke(false);
      }
    }
    if (this._active !== null) return; // one stroke at a time; ignore extra fingers
    // Remember the drawing before a finger stroke, so it can be undone when a pencil takes
    // over (palm) or a second finger turns it into a pinch.
    this._touchSnapshot = e.pointerType === "touch" ? this._takeSnapshot() : null;

    e.preventDefault();
    try { this.el.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
    this._active = e.pointerId;
    this._activeType = e.pointerType;
    const point = this._point(e);
    this._last = point;
    this._dot(point);
    this.dispatchEvent(new Event("strokestart"));
  }

  _move(e) {
    if (this._fingers.has(e.pointerId)) {
      this._fingers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this._pinch && this._fingers.size >= 2) {
        e.preventDefault();
        const span = this._fingerSpan();
        this.dispatchEvent(new CustomEvent("pinch", {
          detail: { scale: span.distance / this._pinch.distance, x: span.x, y: span.y } }));
        return;
      }
    }
    if (e.pointerId !== this._active) return;
    e.preventDefault();
    const events = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [];
    const list = events.length ? events : [e];
    for (const ev of list) {
      const point = this._point(ev);
      this._segment(this._last, point);
      this._last = point;
    }
  }

  _up(e) {
    if (this._fingers.delete(e.pointerId) && this._pinch) {
      // The pinch ends when the fingers lift; a finger left on the glass does not draw.
      if (this._fingers.size === 0) {
        this._pinch = null;
        this.dispatchEvent(new Event("pinchend"));
      }
      return;
    }
    if (e.pointerType === "pen") {
      this._penDown = false;
      this._penUpAt = Date.now();
    }
    if (e.pointerId !== this._active) return;
    try { this.el.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    this._endStroke(true);
  }

  /** Copy the stroke layer into the reusable offscreen canvas (no pixel read-back). */
  _takeSnapshot() {
    const snapshot = this._snapshotCanvas;
    if (snapshot.width !== this.src.width || snapshot.height !== this.src.height) {
      snapshot.width = this.src.width; // resizing also clears it
      snapshot.height = this.src.height;
    } else {
      snapshot.getContext("2d").clearRect(0, 0, snapshot.width, snapshot.height);
    }
    snapshot.getContext("2d").drawImage(this.src, 0, 0);
    return snapshot;
  }

  _restoreTouchSnapshot() {
    if (!this._touchSnapshot) return;
    this.sctx.save();
    this.sctx.globalCompositeOperation = "source-over";
    this.sctx.clearRect(0, 0, this.src.width, this.src.height);
    this.sctx.drawImage(this._touchSnapshot, 0, 0);
    this.sctx.restore();
    this._touchSnapshot = null;
    this._ink = "unknown"; // the undone stroke may have been the only ink
    this._repaint();
    this._dirty = true;
  }

  _endStroke(emit) {
    this._touchSnapshot = null;
    if (this._active === null) return;
    this._active = null;
    this._activeType = null;
    this._last = null;
    if (emit) this.dispatchEvent(new Event("strokeend"));
  }

  _dot(point) {
    this._both((ctx) => {
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(point.x, point.y, this.width / 2, 0, Math.PI * 2);
      ctx.fill();
    });
    this._dirty = true;
  }

  _segment(from, to) {
    this._both((ctx) => {
      ctx.strokeStyle = this.color;
      ctx.lineWidth = this.width;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    });
    this._dirty = true;
  }
}
