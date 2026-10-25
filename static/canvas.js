// Drawing surface. The drawing itself lives on a fixed 512x512 backing canvas (what is
// sent to the server); every stroke is mirrored onto the visible canvas, whose pixel size
// follows its CSS size x devicePixelRatio so lines stay crisp on Retina screens.
// Emits "strokestart" and "strokeend" events (EventTarget).

const SIZE = 512;
const MAX_DISPLAY_PX = 2048;   // cap the visible canvas's backing store
const PEN_GRACE_MS = 800;      // ignore touches this long after the pencil lifts (palm)

export class DrawingCanvas extends EventTarget {
  /** @param {HTMLCanvasElement} el */
  constructor(el) {
    super();
    this.el = el;
    this.src = document.createElement("canvas"); // 512x512 source of truth
    this.src.width = SIZE;
    this.src.height = SIZE;
    this.sctx = this.src.getContext("2d");
    el.width = SIZE;
    el.height = SIZE;
    this.ctx = el.getContext("2d");
    this.color = "#2d3561";
    this.width = 12;
    this._dirty = false;
    this._active = null;     // pointerId of the stroke in progress
    this._activeType = null; // its pointerType
    this._last = null;       // last point {x, y} in backing pixels
    this._penSeen = false;
    this._penDown = false;
    this._penUpAt = 0;
    this._touchSnapshot = null; // drawing before a touch stroke, undone if a pen takes over
    this._fillWhite();
    this._bind();
    this._fitDisplay();
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => this._fitDisplay()).observe(el);
    } else {
      window.addEventListener("resize", () => this._fitDisplay());
    }
  }

  isDirty() { return this._dirty; }
  clearDirty() { this._dirty = false; }

  /** @returns {Promise<Blob>} PNG of the current drawing. */
  toBlob() {
    return new Promise((resolve, reject) => {
      this.src.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png");
    });
  }

  clear() {
    this._endStroke(false);
    this._fillWhite();
    this._dirty = true;
  }

  setColor(color) { this.color = color; }
  setWidth(w) { this.width = Math.max(1, Number(w) || 1); }

  /** CSS pixels per backing pixel, for previews. */
  displayScale() {
    const r = this.el.getBoundingClientRect();
    return r.width ? r.width / SIZE : 1;
  }

  // ---- internals ----

  /** Size the visible canvas to its CSS box x DPR and repaint it from the source. */
  _fitDisplay() {
    const r = this.el.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    const px = Math.max(1, Math.min(MAX_DISPLAY_PX, Math.round(r.width * dpr)));
    if (px === this.el.width) return;
    this.el.width = px;
    this.el.height = px;
    this._repaint();
  }

  _repaint() {
    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.imageSmoothingQuality = "high";
    c.drawImage(this.src, 0, 0, this.el.width, this.el.height);
    // From now on draw in 512-space; the transform scales to display pixels.
    const k = this.el.width / SIZE;
    c.setTransform(k, 0, 0, k, 0, 0);
  }

  /** Run a drawing function on the source and the visible canvas alike. */
  _both(fn) {
    fn(this.sctx);
    fn(this.ctx);
  }

  _fillWhite() {
    this._both((c) => {
      c.save();
      c.globalCompositeOperation = "source-over";
      c.fillStyle = "#ffffff";
      c.fillRect(0, 0, SIZE, SIZE);
      c.restore();
    });
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

  _point(e) {
    const r = this.el.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * SIZE,
      y: ((e.clientY - r.top) / r.height) * SIZE,
    };
  }

  _down(e) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (e.pointerType === "pen") {
      this._penSeen = true;
      this._penDown = true;
      // Pen wins over a palm/finger stroke that started first: undo what it drew.
      if (this._active !== null && this._activeType === "touch") {
        this._restoreTouchSnapshot();
        this._endStroke(false);
      }
    } else if (e.pointerType === "touch" && this._penSeen
               && (this._penDown || Date.now() - this._penUpAt < PEN_GRACE_MS)) {
      return; // palm rejection: the hand rests on the glass while / just after writing
    }
    if (this._active !== null) return; // one stroke at a time; ignore extra fingers
    // Once a pencil is in use, remember the drawing so a palm stroke can be undone.
    this._touchSnapshot = e.pointerType === "touch" && this._penSeen
      ? this.sctx.getImageData(0, 0, SIZE, SIZE) : null;

    e.preventDefault();
    try { this.el.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
    this._active = e.pointerId;
    this._activeType = e.pointerType;
    const p = this._point(e);
    this._last = p;
    this._dot(p);
    this.dispatchEvent(new Event("strokestart"));
  }

  _move(e) {
    if (e.pointerId !== this._active) return;
    e.preventDefault();
    const events = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [];
    const list = events.length ? events : [e];
    for (const ev of list) {
      const p = this._point(ev);
      this._segment(this._last, p);
      this._last = p;
    }
  }

  _up(e) {
    if (e.pointerType === "pen") {
      this._penDown = false;
      this._penUpAt = Date.now();
    }
    if (e.pointerId !== this._active) return;
    try { this.el.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    this._endStroke(true);
  }

  _restoreTouchSnapshot() {
    if (!this._touchSnapshot) return;
    this.sctx.putImageData(this._touchSnapshot, 0, 0);
    this._touchSnapshot = null;
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

  _dot(p) {
    this._both((c) => {
      c.fillStyle = this.color;
      c.beginPath();
      c.arc(p.x, p.y, this.width / 2, 0, Math.PI * 2);
      c.fill();
    });
    this._dirty = true;
  }

  _segment(a, b) {
    this._both((c) => {
      c.strokeStyle = this.color;
      c.lineWidth = this.width;
      c.lineCap = "round";
      c.lineJoin = "round";
      c.beginPath();
      c.moveTo(a.x, a.y);
      c.lineTo(b.x, b.y);
      c.stroke();
    });
    this._dirty = true;
  }
}
