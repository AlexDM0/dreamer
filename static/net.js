// Kid WebSocket client (see SPEC "Protocol").
//
// Events (EventTarget, payload in event.detail):
//   "state"    {state: "connecting"|"open"|"off"|"replaced"}
//   "ready"    the server's ready/styles message (topics, styles, topic, style, view_mode, ...)
//   "status"   {state: "busy"|"ready"|"error", msg}
//   "picture"  {seq, jpeg: ArrayBuffer, final: boolean}
//   "saved"    {artwork}            a Save went through (SPEC-addendum-gallery)
//   "continue" {id, sketch, topic, style, background, orientation}  keep drawing this one
//   "gallery"  {}                   saved pictures changed (e.g. the parent deleted one)
//
// Flow control: at most one SKETCH in flight. A frame is in flight until its PICTURE
// (any seq >= it) arrives or 2 s pass. sendNow() asks for a "final" frame of the current
// drawing; if a frame is in flight it is queued (exactly one pending final) and sent as
// soon as the slot frees. `finalSeq` is the seq of the newest final frame.
//
// Drawing layers: sketches are the transparent stroke layer only. The paper colour and
// orientation go separately as {"t":"canvas"} (sendCanvas): right after "ready", on a new
// drawing, and whenever the page changes them.

const SKETCH = 0x01;
const PICTURE = 0x02;
const HEADER = 5;
const INFLIGHT_TIMEOUT_MS = 2000;
const PING_MS = 10000;
const DEAD_MS = 25000;       // no traffic for this long -> assume the socket is dead
const BACKOFF_MIN_MS = 500;
const BACKOFF_MAX_MS = 5000;
const CLOSE_REPLACED = 4000; // server close code: a newer kid socket took over
const FINAL_RETRIES = 3;     // resend an unanswered final frame at most this many times

export class Net extends EventTarget {
  /**
   * @param {{isDirty(): boolean, clearDirty(): void, toBlob(): Promise<Blob>,
   *          info(): {background: string, orientation: string}}} canvas
   * @param {string} [url] defaults to ws(s)://<host>/ws
   */
  constructor(canvas, url) {
    super();
    this.canvas = canvas;
    this.url = url || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    this.ws = null;
    this.seq = 0;              // last seq sent; keeps counting across reconnects
    this.finalSeq = 0;         // seq of the latest "final" frame (sent after a stroke ends)
    this.pendingFinal = false; // a final frame is waiting for the in-flight slot
    this.floorSeq = 0;         // pictures older than this are stale (e.g. made before a clear)
    this.lastPictureSeq = 0;
    this._inflight = null;     // {seq, timer}
    this._sending = false;     // encoding a PNG right now
    this._backoff = BACKOFF_MIN_MS;
    this._retryTimer = null;
    this._pingTimer = null;
    this._lastRx = 0;
    this._stopped = false;
    this._ready = false;       // got "ready" on the current socket
    this._finalRetries = 0;    // resends of the current unanswered final frame
    // The saved artwork the canvas holds (Save updates it), or null for a new drawing.
    // It belongs to this page: sent with every hello so a reconnect keeps it, while a
    // freshly loaded page starts with null and can never overwrite an artwork by mistake.
    this.continuingId = null;
    this.state = "off";
    this._loop = this._loop.bind(this);
  }

  /** Open the socket (and keep it open). */
  start() {
    this._stopped = false;
    this._connect();
    requestAnimationFrame(this._loop);
  }

  get connected() { return !!this.ws && this.ws.readyState === WebSocket.OPEN; }

  /** Connected and greeted: sketches may be sent. */
  get live() { return this.connected && this._ready; }

  /** True while the newest final frame has not come back yet. */
  get awaitingFinal() { return this.pendingFinal || this.lastPictureSeq < this.finalSeq; }

  selectTopic(id) { this._sendJson({ t: "topic", id }); }
  selectStyle(id) { this._sendJson({ t: "style", id }); }

  /** Tell the server to drop queued work, then send the (now blank) canvas as a final frame. */
  clear() {
    this.floorSeq = this.seq + 1;
    this.continuingId = null;
    this._sendJson({ t: "clear" });
    this.sendNow();
  }

  /** Tell the server the drawing's background colour and orientation. */
  sendCanvas() {
    const { background, orientation } = this.canvas.info();
    return this._sendJson({ t: "canvas", background, orientation });
  }

  /** Save what is on screen as an artwork; the server answers "saved" or an error status. */
  save() { return this._sendJson({ t: "save" }); }

  /** Start a fresh drawing: the server forgets the draft and what was being continued. */
  newDrawing() {
    this.floorSeq = this.seq + 1;
    this.continuingId = null;
    this._sendJson({ t: "new" });
    this.sendCanvas();
    this.sendNow();
  }

  /** Ask to keep drawing a saved artwork; the reply arrives as a "continue" event. */
  continueArtwork(id) { return this._sendJson({ t: "continue", id }); }

  /** The saved artwork's sketch is on the canvas now: Save updates that artwork. Call it
   * before sending the restored sketch (the server starts a new drawing on it). */
  continuing(id) {
    this.continuingId = id;
    this._sendJson({ t: "continuing", id });
  }

  /** The draft is back on the canvas; it continues `source` (a saved artwork) or nothing. */
  restored(draftId, source) {
    this.continuingId = source || null;
    this._sendJson({ t: "restored", id: draftId });
  }

  /** Pictures made from sketches sent so far are stale (the canvas content was replaced). */
  dropOlderPictures() { this.floorSeq = this.seq + 1; }

  /** Send the current drawing as a final frame, now or as soon as the in-flight slot frees. */
  sendNow() {
    this._finalRetries = 0; // a fresh request, not a retry
    if (this._inflight || this._sending || !this.live) {
      this.pendingFinal = true;
      return;
    }
    this._sendSketch(true);
  }

  // ---- connection ----

  _setState(state) {
    if (this.state === state) return;
    this.state = state;
    this._emit("state", { state });
  }

  _connect() {
    clearTimeout(this._retryTimer);
    if (this._stopped) return;
    this._setState("connecting");
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this._scheduleRetry();
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this._backoff = BACKOFF_MIN_MS;
      this._lastRx = Date.now();
      this._setState("open");
      this._sendJson({ t: "hello", v: 1, continuing: this.continuingId });
      clearInterval(this._pingTimer);
      this._pingTimer = setInterval(() => this._ping(), PING_MS);
    };
    ws.onmessage = (ev) => {
      this._lastRx = Date.now();
      if (typeof ev.data === "string") this._onText(ev.data);
      else this._onBinary(ev.data);
    };
    ws.onclose = (ev) => {
      if (this.ws !== ws) return;
      this._teardown();
      // The close code is a backstop in case the {"t":"replaced"} message was lost.
      if (ev.code === CLOSE_REPLACED) this._setState("replaced");
      if (this.state === "replaced") return; // another screen took over; wait for the kid
      this._setState("off");
      this._scheduleRetry();
    };
    ws.onerror = () => { /* onclose follows */ };
  }

  /** Reconnect after being replaced by another screen. */
  takeOver() {
    this._setState("connecting");
    this._connect();
  }

  _teardown() {
    clearInterval(this._pingTimer);
    this._pingTimer = null;
    this._clearInflight();
    this._ready = false;
    this.ws = null;
  }

  _scheduleRetry() {
    clearTimeout(this._retryTimer);
    this._retryTimer = setTimeout(() => this._connect(), this._backoff);
    this._backoff = Math.min(this._backoff * 2, BACKOFF_MAX_MS);
  }

  _ping() {
    if (!this.connected) return;
    if (Date.now() - this._lastRx > DEAD_MS) {
      this.ws.close(); // half-open socket (e.g. iPad slept); onclose reconnects
      return;
    }
    this._sendJson({ t: "ping" });
  }

  _sendJson(obj) {
    if (!this.connected) return false;
    this.ws.send(JSON.stringify(obj));
    return true;
  }

  // ---- incoming ----

  _onText(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    if (!msg || typeof msg.t !== "string") return;
    switch (msg.t) {
      case "ready":
      case "styles":
        this._emit("ready", msg); // the page applies msg.canvas sizes before we go on
        if (msg.t === "ready") {
          this._ready = true;
          this.sendCanvas(); // before the first sketch on this socket
          this.sendNow();    // refresh the picture after (re)connecting
        }
        break;
      case "status":
        this._emit("status", { state: msg.state, msg: msg.msg || "" });
        break;
      case "replaced":
        this._setState("replaced");
        break;
      case "saved":
        if (msg.artwork && msg.artwork.id) this.continuingId = msg.artwork.id;
        this._emit(msg.t, msg);
        break;
      case "continue":
      case "gallery":
        this._emit(msg.t, msg);
        break;
      default:
        break; // pong and unknown messages need no handling
    }
  }

  _onBinary(buf) {
    if (!(buf instanceof ArrayBuffer) || buf.byteLength < HEADER) return;
    const view = new DataView(buf);
    if (view.getUint8(0) !== PICTURE) return;
    const seq = view.getUint32(1, false);
    // Never sent by this page (e.g. made from another screen's sketch): ignore it, or
    // it would block our own lower-numbered pictures.
    if (seq > this.seq) return;
    if (this._inflight && seq >= this._inflight.seq) this._clearInflight();
    // Drop stale or out-of-order pictures. The same seq again is a "settled" picture
    // (the GPU pipeline caught up with that sketch) and replaces the one shown.
    if (seq < this.floorSeq || seq < this.lastPictureSeq) return;
    this.lastPictureSeq = seq;
    const final = !this.pendingFinal && seq >= this.finalSeq;
    this._emit("picture", { seq, jpeg: buf.slice(HEADER), final });
  }

  // ---- outgoing sketches ----

  _loop() {
    if (!this._stopped) requestAnimationFrame(this._loop);
    if (!this.live || this._inflight || this._sending) return;
    if (this.pendingFinal) this._sendSketch(true);
    else if (this.canvas.isDirty()) this._sendSketch(false);
  }

  async _sendSketch(final) {
    this._sending = true;
    if (final) this.pendingFinal = false;
    this.canvas.clearDirty(); // strokes drawn while encoding mark it dirty again
    const seq = (this.seq + 1) >>> 0;
    this.seq = seq;
    if (final) this.finalSeq = seq;
    try {
      const blob = await this.canvas.toBlob();
      const png = new Uint8Array(await blob.arrayBuffer());
      if (!this.connected) throw new Error("socket closed");
      const frame = new Uint8Array(HEADER + png.length);
      const view = new DataView(frame.buffer);
      view.setUint8(0, SKETCH);
      view.setUint32(1, seq, false);
      frame.set(png, HEADER);
      this.ws.send(frame.buffer);
      this._inflight = { seq, timer: setTimeout(() => this._inflightTimedOut(seq), INFLIGHT_TIMEOUT_MS) };
    } catch {
      // Not sent (socket closed mid-encode): resend the drawing once we are back.
      this.pendingFinal = true;
    } finally {
      this._sending = false;
    }
  }

  /** No PICTURE within the timeout: free the slot, and retry an unanswered final frame
   * (the server drops frames that fail) a few times so Auto mode still shows a picture. */
  _inflightTimedOut(seq) {
    this._clearInflight();
    const unanswered = seq === this.finalSeq && this.lastPictureSeq < seq && !this.pendingFinal;
    if (unanswered && this._finalRetries < FINAL_RETRIES) {
      this._finalRetries += 1;
      this.pendingFinal = true;
    }
  }

  _clearInflight() {
    if (!this._inflight) return;
    clearTimeout(this._inflight.timer);
    this._inflight = null;
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}
