// Toolbar UI: colours, brush size, clear, view modes, topic/style pickers, connection dot.
// Pure UI: it renders state it is given and emits events; app.js wires it up.
//
// Events (CustomEvent detail): "color" {color, eraser}, "width" {width}, "clear",
// "mode" {mode}, "topic" {id}, "style" {id}.

import { isDark, tooClose } from "./paper.js";

const SWATCHES = [
  { color: "#22223b", name: "Black" },
  { color: "#ffffff", name: "White" },
  { color: "#ff4d5e", name: "Red" },
  { color: "#ff9f1c", name: "Orange" },
  { color: "#ffd23f", name: "Yellow" },
  { color: "#3ecf6e", name: "Green" },
  { color: "#2fa8ff", name: "Blue" },
  { color: "#8b5cf6", name: "Purple" },
  { color: "#ff7eb6", name: "Pink" },
  { color: "", name: "Eraser", eraser: true },
];

const CONN_LABELS = {
  off: "Not connected",
  connecting: "Connecting…",
  busy: "Getting ready…",
  ready: "Ready to dream",
  error: "Something went wrong, still trying",
  replaced: "Drawing on another screen",
};

// Line icons that start each row: a globe for worlds (topics), a palette for styles.
const ROW_ICONS = {
  topics: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/>'
    + '<path d="M3.5 12h17M12 3.5c2.4 2.5 3.6 5.3 3.6 8.5s-1.2 6-3.6 8.5c-2.4-2.5-3.6-5.3-3.6-8.5s1.2-6 3.6-8.5z"/></svg>',
  styles: '<svg viewBox="0 0 24 24" aria-hidden="true">'
    + '<path d="M12 3.5a8.5 8.5 0 1 0 0 17c1.1 0 1.7-.8 1.7-1.6 0-1.2-1-1.5-1-2.6 0-.9.7-1.6 1.7-1.6h2.1a3.9 3.9 0 0 0 3.9-3.9c0-4-3.8-7.3-8.4-7.3z"/>'
    + '<circle class="dot" cx="7.6" cy="11.4" r="1.25"/><circle class="dot" cx="10" cy="7.6" r="1.25"/>'
    + '<circle class="dot" cx="14.6" cy="7.6" r="1.25"/></svg>',
};

export class Toolbar extends EventTarget {
  constructor(root = document) {
    super();
    this.$ = (sel) => root.querySelector(sel);
    this.color = SWATCHES[0].color;
    this.eraser = false;
    this.background = "#000000"; // the drawing's paper, for the pen contrast and preview
    this.width = Number(this.$("#brush-size").value);
    this.scale = 1; // CSS px per canvas px, for the brush preview
    this.topic = null;
    this.style = null;
    this._items = { topics: [], styles: [] };
    this.sheetKind = null; // "topics" | "styles" while the picker sheet is open
    this._renderSwatches();
    this._bindBrush();
    this._bindButtons();
    this._bindSheet();
    this.setItems("topics", null);
    this.setItems("styles", null);
    this._updatePreview();
  }

  // ---- public API ----

  /** Highlight the active mode button. */
  setMode(mode) {
    for (const b of this.$("#modes").querySelectorAll(".mode")) {
      const on = b.dataset.mode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", String(on));
    }
  }

  /** Connection dot: off | connecting | busy | ready | error | replaced. */
  setConn(state, detail = "") {
    const dot = this.$("#conn");
    dot.dataset.state = state;
    const label = CONN_LABELS[state] || state;
    dot.title = detail ? `${label} (${detail})` : label;
    dot.setAttribute("aria-label", label);
  }

  /**
   * The drawing's paper colour changed (or a new drawing started). If the pen would
   * (nearly) vanish on it, switch to white on dark paper or black on light paper.
   */
  setBackground(color) {
    this.background = color;
    const chip = this.$("#paper-chip");
    if (chip) chip.style.setProperty("--c", color);
    if (!this.eraser && tooClose(this.color, color)) {
      const contrast = isDark(color) ? "#ffffff" : SWATCHES[0].color;
      const swatch = this.$(`#swatches .swatch[data-color="${contrast}"]`);
      this._pickColor(contrast, swatch, false);
    } else {
      this._updatePreview();
    }
  }

  /** Display scale of the drawing canvas, so the brush preview matches what you draw. */
  setScale(scale) {
    this.scale = scale || 1;
    this._updatePreview();
  }

  /**
   * Fill a topic or style row: one button showing the current pick, which opens the
   * picker sheet. `items` null shows a "waiting" placeholder.
   * @param {"topics"|"styles"} kind
   * @param {Array<{id:string,label:string,emoji:string}>|null} items
   * @param {string} [current]
   */
  setItems(kind, items, current) {
    this._items[kind] = items || [];
    const row = this.$(`#${kind}`);
    row.textContent = "";
    row.classList.toggle("waiting", !items);
    const icon = document.createElement("span");
    icon.className = "row-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.innerHTML = ROW_ICONS[kind]; // static markup from this file, never user data
    row.append(icon);

    if (!items) {
      const ph = document.createElement("span");
      ph.className = "chooser placeholder";
      row.append(ph);
      return;
    }
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chooser";
    b.setAttribute("aria-haspopup", "dialog");
    const emoji = document.createElement("span");
    emoji.className = "emoji";
    const label = document.createElement("span");
    label.className = "label";
    const chevron = document.createElement("span");
    chevron.className = "chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6.5 9.5l5.5 5.5 5.5-5.5"/></svg>';
    b.append(emoji, label, chevron);
    b.addEventListener("click", () => this.openSheet(kind));
    row.append(b);
    if (this.sheetKind === kind) this._fillSheet(kind); // a parent changed the list
    this.select(kind, current);
  }

  /** Mark the current topic/style on its button and in the sheet. */
  select(kind, id) {
    if (kind === "topics") this.topic = id; else this.style = id;
    const b = this.$(`#${kind} .chooser`);
    if (b && b.tagName === "BUTTON") {
      const item = this._items[kind].find((it) => it.id === id);
      b.dataset.id = item ? item.id : "";
      b.querySelector(".emoji").textContent = item ? item.emoji || "✨" : "✨";
      b.querySelector(".label").textContent = item ? item.label : (kind === "topics" ? "Pick a world" : "Pick a style");
      b.setAttribute("aria-label", `${kind === "topics" ? "World" : "Style"}: ${item ? item.label : "none"}. Change`);
    }
    if (this.sheetKind === kind) {
      for (const t of this.$("#pick-grid").querySelectorAll(".pick-tile")) {
        const on = t.dataset.id === id;
        t.classList.toggle("active", on);
        t.setAttribute("aria-checked", String(on));
      }
    }
  }

  /** Show the sheet of all worlds or styles. */
  openSheet(kind) {
    this.sheetKind = kind;
    this._fillSheet(kind);
    const sheet = this.$("#pick-sheet");
    sheet.dataset.kind = kind;
    sheet.hidden = false;
    const grid = this.$("#pick-grid");
    const active = grid.querySelector(".pick-tile.active") || grid.querySelector(".pick-tile");
    if (active) {
      try { active.focus({ preventScroll: true }); } catch { /* old Safari */ }
      active.scrollIntoView({ block: "nearest" });
    }
  }

  closeSheet() {
    const kind = this.sheetKind;
    if (!kind) return;
    this.sheetKind = null;
    this.$("#pick-sheet").hidden = true;
    try { this.$(`#${kind} .chooser`).focus({ preventScroll: true }); } catch { /* gone */ }
  }

  get sheetOpen() { return !!this.sheetKind; }

  _fillSheet(kind) {
    this.$("#pick-title").textContent = kind === "topics" ? "Pick a world" : "Pick a style";
    const grid = this.$("#pick-grid");
    grid.textContent = "";
    const current = kind === "topics" ? this.topic : this.style;
    for (const item of this._items[kind]) {
      const t = document.createElement("button");
      t.type = "button";
      t.className = "pick-tile";
      t.dataset.id = item.id;
      t.setAttribute("role", "radio");
      const on = item.id === current;
      t.classList.toggle("active", on);
      t.setAttribute("aria-checked", String(on));
      const emoji = document.createElement("span");
      emoji.className = "emoji";
      emoji.textContent = item.emoji || "✨";
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = item.label;
      t.append(emoji, label);
      t.addEventListener("click", () => {
        this.select(kind, item.id);
        this.closeSheet();
        this._emit(kind === "topics" ? "topic" : "style", { id: item.id });
      });
      grid.append(t);
    }
  }

  // ---- internals ----

  _bindSheet() {
    const sheet = this.$("#pick-sheet");
    this.$("#pick-close").addEventListener("click", () => this.closeSheet());
    // Tapping the dimmed area around the card closes it too.
    sheet.addEventListener("click", (e) => { if (e.target === sheet) this.closeSheet(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && this.sheetOpen) this.closeSheet(); });
  }

  _renderSwatches() {
    const box = this.$("#swatches");
    for (const s of SWATCHES) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "swatch" + (s.eraser ? " eraser" : "");
      if (!s.eraser) {
        b.style.setProperty("--c", s.color);
        b.dataset.color = s.color;
        b.classList.toggle("light", isLight(s.color)); // white needs an edge on white
      }
      b.setAttribute("role", "radio");
      b.setAttribute("aria-label", s.name);
      if (s.eraser) {
        b.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 19h11M5.5 15.5l8-8a2 2 0 0 1 2.8 0l2.2 2.2a2 2 0 0 1 0 2.8L12 19H8.5l-3-3a.7.7 0 0 1 0-.5zM9.5 11.5l5 5"/></svg>';
      }
      b.addEventListener("click", () => this._pickColor(s.color, b, s.eraser));
      box.append(b);
    }

    // "More colours": a rainbow swatch with the native picker laid invisibly on top.
    const more = document.createElement("label");
    more.className = "swatch more";
    more.setAttribute("aria-label", "More colours");
    more.innerHTML = '<span class="chip"></span><input type="color" id="more-color" value="#00b3a4">';
    const input = more.querySelector("input");
    const pick = () => {
      more.style.setProperty("--c", input.value);
      more.classList.add("chosen");
      this._pickColor(input.value, more);
    };
    input.addEventListener("input", pick);
    input.addEventListener("change", pick);
    box.append(more);

    this._pickColor(this.color, box.querySelector(".swatch"));
  }

  _pickColor(color, el, eraser = false) {
    this.eraser = !!eraser;
    if (!eraser) this.color = color;
    for (const s of this.$("#swatches").querySelectorAll(".swatch")) {
      const on = s === el;
      s.classList.toggle("active", on);
      if (s.getAttribute("role") === "radio") s.setAttribute("aria-checked", String(on));
    }
    this._updatePreview();
    this._emit("color", { color: this.color, eraser: this.eraser });
  }

  _bindBrush() {
    const range = this.$("#brush-size");
    const onInput = () => {
      this.width = Number(range.value);
      this._updatePreview();
      this._emit("width", { width: this.width });
    };
    range.addEventListener("input", onInput);
    range.addEventListener("change", onInput);
    // Touch the track anywhere and slide: iPad Safari's own slider only moves when the small
    // thumb itself is grabbed, which is hard for little fingers.
    const setFromPointer = (e) => {
      const box = range.getBoundingClientRect();
      const min = Number(range.min), max = Number(range.max), step = Number(range.step) || 1;
      const fraction = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
      const value = Math.round((min + fraction * (max - min)) / step) * step;
      if (value !== Number(range.value)) {
        range.value = String(value);
        onInput();
      }
    };
    let sliding = null; // pointerId of the finger moving the slider
    range.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      sliding = e.pointerId;
      setFromPointer(e);
      try { range.setPointerCapture(e.pointerId); } catch { /* keep following moves on the slider */ }
    });
    range.addEventListener("pointermove", (e) => { if (e.pointerId === sliding) setFromPointer(e); });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      range.addEventListener(type, (e) => { if (e.pointerId === sliding) sliding = null; });
    }
  }

  _updatePreview() {
    const dot = this.$("#brush-preview span");
    const d = Math.max(4, Math.min(44, this.width * this.scale));
    dot.style.width = dot.style.height = `${d}px`;
    // The eraser previews as a dot of paper (it uncovers the background).
    const shown = this.eraser ? this.background : this.color;
    dot.style.background = shown;
    dot.classList.toggle("light", this.eraser || isLight(shown));
    const range = this.$("#brush-size");
    const pct = ((this.width - range.min) / (range.max - range.min)) * 100;
    range.style.setProperty("--fill", `${pct}%`);
  }

  _bindButtons() {
    this.$("#clear").addEventListener("click", () => {
      const btn = this.$("#clear");
      btn.classList.remove("poof");
      void btn.offsetWidth; // restart the little animation
      btn.classList.add("poof");
      this._emit("clear");
    });
    for (const b of this.$("#modes").querySelectorAll(".mode")) {
      b.setAttribute("role", "radio");
      b.addEventListener("click", () => this._emit("mode", { mode: b.dataset.mode }));
    }
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

/** Rough perceived brightness check, to outline very light colours. */
function isLight(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return false;
  const n = parseInt(m[1], 16);
  const r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return 0.299 * r + 0.587 * g + 0.114 * b > 225;
}
