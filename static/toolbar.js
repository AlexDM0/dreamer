// Toolbar UI: colours, brush size, clear, view modes, topic/style rows, connection dot.
// Pure UI: it renders state it is given and emits events; app.js wires it up.
//
// Events (CustomEvent detail): "color" {color}, "width" {width}, "clear",
// "mode" {mode}, "topic" {id}, "style" {id}.

const SWATCHES = [
  { color: "#22223b", name: "Black" },
  { color: "#ff4d5e", name: "Red" },
  { color: "#ff9f1c", name: "Orange" },
  { color: "#ffd23f", name: "Yellow" },
  { color: "#3ecf6e", name: "Green" },
  { color: "#2fa8ff", name: "Blue" },
  { color: "#8b5cf6", name: "Purple" },
  { color: "#ff7eb6", name: "Pink" },
  { color: "#ffffff", name: "Eraser", eraser: true },
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
    this.width = Number(this.$("#brush-size").value);
    this.scale = 1; // CSS px per canvas px, for the brush preview
    this.topic = null;
    this.style = null;
    this._renderSwatches();
    this._bindBrush();
    this._bindButtons();
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

  /** Display scale of the drawing canvas, so the brush preview matches what you draw. */
  setScale(scale) {
    this.scale = scale || 1;
    this._updatePreview();
  }

  /**
   * Fill a topic or style row. `items` null shows a "waiting" placeholder row.
   * @param {"topics"|"styles"} kind
   * @param {Array<{id:string,label:string,emoji:string}>|null} items
   * @param {string} [current]
   */
  setItems(kind, items, current) {
    const row = this.$(`#${kind}`);
    row.textContent = "";
    row.classList.toggle("waiting", !items);
    const icon = document.createElement("span");
    icon.className = "row-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.innerHTML = ROW_ICONS[kind]; // static markup from this file, never user data
    row.append(icon);

    if (!items) {
      for (let i = 0; i < 4; i++) {
        const ph = document.createElement("span");
        ph.className = "pick placeholder";
        ph.style.animationDelay = `${i * 120}ms`;
        row.append(ph);
      }
      return;
    }
    for (const item of items) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "pick";
      b.dataset.id = item.id;
      b.setAttribute("aria-pressed", "false");
      const emoji = document.createElement("span");
      emoji.className = "emoji";
      emoji.textContent = item.emoji || "✨";
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = item.label;
      b.append(emoji, label);
      b.addEventListener("click", () => {
        this.select(kind, item.id);
        this._emit(kind === "topics" ? "topic" : "style", { id: item.id });
      });
      row.append(b);
    }
    this.select(kind, current, false);
  }

  /** Highlight the current topic/style and scroll it into view. */
  select(kind, id, scroll = true) {
    if (kind === "topics") this.topic = id; else this.style = id;
    const row = this.$(`#${kind}`);
    for (const b of row.querySelectorAll("button.pick")) {
      const on = b.dataset.id === id;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", String(on));
      if (on && scroll) b.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    }
  }

  // ---- internals ----

  _renderSwatches() {
    const box = this.$("#swatches");
    for (const s of SWATCHES) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "swatch" + (s.eraser ? " eraser" : "");
      b.style.setProperty("--c", s.color);
      b.dataset.color = s.color;
      b.setAttribute("role", "radio");
      b.setAttribute("aria-label", s.name);
      if (s.eraser) {
        b.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 19h11M5.5 15.5l8-8a2 2 0 0 1 2.8 0l2.2 2.2a2 2 0 0 1 0 2.8L12 19H8.5l-3-3a.7.7 0 0 1 0-.5zM9.5 11.5l5 5"/></svg>';
      }
      b.addEventListener("click", () => this._pickColor(s.color, b));
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

  _pickColor(color, el) {
    this.color = color;
    for (const s of this.$("#swatches").querySelectorAll(".swatch")) {
      const on = s === el;
      s.classList.toggle("active", on);
      if (s.getAttribute("role") === "radio") s.setAttribute("aria-checked", String(on));
    }
    this._updatePreview();
    this._emit("color", { color });
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
  }

  _updatePreview() {
    const dot = this.$("#brush-preview span");
    const d = Math.max(4, Math.min(44, this.width * this.scale));
    dot.style.width = dot.style.height = `${d}px`;
    dot.style.background = this.color;
    dot.classList.toggle("light", isLight(this.color));
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
