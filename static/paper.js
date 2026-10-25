// Paper (the drawing's background layer): the colours a child can pick, the picker sheet,
// and the small colour helpers that keep the pen visible on the paper.
//
// The paper colours are content (like the pen colours in toolbar.js), not chrome, so they
// live here as values that are also sent to the server.
//
// PaperPicker events (CustomEvent detail): "pick" {color, purpose}, "cancel" {purpose}.
// `purpose` is what open() was called with ("new" starts a drawing, "change" recolours).

export const PAPERS = [
  { color: "#000000", name: "Black" },
  { color: "#ffffff", name: "White" },
  { color: "#1e2a5a", name: "Night blue" },
  { color: "#8fd3ff", name: "Sky blue" },
  { color: "#4caf50", name: "Grass green" },
  { color: "#ffd23f", name: "Sunny yellow" },
  { color: "#ff9ec7", name: "Pink" },
  { color: "#c9b6f2", name: "Lilac" },
];

function channels(hex) {
  const value = parseInt(String(hex).slice(1, 7), 16) || 0;
  return [value >> 16, (value >> 8) & 255, value & 255];
}

/** Perceived brightness, 0 (black) .. 1 (white). */
export function luminance(hex) {
  const [red, green, blue] = channels(hex);
  return (0.299 * red + 0.587 * green + 0.114 * blue) / 255;
}

export function isDark(hex) { return luminance(hex) < 0.5; }

/** True when a pen of this colour would (nearly) vanish on this paper. */
export function tooClose(penHex, paperHex) {
  const pen = channels(penHex);
  const paper = channels(paperHex);
  const distance = Math.hypot(pen[0] - paper[0], pen[1] - paper[1], pen[2] - paper[2]);
  return distance < 90;
}

export class PaperPicker extends EventTarget {
  constructor(root = document) {
    super();
    this.$ = (sel) => root.querySelector(sel);
    this.el = this.$("#bg-picker");
    this.grid = this.$("#bg-grid");
    this.purpose = null;
    for (const paper of PAPERS) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "paper-swatch";
      button.dataset.bg = paper.color;
      button.style.setProperty("--c", paper.color);
      button.setAttribute("role", "radio");
      button.setAttribute("aria-label", paper.name);
      button.classList.toggle("light", !isDark(paper.color));
      button.addEventListener("click", () => this._pick(paper.color));
      this.grid.append(button);
    }
    this.$("#bg-close").addEventListener("click", () => this.close());
    // Tapping the dimmed area around the card closes it too.
    this.el.addEventListener("click", (e) => { if (e.target === this.el) this.close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && this.isOpen) this.close(); });
  }

  get isOpen() { return !this.el.hidden; }

  /** Show the sheet with `selected` marked. `purpose`: "new" | "change". */
  open(purpose, selected) {
    this.purpose = purpose;
    for (const button of this.grid.querySelectorAll(".paper-swatch")) {
      const on = button.dataset.bg === selected;
      button.classList.toggle("active", on);
      button.setAttribute("aria-checked", String(on));
    }
    this.el.dataset.purpose = purpose;
    this.el.hidden = false;
    const active = this.grid.querySelector(".paper-swatch.active") || this.grid.querySelector(".paper-swatch");
    try { active.focus({ preventScroll: true }); } catch { /* old Safari */ }
  }

  /** Close without picking. */
  close() {
    if (!this.isOpen) return;
    this.el.hidden = true;
    const purpose = this.purpose;
    this.purpose = null;
    this.dispatchEvent(new CustomEvent("cancel", { detail: { purpose } }));
  }

  _pick(color) {
    const purpose = this.purpose;
    this.el.hidden = true;
    this.purpose = null;
    this.dispatchEvent(new CustomEvent("pick", { detail: { color, purpose } }));
  }
}
