// Parent gate: the Dreamer icon in the toolbar opens a PIN pad; the right PIN logs in
// (the same /api/admin/login as the admin page, so the server's lock-out applies) and
// opens /admin. It always asks, even when the iPad still has a valid login, and a login
// from here lasts 15 minutes (the phone's admin page keeps its 30 days): the child's own
// screen must never open the parent page by itself.

import { PIN_MAX, renderPinDots, shakePinDots } from "./pin.js";

export class ParentGate {
  constructor(button, overlay) {
    this.overlay = overlay;
    this.$ = (sel) => overlay.querySelector(sel);
    this.pin = "";
    this.busy = false;
    button.addEventListener("click", () => this.open());
    this.$("#gate-close").addEventListener("click", () => this.close());
    overlay.addEventListener("click", (e) => { if (e.target === overlay) this.close(); });
    this.$("#gate-pad").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-k]");
      if (b) this.press(b.dataset.k);
    });
    document.addEventListener("keydown", (e) => {
      if (overlay.hidden || e.metaKey || e.ctrlKey || e.altKey) return;
      if (/^\d$/.test(e.key)) this.press(e.key);
      else if (e.key === "Backspace") this.press("back");
      else if (e.key === "Enter") this.press("ok");
      else if (e.key === "Escape") this.close();
      else return;
      e.preventDefault();
    });
  }

  open() {
    this.pin = "";
    this.$("#gate-msg").textContent = "";
    this._render();
    this.overlay.hidden = false;
  }

  close() {
    this.overlay.hidden = true;
    this.pin = "";
  }

  press(k) {
    if (k === "ok") { this._submit(); return; }
    if (k === "back") this.pin = this.pin.slice(0, -1);
    else if (/^\d$/.test(k) && this.pin.length < PIN_MAX) this.pin += k;
    this.$("#gate-msg").textContent = "";
    this._render();
  }

  async _submit() {
    if (!this.pin || this.busy) return;
    this.busy = true;
    try {
      const res = await fetch("/api/admin/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pin: this.pin, short: true }),
      });
      if (res.ok) {
        location.href = "/admin";
        return;
      }
      let msg = res.status === 401 ? "Wrong PIN" : "That didn't work. Try again.";
      try { msg = (await res.json()).detail || msg; } catch { /* not JSON */ }
      this._fail(typeof msg === "string" ? msg : "Wrong PIN");
    } catch {
      this._fail("Can't reach the computer.");
    } finally {
      this.busy = false;
    }
  }

  _fail(msg) {
    this.pin = "";
    this._render();
    this.$("#gate-msg").textContent = msg;
    shakePinDots(this.$("#gate-dots"));
  }

  _render() {
    renderPinDots(this.$("#gate-dots"), this.pin.length);
  }
}
