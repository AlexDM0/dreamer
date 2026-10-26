// Gallery view (SPEC-addendum-gallery): a full-screen grid of saved pictures, paged from
// GET /api/artworks, and a detail view with the picture and the drawing side by side.
// Kids can look and "Keep drawing"; there is no delete here (that is a grown-up thing).
//
// Events (CustomEvent detail): "continue" {artwork}, "close".
//
// A saved sketch.png is the transparent stroke layer; it is shown over the artwork's
// `background` colour (older artworks: an opaque sketch on white).

import { normaliseColor } from "./canvas.js";

const PAGE = 24;
const RELOAD_MAX = 100; // the server's page size limit

export class Gallery extends EventTarget {
  constructor(root = document) {
    super();
    this.$ = (sel) => root.querySelector(sel);
    this.el = this.$("#gallery");
    this.grid = this.$("#gallery-grid");
    this.more = this.$("#gallery-more");
    this.error = this.$("#gallery-error");
    this.scroller = this.$("#gallery-scroll");
    this.items = [];
    this.hasMore = false;
    this.loading = false;
    this.failed = false;    // the last load could not reach the server
    this.current = null;    // artwork shown in the detail view
    this._gen = 0;          // newest load wins (refresh while a page is loading)

    this.$("#gallery-back").addEventListener("click", () => this.close());
    this.$("#detail-back").addEventListener("click", () => this.showDetail(null));
    this.$("#detail-continue").addEventListener("click", () => {
      if (this.current) this._emit("continue", { artwork: this.current });
    });
    this.more.addEventListener("click", () => this._loadPage());
    this.$("#gallery-retry").addEventListener("click", () => this.update());
    // Infinite scroll: load the next page as the "more" button comes into view.
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting) && this.hasMore && !this.el.hidden) this._loadPage();
      }, { root: this.scroller, rootMargin: "300px" }).observe(this.more);
    }
  }

  get isOpen() { return !this.el.hidden; }

  open() {
    this.el.hidden = false;
    this.showDetail(null);
    this.refresh();
  }

  close() {
    if (this.el.hidden) return;
    this.el.hidden = true;
    this._emit("close", {});
  }

  /** Reload from the first page (when the gallery opens). */
  refresh() {
    this._gen += 1;
    this.items = [];
    this.hasMore = false;
    this.loading = false;
    this.scroller.scrollTop = 0;
    this._loadPage(true);
  }

  /** Something changed (a save, or the parent renamed / deleted / restored a picture):
   * reload what is already shown and update the grid in place, keeping the scroll spot. */
  async update() {
    if (!this.items.length) { this.refresh(); return; }
    this._gen += 1;
    const gen = this._gen;
    this.loading = true;
    const limit = Math.min(Math.max(this.items.length, PAGE), RELOAD_MAX);
    try {
      const data = await this._fetch(limit, 0);
      if (gen !== this._gen) return;
      this.items = data.artworks || [];
      this.hasMore = !!data.more;
      this.failed = false;
      this._render(false);
      this._dropMissingDetail();
    } catch {
      if (gen === this._gen) { this.failed = true; this._render(false); }
    } finally {
      if (gen === this._gen) {
        this.loading = false;
        this.more.disabled = false; // in case this replaced a page load that disabled it
      }
    }
  }

  /** Detail view for one artwork, or back to the grid with null. */
  showDetail(art) {
    this.current = art;
    this.$("#detail").hidden = !art;
    this.scroller.hidden = !!art;
    if (!art) return;
    this.$("#detail-picture").src = art.picture;
    const sketch = this.$("#detail-sketch");
    sketch.style.backgroundColor = normaliseColor(art.background, "#ffffff");
    sketch.src = art.sketch;
  }

  async _loadPage(first = false) {
    if (this.loading) return;
    this.loading = true;
    const gen = this._gen;
    this.more.disabled = true;
    try {
      const data = await this._fetch(PAGE, this.items.length);
      if (gen !== this._gen) return;
      const fresh = (data.artworks || []).filter((a) => !this.items.some((b) => b.id === a.id));
      this.items.push(...fresh);
      this.hasMore = !!data.more;
      this.failed = false;
      this._render(first);
      this._dropMissingDetail();
    } catch {
      if (gen === this._gen) { this.failed = true; this._render(first); }
    } finally {
      if (gen === this._gen) {
        this.loading = false;
        this.more.disabled = false; // a newer load owns the button otherwise
      }
    }
  }

  async _fetch(limit, offset) {
    const res = await fetch(`/api/artworks?limit=${limit}&offset=${offset}`, { cache: "no-store" });
    if (!res.ok) throw new Error(String(res.status));
    return res.json();
  }

  /** A deleted artwork that is open in the detail view goes away. */
  _dropMissingDetail() {
    if (this.current && !this.items.some((a) => a.id === this.current.id) && !this.hasMore) {
      this.showDetail(null);
    }
  }

  /** Make the grid match `items`, reusing the buttons of unchanged pictures. */
  _render(first) {
    if (first) this.grid.textContent = "";
    const old = new Map([...this.grid.children].map((c) => [c.dataset.id, c]));
    const buttons = this.items.map((art) => {
      const b = old.get(art.id);
      return b && b.dataset.key === this._key(art) ? b : this._thumb(art);
    });
    if (buttons.length !== this.grid.children.length
        || buttons.some((b, i) => this.grid.children[i] !== b)) {
      const top = this.scroller.scrollTop;
      this.grid.replaceChildren(...buttons);
      this.scroller.scrollTop = top;
    }
    const empty = this.items.length === 0;
    this.error.hidden = !this.failed;
    this.$("#gallery-empty").hidden = !empty || this.hasMore || this.failed;
    this.more.hidden = !this.hasMore || this.failed;
  }

  _key(art) { return `${art.thumb}|${art.title || ""}`; }

  _thumb(art) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "thumb";
    b.dataset.id = art.id;
    b.setAttribute("aria-label", art.title || "A saved picture");
    const img = document.createElement("img");
    img.src = art.thumb;
    img.alt = "";
    img.loading = "lazy";
    img.decoding = "async";
    b.append(img);
    if (art.title) {
      const t = document.createElement("span");
      t.className = "thumb-title";
      t.textContent = art.title;
      b.append(t);
    }
    b.dataset.key = this._key(art);
    b.addEventListener("click", () => this.showDetail(art));
    return b;
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}
