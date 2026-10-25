// Gallery view (SPEC-addendum-gallery): a full-screen grid of saved pictures, paged from
// GET /api/artworks, and a detail view with the picture and the drawing side by side.
// Kids can look and "Keep drawing"; there is no delete here (that is a grown-up thing).
//
// Events (CustomEvent detail): "continue" {artwork}, "close".

const PAGE = 24;

export class Gallery extends EventTarget {
  constructor(root = document) {
    super();
    this.$ = (sel) => root.querySelector(sel);
    this.el = this.$("#gallery");
    this.grid = this.$("#gallery-grid");
    this.more = this.$("#gallery-more");
    this.scroller = this.$("#gallery-scroll");
    this.items = [];
    this.hasMore = false;
    this.loading = false;
    this.current = null;    // artwork shown in the detail view
    this._gen = 0;          // newest load wins (refresh while a page is loading)

    this.$("#gallery-back").addEventListener("click", () => this.close());
    this.$("#detail-back").addEventListener("click", () => this.showDetail(null));
    this.$("#detail-continue").addEventListener("click", () => {
      if (this.current) this._emit("continue", { artwork: this.current });
    });
    this.more.addEventListener("click", () => this._loadPage());
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

  /** Reload from the first page (after a save, or when the parent changed something). */
  refresh() {
    this._gen += 1;
    this.items = [];
    this.hasMore = false;
    this.loading = false;
    this._loadPage(true);
  }

  /** Detail view for one artwork, or back to the grid with null. */
  showDetail(art) {
    this.current = art;
    this.$("#detail").hidden = !art;
    this.scroller.hidden = !!art;
    if (!art) return;
    this.$("#detail-picture").src = art.picture;
    this.$("#detail-sketch").src = art.sketch;
  }

  async _loadPage(first = false) {
    if (this.loading) return;
    this.loading = true;
    const gen = this._gen;
    this.more.disabled = true;
    try {
      const res = await fetch(`/api/artworks?limit=${PAGE}&offset=${this.items.length}`,
                              { cache: "no-store" });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      if (gen !== this._gen) return;
      const fresh = (data.artworks || []).filter((a) => !this.items.some((b) => b.id === a.id));
      this.items.push(...fresh);
      this.hasMore = !!data.more;
      this._render(first);
      // A deleted artwork that is open in the detail view goes away.
      if (this.current && !this.items.some((a) => a.id === this.current.id) && !this.hasMore) {
        this.showDetail(null);
      }
    } catch {
      if (gen === this._gen) this.hasMore = true; // offer "More" as a retry
      this._render(first);
    } finally {
      if (gen === this._gen) this.loading = false;
      this.more.disabled = false;
    }
  }

  _render(first) {
    if (first) this.grid.textContent = "";
    const have = new Set([...this.grid.children].map((c) => c.dataset.id));
    for (const art of this.items) {
      if (have.has(art.id)) continue;
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
      b.addEventListener("click", () => this.showDetail(art));
      this.grid.append(b);
    }
    this.$("#gallery-empty").hidden = this.items.length > 0 || this.hasMore;
    this.more.hidden = !this.hasMore;
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}
