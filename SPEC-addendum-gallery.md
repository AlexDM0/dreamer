# Addendum: saved artworks, gallery, resume (Alex, 2026-10-25)

"For storing data locally use a local json store or SQLite. Artworks saved as jpgs on disk in a folder. Save user sketches for future resuming or gallery as well as the diffusion result."

Storage
- Metadata in SQLite via stdlib `sqlite3`: `data/dreamer.db` (WAL mode, one connection per thread or a lock; schema versioned via `PRAGMA user_version` with simple migrations). Settings that already live in JSON (styles.json, params.json) stay JSON.
- Files on disk under `data/artworks/YYYY-MM/<artwork_id>/`:
  - `picture.jpg` — the generated result (JPEG, config jpeg_quality).
  - `sketch.png` — the child's sketch, kept LOSSLESS PNG so "continue drawing" restores clean lines (JPEG ringing would get re-interpreted by the model). Also `sketch.jpg` is NOT needed.
  - `thumb.jpg` — 256px thumbnail of the picture for the gallery grid.
- `artworks` table: id (short random url-safe), created_at, updated_at (ISO UTC), title (nullable), topic_id, style_id, prompt, params_json (GenParams snapshot), seed, dir (relative path), is_draft (0/1), deleted_at (nullable; soft delete, purge files on admin "empty trash").
- Paths are always built server-side from the id; never accept paths from clients. Validate ids with a strict regex.

Server (`server/gallery.py`, `GalleryStore`)
- `save(sketch_png, picture_jpeg, meta) -> Artwork`, `update(id, sketch_png, picture_jpeg, meta)`, `list(limit, offset, include_drafts=False)`, `get(id)`, `soft_delete(id)`, `restore(id)`, `purge_deleted()`.
- Writes files atomically (temp + os.replace), then the DB row; on DB failure remove the files.
- The worker already has the latest sketch PNG and picture JPEG: keep `last_frame = (seq, sketch_png, picture_jpeg, prompt, params)` in the session so Save uses exactly what is on screen (no regeneration).

Protocol (kid WebSocket)
- client -> `{"t":"save"}` -> server saves the current last_frame as a new artwork (or updates the artwork being continued, if the session has `continuing_id`) and replies `{"t":"saved","artwork":{...public fields, "thumb":"/api/artworks/<id>/thumb.jpg"}}`; error `{"t":"status","state":"error","msg":"Nothing to save yet"}` if no frame.
- client -> `{"t":"new"}` -> clears `continuing_id` (fresh drawing).
- client -> `{"t":"continue","id":"..."}` -> server sets `continuing_id`, replies `{"t":"continue","id","sketch":"/api/artworks/<id>/sketch.png","topic","style"}`; the page loads the PNG into the drawing canvas (drawImage onto the 512 backing canvas), restores topic/style, and sends it as the next SKETCH.
- Draft autosave: the server autosaves the current drawing as a draft (is_draft=1, single draft row per session, overwritten) at most every 10 s while frames change, and on clear/new. On kid page load, `ready` includes `"draft":{id, sketch url}|null` and the page offers "Keep drawing?" (or restores automatically — choose automatic restore with an undo/"start fresh" button).

HTTP
- Public (kid LAN): `GET /api/artworks?limit=&offset=` (non-draft, non-deleted, newest first), `GET /api/artworks/{id}`, `GET /api/artworks/{id}/picture.jpg|sketch.png|thumb.jpg` (correct content types, cache headers ok since immutable per update — add `?v=updated_at` in URLs).
- Admin (cookie): `DELETE /api/admin/artworks/{id}` (soft), `POST /api/admin/artworks/{id}/restore`, `POST /api/admin/artworks/purge`, `PATCH /api/admin/artworks/{id}` {title}. Kids cannot delete (prevents accidental loss); the kid page shows no delete.

Kid page
- Toolbar: big "Save" (star/heart icon) with a satisfying confirmation animation (thumbnail flies into the gallery button), and a "Gallery" button.
- Gallery view: full-screen grid of thumbnails (large, 3–4 per row on iPad), tap -> detail with picture and sketch side by side (or flip), buttons "Keep drawing" (continue) and "Back". Infinite scroll / "more" paging.
- "New drawing" button (clears and sends `new`; confirm only if unsaved changes).

Admin page
- Gallery section: grid, open, rename, delete (soft) with Trash view to restore or empty trash; download links for picture.jpg and sketch.png.

Tests: GalleryStore CRUD/atomicity/soft-delete/purge, id validation and no path traversal, ws save/continue/new flows, draft autosave and restore on reconnect, admin-only deletion (401 without cookie), HTTP content types; e2e: draw -> save -> gallery shows it -> continue restores sketch -> admin deletes -> gone from kid gallery.
