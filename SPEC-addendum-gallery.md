# Addendum: saved artworks, gallery, resume (Alex, 2026-10-25)

_Updated 2026-10-26 to match the code; docs/setup.md is the user guide._ The request and original design are kept below; the "Implementation notes" at the end and the edits marked "now" record what the code does today (`server/gallery.py` is the source of truth).

"For storing data locally use a local json store or SQLite. Artworks saved as jpgs on disk in a folder. Save user sketches for future resuming or gallery as well as the diffusion result."

Storage
- Metadata in SQLite via stdlib `sqlite3`: `data/dreamer.db` (WAL mode, one connection per thread or a lock; schema versioned via `PRAGMA user_version` with simple migrations). Settings that already live in JSON (styles.json, params.json) stay JSON.
- Files on disk under `data/artworks/YYYY-MM/<artwork_id>/` (file names carry a version now, see the implementation notes):
  - `picture.jpg` — the generated result (JPEG, config jpeg_quality).
  - `sketch.png` — the child's sketch, kept LOSSLESS PNG so "continue drawing" restores clean lines (JPEG ringing would get re-interpreted by the model). Also `sketch.jpg` is NOT needed.
  - `thumb.jpg` — 256px thumbnail of the picture for the gallery grid.
- `artworks` table: id (short random url-safe), created_at, updated_at (ISO UTC), title (nullable, max 60 chars), topic_id, style_id, prompt, params_json (GenParams snapshot, plus `canvas`), seed, dir (relative path), is_draft (0/1), deleted_at (nullable; soft delete, purge files on admin "empty trash"), source_id (the artwork a draft or continued drawing came from).
- Paths are always built server-side from the id; never accept paths from clients. Validate ids with a strict regex (12 url-safe characters, `secrets.token_urlsafe(9)`).

Server (`server/gallery.py`, `GalleryStore`)
- `save(sketch_png, picture_jpeg, meta) -> Artwork`, `update(id, sketch_png, picture_jpeg, meta)`, `list(limit, offset, include_drafts=False, deleted=False)`, `get(id)`, `count()`, `rename(id, title)`, `soft_delete(id)`, `restore(id)`, `purge_deleted()`, `drafts()`, `delete_draft(id)`, `sweep()`. `DraftAutosaver` (same module) does the draft autosave.
- Writes files atomically (temp + os.replace), then the DB row; on DB failure remove the files.
- The worker already has the latest sketch PNG and picture JPEG: keep `last_frame = (seq, sketch_png, picture_jpeg, prompt, params)` in the session so Save uses exactly what is on screen (no regeneration).

Protocol (kid WebSocket)
- client -> `{"t":"save"}` -> server saves the current last_frame as a new artwork (or updates the artwork being continued; "continuing" is tracked per page, see the notes) and replies `{"t":"saved","artwork":{...public fields, "thumb":"/api/artworks/<id>/thumb.jpg"}}`; error `{"t":"status","state":"error","msg":"Nothing to save yet"}` if no frame.
- client -> `{"t":"new"}` -> clears `continuing_id` (fresh drawing).
- client -> `{"t":"continue","id":"..."}` -> server sets `continuing_id`, replies `{"t":"continue","id","sketch":"/api/artworks/<id>/sketch.png","topic","style"}`; the page loads the PNG into the drawing canvas (drawn onto the stroke layer of the drawing canvas, whose size follows the drawing's orientation, see SPEC-addendum-canvas.md), restores topic/style, and sends it as the next SKETCH.
- Draft autosave: the server autosaves the current drawing as a draft (is_draft=1, single draft row per session, overwritten) at most every 10 s while frames change, and on clear/new. On kid page load, `ready` includes `"draft":{id, sketch url}|null` and the page offers "Keep drawing?" (or restores automatically — choose automatic restore with an undo/"start fresh" button).

HTTP
- Public (kid LAN): `GET /api/artworks?limit=&offset=` (non-draft, non-deleted, newest first; limit capped at 100; returns `{artworks, more, ...}`), `GET /api/artworks/{id}`, `GET /api/artworks/{id}/picture.jpg|sketch.png|thumb.jpg` (correct content types, cache headers ok since immutable per update — add `?v=updated_at` in URLs).
- Admin (cookie): `DELETE /api/admin/artworks/{id}` (soft), `POST /api/admin/artworks/{id}/restore`, `POST /api/admin/artworks/purge`, `PATCH /api/admin/artworks/{id}` {title}, `GET /api/admin/artworks?trash=` (list or trash view, with counts) and `GET /api/admin/artworks/{id}/{name}?download=true` for the download links. Kids cannot delete (prevents accidental loss); the kid page shows no delete.

Kid page
- Toolbar: big "Save" (star/heart icon) with a satisfying confirmation animation (thumbnail flies into the gallery button), and a "Gallery" button.
- Gallery view: full-screen grid of thumbnails (large, 3–4 per row on iPad), tap -> detail with picture and sketch side by side (or flip), buttons "Keep drawing" (continue) and "Back". Infinite scroll / "more" paging.
- "New drawing" button (clears and sends `new`; confirm only if unsaved changes).

Admin page
- Gallery section: grid, open, rename, delete (soft) with Trash view to restore or empty trash; download links for picture.jpg and sketch.png.

Tests: GalleryStore CRUD/atomicity/soft-delete/purge, id validation and no path traversal, ws save/continue/new flows, draft autosave and restore on reconnect, admin-only deletion (401 without cookie), HTTP content types; e2e: draw -> save -> gallery shows it -> continue restores sketch -> admin deletes -> gone from kid gallery.

Implementation notes (deviations found in review, 2026-10-25)
- "Being continued" belongs to the page's canvas, not to the server session. `hello` carries `"continuing": <id>|null` (the artwork the page's canvas holds, so a reconnect keeps it; a freshly loaded page sends null). `continue` only answers with the sketch to load; the page then sends `{"t":"continuing","id"}` once the sketch is on its canvas (a sketch that fails to load changes nothing). After restoring the draft the page sends `{"t":"restored","id":<draft id>}` and continues the draft's `source` (now included in `ready.draft`). This way a page that never restored anything cannot save over an existing artwork.
- The session has a drawing epoch, bumped on hello/clear/new/continuing/restored. A picture made from a sketch of an older epoch is still sent (the page drops it) but never becomes `last_frame`, so it cannot be saved or autosaved as the draft.
- Files on disk carry the version: `picture.<v>.jpg`, `sketch.<v>.png`, `thumb.<v>.jpg` (`<v>` = the `?v=` of the URLs). An update writes the new version, commits the row, then deletes the old files, so a failure never mixes versions. Folders with plain names from earlier builds are still read. `GalleryStore.sweep()` (at startup and after emptying the trash) removes orphaned folders, old versions and temp files.
- Clear asks first when the drawing is unsaved (like New), because it also drops the draft.
