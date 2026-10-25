"""Saved artworks (SPEC-addendum-gallery): SQLite metadata + image files on disk.

Layout under ``data_dir``::

    dreamer.db                              metadata (WAL mode, PRAGMA user_version schema)
    artworks/YYYY-MM/<artwork_id>/picture.<v>.jpg   generated picture (as shown on screen)
                                  sketch.<v>.png    the child's sketch, lossless
                                  thumb.<v>.jpg     256 px thumbnail of the picture

``<v>`` is the artwork's version (its ``updated_at`` digits, also the ``?v=`` of its URLs),
so the files of one version never change: an update writes the next version's files next
to the old ones, commits the row that points at them, and only then deletes the old
files. A failure at any point leaves one complete, consistent set (and at worst stray
files, which ``sweep()`` removes). Folders from before versioned names hold plain
``picture.jpg`` etc.; those are still read, and replaced by the first update.

Every path is built here from a validated id; nothing path-like is ever taken from a
client. Files are written atomically (temp file + ``os.replace``) before the database
row, and removed again when the row cannot be written.

``GalleryStore`` is thread-safe (one connection guarded by a lock), so the app calls it
through ``asyncio.to_thread``. ``DraftAutosaver`` runs on the event loop and keeps a single
draft row up to date with what the kid is drawing.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image

from . import imaging

log = logging.getLogger("dreamer.gallery")

DB_FILE = "dreamer.db"
ARTWORKS_DIR = "artworks"
ID_RE = re.compile(r"[A-Za-z0-9_-]{12}")      # secrets.token_urlsafe(9)
DIR_RE = re.compile(r"artworks/\d{4}-\d{2}/([A-Za-z0-9_-]{12})")
MONTH_RE = re.compile(r"\d{4}-\d{2}")
FILES = {"picture.jpg": "image/jpeg", "sketch.png": "image/png", "thumb.jpg": "image/jpeg"}
THUMB_SIZE = 256
THUMB_QUALITY = 80
TITLE_MAX = 60
LIST_MAX = 100
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"
REPLACE_RETRIES = 5  # Windows refuses to replace a file someone is reading; retry briefly

# Schema migrations: (version, statements). Append only; never edit a released entry.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, [
        """CREATE TABLE artworks (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            title TEXT,
            topic_id TEXT,
            style_id TEXT,
            prompt TEXT NOT NULL DEFAULT '',
            params_json TEXT NOT NULL DEFAULT '{}',
            seed INTEGER,
            dir TEXT NOT NULL,
            is_draft INTEGER NOT NULL DEFAULT 0,
            deleted_at TEXT,
            source_id TEXT
        )""",
        "CREATE INDEX artworks_listing ON artworks (is_draft, deleted_at, updated_at)",
    ]),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]

_COLUMNS = ("id, created_at, updated_at, title, topic_id, style_id, prompt, params_json, "
            "seed, dir, is_draft, deleted_at, source_id")


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _version(stamp: str) -> str:
    """Cache-busting / file version token of an ``updated_at`` timestamp."""
    return re.sub(r"\D", "", stamp)


def _versioned(name: str, version: str) -> str:
    """``picture.jpg`` -> ``picture.<version>.jpg`` (the name on disk)."""
    stem, ext = name.rsplit(".", 1)
    return f"{stem}.{version}.{ext}"


def is_blank_sketch(png: bytes) -> bool:
    """True when a sketch PNG has nothing drawn on it (or cannot be decoded)."""
    try:
        img = Image.open(io.BytesIO(png), formats=["PNG"])
        img.load()
    except Exception:
        return True
    rgba = img.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return imaging.is_blank(Image.alpha_composite(white, rgba))


def make_thumb(picture_jpeg: bytes) -> bytes:
    img = Image.open(io.BytesIO(picture_jpeg), formats=["JPEG"])
    img.draft("RGB", (THUMB_SIZE, THUMB_SIZE))
    img = img.convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
    return imaging.encode_jpeg(img, THUMB_QUALITY)


def clean_title(value: Any) -> str | None:
    """Admin rename: strip, collapse whitespace, max TITLE_MAX; empty -> None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("title must be text")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("title has control characters")
    text = " ".join(value.split())
    if len(text) > TITLE_MAX:
        raise ValueError(f"title is too long (max {TITLE_MAX})")
    return text or None


@dataclass(frozen=True)
class Artwork:
    id: str
    created_at: str
    updated_at: str
    title: str | None
    topic_id: str | None
    style_id: str | None
    prompt: str
    params: dict[str, Any]
    seed: int | None
    dir: str
    is_draft: bool
    deleted_at: str | None
    source_id: str | None

    @property
    def version(self) -> str:
        """Cache-busting token for file URLs; changes on every update."""
        return _version(self.updated_at)

    def file_url(self, name: str, prefix: str = "/api/artworks") -> str:
        return f"{prefix}/{self.id}/{name}?v={self.version}"

    def public(self) -> dict[str, Any]:
        """What the kid page may see: no prompt, params or deletion state."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "topic": self.topic_id,
            "style": self.style_id,
            "picture": self.file_url("picture.jpg"),
            "sketch": self.file_url("sketch.png"),
            "thumb": self.file_url("thumb.jpg"),
        }

    def admin(self) -> dict[str, Any]:
        prefix = "/api/admin/artworks"
        return {
            **self.public(),
            "prompt": self.prompt,
            "params": self.params,
            "seed": self.seed,
            "is_draft": self.is_draft,
            "deleted_at": self.deleted_at,
            "picture": self.file_url("picture.jpg", prefix),
            "sketch": self.file_url("sketch.png", prefix),
            "thumb": self.file_url("thumb.jpg", prefix),
        }


def _row_to_artwork(row: tuple) -> Artwork:
    try:
        params = json.loads(row[7] or "{}")
    except ValueError:
        params = {}
    return Artwork(id=row[0], created_at=row[1], updated_at=row[2], title=row[3],
                   topic_id=row[4], style_id=row[5], prompt=row[6] or "",
                   params=params if isinstance(params, dict) else {}, seed=row[8],
                   dir=row[9], is_draft=bool(row[10]), deleted_at=row[11], source_id=row[12])


def _meta_values(meta: dict[str, Any]) -> dict[str, Any]:
    """Pick the known metadata fields (anything else is ignored)."""
    params = meta.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    seed = meta.get("seed", params.get("seed"))
    source = meta.get("source_id")
    return {
        "topic_id": meta.get("topic_id"),
        "style_id": meta.get("style_id"),
        "prompt": str(meta.get("prompt") or ""),
        "params_json": json.dumps(params, separators=(",", ":"), default=list),
        "seed": int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None,
        "source_id": source if valid_id(source) else None,
    }


def _check_images(sketch_png: bytes, picture_jpeg: bytes) -> None:
    if not isinstance(sketch_png, (bytes, bytearray)) or not sketch_png.startswith(PNG_MAGIC):
        raise ValueError("sketch must be a PNG")
    if not isinstance(picture_jpeg, (bytes, bytearray)) or not picture_jpeg.startswith(JPEG_MAGIC):
        raise ValueError("picture must be a JPEG")


def _replace(src: Path, dst: Path) -> None:
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write_temp(directory: Path, name: str, data: bytes) -> Path:
    """Write ``data`` to a fresh temp file next to ``name`` (fsynced); returns its path."""
    tmp = directory / f".{name}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return tmp


class GalleryStore:
    """Artworks: metadata rows in SQLite, files under ``artworks/``. Thread-safe."""

    def __init__(self, data_dir: str | os.PathLike, clock: Callable[[], str] = _now) -> None:
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.root / DB_FILE, check_same_thread=False,
                                     isolation_level=None, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    # ----- schema ----------------------------------------------------------------------

    def _migrate(self) -> None:
        with self._lock:
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if current > SCHEMA_VERSION:
                raise RuntimeError(f"{DB_FILE} is from a newer Dreamer (schema {current}, "
                                   f"this version knows {SCHEMA_VERSION})")
            for version, statements in MIGRATIONS:
                if version <= current:
                    continue
                with self._tx():
                    for sql in statements:
                        self._conn.execute(sql)
                    self._conn.execute(f"PRAGMA user_version = {int(version)}")
                log.info("gallery database migrated to schema %d", version)

    @property
    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("PRAGMA user_version").fetchone()[0]

    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- paths -----------------------------------------------------------------------

    def _dir(self, art: Artwork) -> Path:
        """The artwork's folder, re-derived and checked: never trust the stored string."""
        m = DIR_RE.fullmatch(art.dir)
        if not m or m.group(1) != art.id or not valid_id(art.id):
            raise ValueError(f"artwork {art.id!r} has a bad folder")
        path = (self.root / art.dir).resolve()
        if path.parent.parent.parent != self.root:
            raise ValueError(f"artwork {art.id!r} has a bad folder")
        return path

    def file_path(self, art: Artwork, name: str) -> Path:
        """Where ``name`` (picture.jpg / sketch.png / thumb.jpg) of this version lives."""
        if name not in FILES:
            raise KeyError(name)
        folder = self._dir(art)
        path = folder / _versioned(name, art.version)
        legacy = folder / name  # written before file names carried the version
        return legacy if not path.exists() and legacy.exists() else path

    def _write_files(self, folder: Path, version: str,
                     files: tuple[tuple[str, bytes], ...]) -> list[Path]:
        """Write one version's files (each atomically); on failure remove what was written."""
        written: list[Path] = []
        try:
            for name, data in files:
                dst = folder / _versioned(name, version)
                _replace(_write_temp(folder, name, data), dst)
                written.append(dst)
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        return written

    # ----- reads -----------------------------------------------------------------------

    def get(self, artwork_id: str, include_deleted: bool = False) -> Artwork | None:
        if not valid_id(artwork_id):
            return None
        sql = f"SELECT {_COLUMNS} FROM artworks WHERE id = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        with self._lock:
            row = self._conn.execute(sql, (artwork_id,)).fetchone()
        return _row_to_artwork(row) if row else None

    def _where(self, include_drafts: bool, deleted: bool) -> str:
        clauses = ["deleted_at IS NOT NULL" if deleted else "deleted_at IS NULL"]
        if not include_drafts:
            clauses.append("is_draft = 0")
        return " AND ".join(clauses)

    def list(self, limit: int = 24, offset: int = 0, include_drafts: bool = False,
             deleted: bool = False) -> list[Artwork]:
        """Newest (most recently updated) first. ``deleted=True`` lists the trash."""
        limit = max(1, min(int(limit), LIST_MAX))
        offset = max(0, int(offset))
        sql = (f"SELECT {_COLUMNS} FROM artworks WHERE {self._where(include_drafts, deleted)} "
               "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?")
        with self._lock:
            rows = self._conn.execute(sql, (limit, offset)).fetchall()
        return [_row_to_artwork(r) for r in rows]

    def count(self, include_drafts: bool = False, deleted: bool = False) -> int:
        sql = f"SELECT COUNT(*) FROM artworks WHERE {self._where(include_drafts, deleted)}"
        with self._lock:
            return self._conn.execute(sql).fetchone()[0]

    def drafts(self) -> list[Artwork]:
        """Live drafts, newest first."""
        sql = (f"SELECT {_COLUMNS} FROM artworks WHERE is_draft = 1 AND deleted_at IS NULL "
               "ORDER BY updated_at DESC, id DESC")
        with self._lock:
            return [_row_to_artwork(r) for r in self._conn.execute(sql).fetchall()]

    # ----- writes ----------------------------------------------------------------------

    def _new_id(self) -> str:
        while True:
            candidate = secrets.token_urlsafe(9)
            if self.get(candidate, include_deleted=True) is None:
                return candidate

    def save(self, sketch_png: bytes, picture_jpeg: bytes, meta: dict[str, Any] | None = None,
             is_draft: bool = False) -> Artwork:
        """Store a new artwork. Files first (atomically), then the row."""
        _check_images(sketch_png, picture_jpeg)
        thumb = make_thumb(picture_jpeg)
        values = _meta_values(meta or {})
        title = clean_title((meta or {}).get("title"))
        with self._lock:
            artwork_id = self._new_id()
            now = self._clock()
            rel = f"{ARTWORKS_DIR}/{now[:7]}/{artwork_id}"
            folder = self.root / rel
            folder.mkdir(parents=True, exist_ok=False)
            try:
                self._write_files(folder, _version(now), (
                    ("sketch.png", sketch_png), ("picture.jpg", picture_jpeg), ("thumb.jpg", thumb)))
                with self._tx() as db:
                    db.execute(
                        f"INSERT INTO artworks ({_COLUMNS}) VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                        (artwork_id, now, now, title, values["topic_id"], values["style_id"],
                         values["prompt"], values["params_json"], values["seed"], rel,
                         1 if is_draft else 0, values["source_id"]))
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        art = self.get(artwork_id)
        assert art is not None
        return art

    def update(self, artwork_id: str, sketch_png: bytes, picture_jpeg: bytes,
               meta: dict[str, Any] | None = None) -> Artwork:
        """Replace an artwork's images and metadata (keeps id, title and created_at).

        Raises KeyError if it does not exist (or is in the trash). The new version's files
        are written next to the old ones before the row points at them, and the old files
        are deleted only after the commit: a failure leaves the old row with its own files.
        """
        _check_images(sketch_png, picture_jpeg)
        thumb = make_thumb(picture_jpeg)
        values = _meta_values(meta or {})
        with self._lock:
            art = self.get(artwork_id)
            if art is None:
                raise KeyError(artwork_id)
            folder = self._dir(art)
            folder.mkdir(parents=True, exist_ok=True)
            old = [self.file_path(art, name) for name in FILES]
            now = max(self._clock(), art.updated_at)
            if now == art.updated_at:  # same clock tick: still bump the version
                now = _bump(now)
            written = self._write_files(folder, _version(now), (
                ("sketch.png", sketch_png), ("picture.jpg", picture_jpeg), ("thumb.jpg", thumb)))
            try:
                with self._tx() as db:
                    db.execute(
                        "UPDATE artworks SET updated_at = ?, topic_id = ?, style_id = ?, "
                        "prompt = ?, params_json = ?, seed = ?, source_id = ? WHERE id = ?",
                        (now, values["topic_id"], values["style_id"], values["prompt"],
                         values["params_json"], values["seed"], values["source_id"],
                         artwork_id))
            except BaseException:
                for path in written:
                    path.unlink(missing_ok=True)
                raise
            for path in old:  # committed: the old version is no longer referenced
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:  # e.g. still being served on Windows
                    log.warning("could not remove old file %s (sweep retries): %s", path, exc)
        art = self.get(artwork_id)
        assert art is not None
        return art

    def rename(self, artwork_id: str, title: Any) -> Artwork:
        title = clean_title(title)
        with self._tx() as db:
            cur = db.execute("UPDATE artworks SET title = ? WHERE id = ? AND is_draft = 0",
                             (title, artwork_id))
            if cur.rowcount == 0:
                raise KeyError(artwork_id)
        art = self.get(artwork_id, include_deleted=True)
        assert art is not None
        return art

    def soft_delete(self, artwork_id: str) -> Artwork:
        """Move to the trash (files stay until ``purge_deleted``). KeyError if not live."""
        if not valid_id(artwork_id):
            raise KeyError(artwork_id)
        with self._tx() as db:
            cur = db.execute("UPDATE artworks SET deleted_at = ? WHERE id = ? "
                             "AND deleted_at IS NULL", (self._clock(), artwork_id))
            if cur.rowcount == 0:
                raise KeyError(artwork_id)
        art = self.get(artwork_id, include_deleted=True)
        assert art is not None
        return art

    def restore(self, artwork_id: str) -> Artwork:
        """Take an artwork out of the trash. KeyError if it is not in the trash."""
        if not valid_id(artwork_id):
            raise KeyError(artwork_id)
        with self._tx() as db:
            cur = db.execute("UPDATE artworks SET deleted_at = NULL WHERE id = ? "
                             "AND deleted_at IS NOT NULL", (artwork_id,))
            if cur.rowcount == 0:
                raise KeyError(artwork_id)
        art = self.get(artwork_id)
        assert art is not None
        return art

    def _remove(self, art: Artwork) -> None:
        """Delete the row, then the files (a crash in between leaves only stray files)."""
        try:
            folder = self._dir(art)
        except ValueError:
            log.error("not deleting files of %s: bad folder %r", art.id, art.dir)
            folder = None
        with self._tx() as db:
            db.execute("DELETE FROM artworks WHERE id = ?", (art.id,))
        if folder is not None:
            _rmtree(folder)
            with contextlib.suppress(OSError):
                folder.parent.rmdir()  # the YYYY-MM folder, once empty

    def purge_deleted(self) -> int:
        """Empty the trash: rows and files of every soft-deleted artwork. Returns the count."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM artworks WHERE deleted_at IS NOT NULL").fetchall()
            for row in rows:
                self._remove(_row_to_artwork(row))
            self.sweep()  # also retries files an earlier delete could not remove
        return len(rows)

    def sweep(self) -> int:
        """Remove what no row refers to: folders of artworks that are gone (or were never
        committed), files of old versions, and temp files left by a crash. Returns how
        many folders and files were removed. Only names this store creates are touched."""
        removed = 0
        with self._lock:
            rows = [_row_to_artwork(r) for r in self._conn.execute(
                f"SELECT {_COLUMNS} FROM artworks").fetchall()]
            known = {art.dir: art for art in rows}
            base = self.root / ARTWORKS_DIR
            months = [p for p in base.iterdir() if p.is_dir() and MONTH_RE.fullmatch(p.name)] \
                if base.is_dir() else []
            for month in months:
                for folder in month.iterdir():
                    if not folder.is_dir() or not ID_RE.fullmatch(folder.name):
                        continue
                    art = known.get(f"{ARTWORKS_DIR}/{month.name}/{folder.name}")
                    if art is None:
                        log.info("removing orphaned artwork folder %s", folder)
                        removed += _rmtree(folder)
                        continue
                    try:
                        keep = {self.file_path(art, name).name for name in FILES}
                    except ValueError:
                        continue
                    for path in folder.iterdir():
                        if path.is_file() and path.name not in keep and _ours(path.name):
                            try:
                                path.unlink()
                                removed += 1
                            except OSError as exc:
                                log.warning("could not remove stray file %s: %s", path, exc)
                with contextlib.suppress(OSError):
                    month.rmdir()  # once empty
        return removed

    def delete_draft(self, artwork_id: str) -> bool:
        """Remove a draft for good (drafts never go to the trash)."""
        with self._lock:
            art = self.get(artwork_id, include_deleted=True)
            if art is None or not art.is_draft:
                return False
            self._remove(art)
            return True


_OURS_RE = re.compile(r"(sketch|picture|thumb)(\.\d+)?\.(png|jpg)|\.(sketch\.png|picture\.jpg|"
                      r"thumb\.jpg)\.[0-9a-f]{8}\.tmp")


def _ours(name: str) -> bool:
    """A file name this store writes (any version, or a temp file)."""
    return _OURS_RE.fullmatch(name) is not None


def _rmtree(folder: Path) -> int:
    """Remove a folder; failures are logged (sweep() retries later). 1 if it is gone."""
    def failed(_func: Any, path: str, exc: Any) -> None:
        log.warning("could not remove %s (sweep retries): %s", path, exc)
    if not folder.exists():
        return 1
    if sys.version_info >= (3, 12):
        shutil.rmtree(folder, onexc=failed)
    else:  # Python 3.10 on the PC
        shutil.rmtree(folder, onerror=lambda f, p, info: failed(f, p, info[1]))
    return 0 if folder.exists() else 1


def _bump(stamp: str) -> str:
    """The next microsecond after an ISO ``...%fZ`` timestamp."""
    dt = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return (dt + timedelta(microseconds=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ----- kid session glue (event loop) ----------------------------------------------------

@dataclass(frozen=True)
class LastFrame:
    """The newest picture shown to the kid, with exactly the sketch it was made from."""
    seq: int
    sketch_png: bytes
    picture_jpeg: bytes
    prompt: str
    topic_id: str | None
    style_id: str | None
    params: dict[str, Any]

    def meta(self, source_id: str | None = None) -> dict[str, Any]:
        return {"topic_id": self.topic_id, "style_id": self.style_id, "prompt": self.prompt,
                "params": self.params, "seed": self.params.get("seed"),
                "source_id": source_id}


class DraftAutosaver:
    """Keeps one draft artwork in step with the kid's drawing (event loop only).

    ``touch()`` after every new frame: the draft is written at most every ``interval``
    seconds while frames change, with a trailing write so the last change is never lost.
    ``flush()`` writes pending changes now; ``discard()`` drops the draft (clear/new).
    Blank frames are ignored, so they never replace a drawing with an empty canvas.
    All gallery writes from the kid socket share ``lock`` so they never interleave.
    """

    def __init__(self, store: GalleryStore, get_frame: Callable[[], LastFrame | None],
                 get_source: Callable[[], str | None], interval: float = 10.0) -> None:
        self.store = store
        self.interval = interval
        self.lock = asyncio.Lock()
        self._get_frame = get_frame
        self._get_source = get_source
        self._saved: LastFrame | None = None   # frame the draft currently holds
        self._last_write = 0.0
        self._task: asyncio.Task | None = None
        drafts = store.drafts()
        self.draft_id: str | None = drafts[0].id if drafts else None
        for extra in drafts[1:]:  # one draft per session; older ones are leftovers
            store.delete_draft(extra.id)

    @property
    def dirty(self) -> bool:
        frame = self._get_frame()
        return frame is not None and frame is not self._saved

    def touch(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._later())

    async def _later(self) -> None:
        wait = self._last_write + self.interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        # From here on the write must not be cancelled half-way (the row would be saved
        # but its id lost), so detach from ``_task``: _cancel() only stops a sleeping timer.
        if self._task is asyncio.current_task():
            self._task = None
        await self._write()

    def _cancel(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def flush(self) -> None:
        """Write pending changes now. Shielded: a socket handler being cancelled (client
        gone) must not abandon a write half-way."""
        self._cancel()
        await asyncio.shield(self._write())

    async def _write(self) -> None:
        async with self.lock:
            frame = self._get_frame()
            if frame is None or frame is self._saved:
                return
            self._last_write = time.monotonic()
            try:
                # A blank canvas never overwrites the draft: a freshly loaded page shows an
                # empty canvas for a moment before it restores the draft. Only an explicit
                # clear / new drawing drops it (discard()).
                if not await asyncio.to_thread(is_blank_sketch, frame.sketch_png):
                    await self._store(frame)
            except (OSError, ValueError, sqlite3.Error) as exc:
                log.error("could not autosave the draft: %s", exc)
                return  # not marked saved: the next frame or flush tries again
            self._saved = frame

    async def _store(self, frame: LastFrame) -> None:
        meta = frame.meta(self._get_source())
        if self.draft_id is not None:
            try:
                await asyncio.to_thread(self.store.update, self.draft_id,
                                        frame.sketch_png, frame.picture_jpeg, meta)
                return
            except KeyError:
                self.draft_id = None
        art = await asyncio.to_thread(self.store.save, frame.sketch_png,
                                      frame.picture_jpeg, meta, True)
        self.draft_id = art.id

    async def _drop(self) -> None:
        draft_id, self.draft_id = self.draft_id, None
        if draft_id is not None:
            try:
                await asyncio.to_thread(self.store.delete_draft, draft_id)
            except (OSError, sqlite3.Error) as exc:
                log.error("could not remove the draft: %s", exc)

    async def discard(self) -> None:
        """The canvas was cleared / a new drawing started: forget the draft."""
        self._cancel()
        async with self.lock:
            self._saved = self._get_frame()
            await self._drop()

    def current(self) -> Artwork | None:
        if self.draft_id is None:
            return None
        art = self.store.get(self.draft_id)
        return art if art is not None and art.is_draft else None
