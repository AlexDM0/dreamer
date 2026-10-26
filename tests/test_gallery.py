"""GalleryStore: CRUD, atomic writes, soft delete / restore / purge, ids and paths."""
import io
import sqlite3

import pytest
from PIL import Image

from conftest import sketch_png
from server import gallery as g
from server.gallery import GalleryStore, valid_id


def jpeg(color="red", size=(256, 256)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


META = {"topic_id": "space", "style_id": "crayon", "prompt": "a rocket",
        "params": {"seed": 7, "t_index_list": [30, 42]}}


@pytest.fixture
def store(tmp_path):
    s = GalleryStore(tmp_path / "data")
    yield s
    s.close()


def artwork_dirs(store):
    root = store.root / g.ARTWORKS_DIR
    return [p for p in root.glob("*/*") if p.is_dir()] if root.exists() else []


def test_save_get_list(store):
    art = store.save(sketch_png(), jpeg(), META)
    assert valid_id(art.id) and not art.is_draft and art.deleted_at is None
    assert art.topic_id == "space" and art.style_id == "crayon" and art.prompt == "a rocket"
    assert art.seed == 7 and art.params["t_index_list"] == [30, 42]
    assert art.dir == f"artworks/{art.created_at[:7]}/{art.id}"
    folder = store.root / art.dir
    assert store.file_path(art, "sketch.png").parent == folder
    assert store.file_path(art, "sketch.png").read_bytes() == sketch_png()  # lossless
    assert store.file_path(art, "picture.jpg").read_bytes() == jpeg()
    thumb = Image.open(store.file_path(art, "thumb.jpg"))
    assert thumb.format == "JPEG" and max(thumb.size) <= g.THUMB_SIZE
    assert not list(folder.glob("*.tmp"))
    assert store.get(art.id) == art
    assert [a.id for a in store.list()] == [art.id]
    assert store.count() == 1


def test_list_newest_first_and_paging(store):
    ids = [store.save(sketch_png(), jpeg(), META).id for _ in range(5)]
    assert [a.id for a in store.list(limit=2)] == ids[::-1][:2]
    assert [a.id for a in store.list(limit=2, offset=2)] == ids[::-1][2:4]
    assert [a.id for a in store.list(limit=10, offset=4)] == [ids[0]]


def test_public_fields_hide_internals(store):
    art = store.save(sketch_png(), jpeg(), META)
    pub = art.public()
    assert pub["thumb"] == f"/api/artworks/{art.id}/thumb.jpg?v={art.version}"
    assert not {"prompt", "params", "seed", "dir", "deleted_at"} & set(pub)
    assert art.admin()["prompt"] == "a rocket"


def test_update_replaces_files_and_bumps_version(store):
    art = store.save(sketch_png(), jpeg("red"), META)
    store.rename(art.id, "My rocket")
    new_sketch = sketch_png(color="blue")
    updated = store.update(art.id, new_sketch, jpeg("blue"), {**META, "topic_id": "ocean"})
    assert updated.id == art.id and updated.created_at == art.created_at
    assert updated.title == "My rocket" and updated.topic_id == "ocean"
    assert updated.updated_at > art.updated_at and updated.version != art.version
    assert store.file_path(updated, "sketch.png").read_bytes() == new_sketch
    assert store.file_path(updated, "sketch.png") != store.file_path(art, "sketch.png")
    # Only the new version's three files are left.
    assert sorted(p.name for p in (store.root / art.dir).iterdir()) == sorted(
        store.file_path(updated, n).name for n in g.FILES)
    assert store.count() == 1


def test_update_file_failure_keeps_old_version_intact(store, monkeypatch):
    """A failed write half-way (e.g. a Windows file lock) leaves the old row with exactly
    its old files: nothing served under the old ?v ever changes."""
    art = store.save(sketch_png(), jpeg("red"), META)
    folder = store.root / art.dir
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    real = g.replace_with_retry
    calls = []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise PermissionError("in use")
        real(src, dst)
    monkeypatch.setattr(g, "replace_with_retry", flaky)
    with pytest.raises(PermissionError):
        store.update(art.id, sketch_png(color="blue"), jpeg("blue"), META)
    monkeypatch.undo()
    store.sweep()  # the temp file of the failed write
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert store.get(art.id) == art


def test_legacy_unversioned_files_are_read_and_replaced(store):
    art = store.save(sketch_png(), jpeg("red"), META)
    folder = store.root / art.dir
    for name in g.FILES:  # a folder written before file names carried the version
        store.file_path(art, name).rename(folder / name)
    assert store.file_path(art, "picture.jpg") == folder / "picture.jpg"
    assert store.sweep() == 0  # legacy files are the current ones: kept
    updated = store.update(art.id, sketch_png(color="blue"), jpeg("blue"), META)
    assert not (folder / "picture.jpg").exists()
    assert store.file_path(updated, "sketch.png").read_bytes() == sketch_png(color="blue")


def test_sweep_removes_orphans_old_versions_and_temp_files(store):
    keep = store.save(sketch_png(), jpeg(), META)
    folder = store.root / keep.dir
    orphan = store.root / "artworks" / "2020-01" / ("Z" * 12)  # files without a row
    orphan.mkdir(parents=True)
    (orphan / "picture.1.jpg").write_bytes(b"x")
    (folder / ".picture.jpg.0123abcd.tmp").write_bytes(b"x")   # crash during a write
    (folder / "picture.20200101000000000000.jpg").write_bytes(b"x")  # old version
    (folder / "notes.txt").write_bytes(b"not ours")               # never touched
    assert store.sweep() == 3
    assert not orphan.parent.exists()
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [store.file_path(keep, n).name for n in g.FILES] + ["notes.txt"])
    assert store.file_path(keep, "picture.jpg").read_bytes() == jpeg()


def test_update_missing_or_deleted_raises(store):
    with pytest.raises(KeyError):
        store.update("A" * 12, sketch_png(), jpeg())
    art = store.save(sketch_png(), jpeg(), META)
    store.soft_delete(art.id)
    with pytest.raises(KeyError):
        store.update(art.id, sketch_png(), jpeg())


def test_rejects_non_images(store):
    with pytest.raises(ValueError):
        store.save(b"not a png", jpeg())
    with pytest.raises(ValueError):
        store.save(sketch_png(), b"\x89PNG not a jpeg")
    assert store.count() == 0 and artwork_dirs(store) == []


def test_db_failure_on_save_removes_files(store):
    store._conn.execute("CREATE TEMP TRIGGER boom BEFORE INSERT ON artworks "
                        "BEGIN SELECT RAISE(ABORT, 'disk on fire'); END")
    with pytest.raises(sqlite3.Error):
        store.save(sketch_png(), jpeg(), META)
    assert artwork_dirs(store) == []
    assert store.count(include_drafts=True) == 0


def test_db_failure_on_update_keeps_old_files(store):
    art = store.save(sketch_png(), jpeg("red"), META)
    folder = store.root / art.dir
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    store._conn.execute("CREATE TEMP TRIGGER boom BEFORE UPDATE ON artworks "
                        "BEGIN SELECT RAISE(ABORT, 'disk on fire'); END")
    with pytest.raises(sqlite3.Error):
        store.update(art.id, sketch_png(color="blue"), jpeg("blue"), META)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before  # no temp files left
    assert store.get(art.id) == art


def test_soft_delete_restore_purge(store):
    keep = store.save(sketch_png(), jpeg(), META)
    gone = store.save(sketch_png(), jpeg(), META)
    store.soft_delete(gone.id)
    assert store.get(gone.id) is None
    assert store.get(gone.id, include_deleted=True).deleted_at is not None
    assert [a.id for a in store.list()] == [keep.id]
    assert [a.id for a in store.list(deleted=True)] == [gone.id]
    with pytest.raises(KeyError):
        store.soft_delete(gone.id)  # already in the trash
    store.restore(gone.id)
    assert store.count() == 2
    with pytest.raises(KeyError):
        store.restore(gone.id)  # not in the trash
    store.soft_delete(gone.id)
    gone_dir = store.root / gone.dir
    assert gone_dir.is_dir()  # soft delete keeps files
    assert store.purge_deleted() == 1
    assert not gone_dir.exists()
    assert store.get(gone.id, include_deleted=True) is None
    assert store.file_path(keep, "picture.jpg").is_file()
    assert store.purge_deleted() == 0


def test_drafts_are_separate(store):
    draft = store.save(sketch_png(), jpeg(), META, is_draft=True)
    art = store.save(sketch_png(), jpeg(), META)
    assert [a.id for a in store.list()] == [art.id]
    assert {a.id for a in store.list(include_drafts=True)} == {art.id, draft.id}
    assert [a.id for a in store.drafts()] == [draft.id]
    assert not store.delete_draft(art.id)  # real artworks are not drafts
    assert store.delete_draft(draft.id)
    assert not (store.root / draft.dir).exists()
    assert store.drafts() == []


@pytest.mark.parametrize("bad", ["", "short", "A" * 13, "../../etc/pw", "abc/def/ghi.", "A" * 11 + ".",
                                 None, 123, "A" * 11 + "\n"])
def test_id_validation(store, bad):
    assert not valid_id(bad)
    assert store.get(bad) is None
    with pytest.raises(KeyError):
        store.soft_delete(bad)
    with pytest.raises(KeyError):
        store.restore(bad)


def test_tampered_dir_is_never_used(store, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "picture.jpg").write_bytes(b"keep me")
    art = store.save(sketch_png(), jpeg(), META)
    for bad_dir in ["../victim", f"artworks/2026-10/../../../victim", f"artworks/x/{art.id}",
                    "artworks/2026-10/AAAAAAAAAAAA"]:
        store._conn.execute("UPDATE artworks SET dir = ? WHERE id = ?", (bad_dir, art.id))
        tampered = store.get(art.id)
        with pytest.raises(ValueError):
            store.file_path(tampered, "picture.jpg")
    with pytest.raises(KeyError):
        store.file_path(art, "../../secret")
    store.soft_delete(art.id)
    store.purge_deleted()
    assert (victim / "picture.jpg").read_bytes() == b"keep me"


def test_schema_version_and_reopen(tmp_path):
    s = GalleryStore(tmp_path)
    assert s.schema_version == g.SCHEMA_VERSION
    art = s.save(sketch_png(), jpeg(), META)
    mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    s.close()
    s2 = GalleryStore(tmp_path)
    assert s2.get(art.id) == art
    s2._conn.execute(f"PRAGMA user_version = {g.SCHEMA_VERSION + 1}")
    s2.close()
    with pytest.raises(RuntimeError):
        GalleryStore(tmp_path)


def test_rename_and_title_cleaning(store):
    art = store.save(sketch_png(), jpeg(), META)
    assert store.rename(art.id, "  Big   red  rocket ").title == "Big red rocket"
    assert store.rename(art.id, "   ").title is None
    with pytest.raises(ValueError):
        store.rename(art.id, "x" * (g.TITLE_MAX + 1))
    with pytest.raises(ValueError):
        store.rename(art.id, "bad\x00title")
    with pytest.raises(KeyError):
        store.rename("A" * 12, "nope")


def test_is_blank_sketch():
    assert g.is_blank_sketch(sketch_png(blank=True))
    assert not g.is_blank_sketch(sketch_png())
    assert g.is_blank_sketch(b"garbage")


def test_list_gives_one_row_more_than_a_full_page(store, monkeypatch):
    """The API asks for limit + 1 rows to know whether there are more: at a full page of
    LIST_MAX that extra row must still come back."""
    monkeypatch.setattr(g, "LIST_MAX", 3)  # instead of saving 100+ artworks
    for _ in range(5):
        store.save(sketch_png(), jpeg("red", (8, 8)), META)
    assert len(store.list(4)) == 4
    assert len(store.list(50)) == 4
