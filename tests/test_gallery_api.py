"""Gallery over the kid socket and HTTP: save / new / continue, drafts, admin endpoints."""
import io
import time

import pytest
from PIL import Image

from conftest import is_json, is_picture, recv_until, sketch_frame, sketch_png
from server import protocol


def hello(ws):
    ws.send_json({"t": "hello", "v": 1})
    return recv_until(ws, is_json("ready"))[1]


def draw(ws, seq, blank=False):
    """Send a sketch and wait for its picture (last_frame is set by then)."""
    ws.send_bytes(sketch_frame(seq, blank))
    return recv_until(ws, is_picture(seq))[1][2]


def save(ws):
    ws.send_json({"t": "save"})
    return recv_until(ws, lambda k, v: k == "json" and (
        v.get("t") == "saved" or (v.get("t") == "status" and v.get("state") == "error")))[1]


def is_error(kind, value):
    return kind == "json" and value.get("t") == "status" and value.get("state") == "error"


@pytest.fixture
def fast_drafts(client):
    """Autosave drafts almost immediately instead of every 10 s."""
    client.app.state.runtime.drafts.interval = 0.05
    return client


def wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition never became true")


# ----- save ------------------------------------------------------------------------------

def test_save_without_frame_is_an_error(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        msg = save(ws)
    assert msg == {"t": "status", "state": "error", "msg": "Nothing to save yet"}
    assert client.get("/api/artworks").json()["artworks"] == []


def test_save_blank_canvas_is_an_error(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1, blank=True)
        assert save(ws)["msg"] == "Nothing to save yet"


def test_save_stores_exactly_what_is_on_screen(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "topic", "id": "space"})
        picture = draw(ws, 3)
        msg = save(ws)
    assert msg["t"] == "saved"
    art = msg["artwork"]
    assert art["thumb"].startswith(f"/api/artworks/{art['id']}/thumb.jpg?v=")
    assert art["topic"] == "space" and art["style"] == "cartoon"
    assert "prompt" not in art and "params" not in art

    listing = client.get("/api/artworks").json()
    assert [a["id"] for a in listing["artworks"]] == [art["id"]] and listing["more"] is False
    assert client.get(f"/api/artworks/{art['id']}").json() == art

    r = client.get(art["picture"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content == picture  # no regeneration: the very JPEG the kid saw
    assert "immutable" in r.headers["cache-control"]
    r = client.get(art["sketch"])
    assert r.headers["content-type"] == "image/png" and r.content == sketch_png()
    r = client.get(art["thumb"])
    assert r.headers["content-type"] == "image/jpeg"
    assert max(Image.open(io.BytesIO(r.content)).size) <= 256
    r = client.get(f"/api/artworks/{art['id']}/picture.jpg")  # unversioned URL
    assert r.headers["cache-control"] == "no-cache"

    stored = client.app.state.gallery.get(art["id"])
    assert "space" in stored.prompt and stored.params["seed"] == stored.seed


def test_saving_again_updates_the_same_artwork(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1)
        first = save(ws)["artwork"]
        draw(ws, 2)
        second = save(ws)["artwork"]
        assert second["id"] == first["id"] and second["updated_at"] > first["updated_at"]
        assert second["thumb"] != first["thumb"]  # cache-busting version changed
        ws.send_json({"t": "new"})
        draw(ws, 3)
        third = save(ws)["artwork"]
    assert third["id"] != first["id"]
    ids = [a["id"] for a in client.get("/api/artworks").json()["artworks"]]
    assert ids == [third["id"], first["id"]]


def test_clear_starts_a_new_artwork(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1)
        first = save(ws)["artwork"]
        ws.send_json({"t": "clear"})
        assert save(ws)["msg"] == "Nothing to save yet"  # cleared: nothing on screen
        draw(ws, 2)
        assert save(ws)["artwork"]["id"] != first["id"]


def test_save_error_restores_status(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        # The worker may still be warming up; the status after the error is the current one.
        recv_until(ws, lambda k, v: k == "json" and v.get("t") == "status" and v.get("state") == "ready")
        ws.send_json({"t": "save"})
        recv_until(ws, is_error)
        follow = recv_until(ws, is_json("status"))[1]
    assert follow["state"] == "ready"  # the page's connection dot goes back to normal


def test_gallery_paging(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        for seq in range(1, 5):
            ws.send_json({"t": "new"})
            draw(ws, seq)
            save(ws)
    page1 = client.get("/api/artworks?limit=3").json()
    page2 = client.get("/api/artworks?limit=3&offset=3").json()
    assert len(page1["artworks"]) == 3 and page1["more"] is True
    assert len(page2["artworks"]) == 1 and page2["more"] is False
    assert client.get("/api/artworks?limit=0&offset=-5").status_code == 200  # clamped


# ----- continue / new --------------------------------------------------------------------

def test_continue_restores_and_updates(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "topic", "id": "dinosaurs"})
        ws.send_json({"t": "style", "id": "clay"})
        draw(ws, 1)
        art = save(ws)["artwork"]
        ws.send_json({"t": "new"})
        ws.send_json({"t": "topic", "id": "ocean"})
        ws.send_json({"t": "continue", "id": art["id"]})
        msg = recv_until(ws, is_json("continue"))[1]
        assert msg == {"t": "continue", "id": art["id"],
                       "sketch": art["sketch"], "topic": "dinosaurs", "style": "clay"}
        assert client.get(msg["sketch"]).content == sketch_png()
        assert save(ws)["msg"] == "Nothing to save yet"  # nothing drawn since continuing
        draw(ws, 2)  # the page sends the restored sketch (plus new strokes)
        again = save(ws)["artwork"]
    assert again["id"] == art["id"]
    assert len(client.get("/api/artworks").json()["artworks"]) == 1


@pytest.mark.parametrize("bad", ["../../../etc", "A" * 12, 42, None, "../" * 4])
def test_continue_unknown_id(client, bad):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "continue", "id": bad})
        msg = recv_until(ws, is_error)[1]
    assert "not there" in msg["msg"]
    assert client.app.state.session.continuing_id is None


def test_new_clears_continuing(client):
    session = client.app.state.session
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1)
        art = save(ws)["artwork"]
        assert session.continuing_id == art["id"]
        ws.send_json({"t": "new"})
        ws.send_json({"t": "ping"})
        recv_until(ws, is_json("pong"))
        assert session.continuing_id is None and session.last_frame is None


def test_continuing_a_deleted_artwork_saves_a_new_one(admin):
    with admin.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1)
        art = save(ws)["artwork"]
        assert admin.delete(f"/api/admin/artworks/{art['id']}").status_code == 200
        draw(ws, 2)
        again = save(ws)["artwork"]
    assert again["id"] != art["id"]


# ----- drafts ----------------------------------------------------------------------------

def test_draft_autosave_and_restore_on_reconnect(fast_drafts):
    client = fast_drafts
    store = client.app.state.gallery
    with client.websocket_connect("/ws") as ws:
        assert hello(ws)["draft"] is None
        draw(ws, 1)
        wait_for(lambda: len(store.drafts()) == 1)
        draw(ws, 2)
        time.sleep(0.2)
        assert len(store.drafts()) == 1  # one draft, overwritten
    assert client.get("/api/artworks").json()["artworks"] == []  # drafts are not listed
    with client.websocket_connect("/ws") as ws:
        draft = hello(ws)["draft"]
    assert draft["id"] == store.drafts()[0].id
    r = client.get(draft["sketch"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content == sketch_png()
    assert client.get(f"/api/artworks/{draft['id']}").status_code == 404  # not an artwork


def test_draft_flushed_when_page_reloads(client):
    """Even with the 10 s throttle, a reload gets the newest drawing as its draft."""
    client.app.state.runtime.drafts.interval = 3600
    store = client.app.state.gallery
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1)
        wait_for(lambda: len(store.drafts()) == 1)  # first write is immediate
        first = store.drafts()[0].updated_at
        draw(ws, 2)
    with client.websocket_connect("/ws") as ws:
        draft = hello(ws)["draft"]
    assert [d.id for d in store.drafts()] == [draft["id"]]
    assert store.get(draft["id"]).updated_at > first


def test_clear_and_new_drop_the_draft(fast_drafts):
    client = fast_drafts
    store = client.app.state.gallery
    for msg in ("clear", "new"):
        with client.websocket_connect("/ws") as ws:
            hello(ws)
            draw(ws, 1)
            wait_for(lambda: len(store.drafts()) == 1)
            folder = store.root / store.drafts()[0].dir
            ws.send_json({"t": msg})
            ws.send_json({"t": "ping"})
            recv_until(ws, is_json("pong"))
            assert store.drafts() == [] and not folder.exists()
        with client.websocket_connect("/ws") as ws:
            assert hello(ws)["draft"] is None


def test_blank_canvas_keeps_the_draft(fast_drafts):
    """A freshly loaded page sends its empty canvas before restoring the draft: that blank
    frame must not overwrite (or drop) the draft. Only clear/new do."""
    store = fast_drafts.app.state.gallery
    with fast_drafts.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1)
        wait_for(lambda: len(store.drafts()) == 1)
        before = store.drafts()[0]
    with fast_drafts.websocket_connect("/ws") as ws:
        draft = hello(ws)["draft"]
        draw(ws, 1, blank=True)
        time.sleep(0.3)
        assert store.drafts() == [before]
        assert fast_drafts.get(draft["sketch"]).content == sketch_png()


def test_draft_survives_restart_and_remembers_continuing(make_client, config):
    with make_client() as client:
        client.app.state.runtime.drafts.interval = 0.05
        with client.websocket_connect("/ws") as ws:
            hello(ws)
            draw(ws, 1)
            art = save(ws)["artwork"]
            draw(ws, 2)
    # A new app on the same data dir (server restart): draft offered, continuing restored.
    with make_client() as client:
        with client.websocket_connect("/ws") as ws:
            draft = hello(ws)["draft"]
            assert draft is not None
            assert client.app.state.session.continuing_id == art["id"]
            draw(ws, 3)
            assert save(ws)["artwork"]["id"] == art["id"]


# ----- admin endpoints -------------------------------------------------------------------

def make_artwork(client) -> dict:
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "new"})
        draw(ws, 1)
        return save(ws)["artwork"]


def test_admin_endpoints_need_cookie(client):
    art = make_artwork(client)
    aid = art["id"]
    for method, path in [("GET", "/api/admin/artworks"),
                         ("GET", f"/api/admin/artworks/{aid}/picture.jpg"),
                         ("DELETE", f"/api/admin/artworks/{aid}"),
                         ("POST", f"/api/admin/artworks/{aid}/restore"),
                         ("POST", "/api/admin/artworks/purge"),
                         ("PATCH", f"/api/admin/artworks/{aid}")]:
        r = client.request(method, path, json={"title": "x"} if method == "PATCH" else None)
        assert r.status_code == 401, (method, path)
    # Kids have no delete of their own either.
    assert client.delete(f"/api/artworks/{aid}").status_code == 405
    assert [a["id"] for a in client.get("/api/artworks").json()["artworks"]] == [aid]


def test_admin_delete_restore_purge(admin):
    art = make_artwork(admin)
    aid = art["id"]
    listing = admin.get("/api/admin/artworks").json()
    assert [a["id"] for a in listing["artworks"]] == [aid]
    assert listing["total"] == 1 and listing["trash"] == 0
    assert listing["artworks"][0]["prompt"]  # admin sees the prompt

    assert admin.delete(f"/api/admin/artworks/{aid}").status_code == 200
    assert admin.get("/api/artworks").json()["artworks"] == []
    assert admin.get(f"/api/artworks/{aid}").status_code == 404
    assert admin.get(art["picture"]).status_code == 404  # kid URLs stop working
    trash = admin.get("/api/admin/artworks?trash=1").json()
    assert [a["id"] for a in trash["artworks"]] == [aid] and trash["trash"] == 1
    r = admin.get(trash["artworks"][0]["thumb"])  # admin can still see it in the trash
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert admin.delete(f"/api/admin/artworks/{aid}").status_code == 404

    assert admin.post(f"/api/admin/artworks/{aid}/restore").status_code == 200
    assert [a["id"] for a in admin.get("/api/artworks").json()["artworks"]] == [aid]

    admin.delete(f"/api/admin/artworks/{aid}")
    r = admin.post("/api/admin/artworks/purge")
    assert r.json() == {"ok": True, "purged": 1}
    assert admin.get("/api/admin/artworks?trash=1").json()["artworks"] == []
    assert admin.post(f"/api/admin/artworks/{aid}/restore").status_code == 404


def test_admin_delete_notifies_kid(admin):
    art = make_artwork(admin)
    with admin.websocket_connect("/ws") as ws:
        hello(ws)
        admin.delete(f"/api/admin/artworks/{art['id']}")
        assert recv_until(ws, is_json("gallery"))[1] == {"t": "gallery"}


def test_admin_rename_and_download(admin):
    art = make_artwork(admin)
    r = admin.patch(f"/api/admin/artworks/{art['id']}", json={"title": "  Dino   party "})
    assert r.status_code == 200 and r.json()["artwork"]["title"] == "Dino party"
    assert admin.get(f"/api/artworks/{art['id']}").json()["title"] == "Dino party"
    assert admin.patch(f"/api/admin/artworks/{art['id']}",
                       json={"title": "x" * 200}).status_code == 400
    assert admin.patch("/api/admin/artworks/AAAAAAAAAAAA", json={"title": "x"}).status_code == 404
    r = admin.get(f"/api/admin/artworks/{art['id']}/sketch.png?download=1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "attachment" in r.headers["content-disposition"]
    assert "dreamer-dino-party-sketch.png" in r.headers["content-disposition"]


@pytest.mark.parametrize("path", [
    "/api/artworks/..%2F..%2Fsecret/picture.jpg",
    "/api/artworks/AAAAAAAAAAAA/picture.jpg",
    "/api/artworks/{id}/secret",
    "/api/artworks/{id}/..%2F..%2F..%2Fsecret",
    "/api/artworks/{id}/sketch.jpg",
    "/api/artworks/..",
])
def test_http_rejects_bad_ids_and_names(client, path):
    art = make_artwork(client)
    r = client.get(path.replace("{id}", art["id"]))
    assert r.status_code == 404


def test_saved_message_reaches_admin_watchers(admin):
    with admin.websocket_connect("/ws/admin") as watch, admin.websocket_connect("/ws") as ws:
        hello(ws)
        draw(ws, 1)
        save(ws)
        assert recv_until(watch, is_json("gallery"))[1] == {"t": "gallery"}


def test_frames_from_replaced_socket_are_not_saved(make_client):
    with make_client(delay_ms=400) as client, client.websocket_connect("/ws") as old:
        hello(old)
        old.send_bytes(protocol.pack(protocol.SKETCH, 9, sketch_png()))
        time.sleep(0.15)  # worker is busy with the old page's frame
        with client.websocket_connect("/ws") as new:
            hello(new)
            time.sleep(0.5)
            assert client.app.state.session.last_frame is None
            assert save(new)["msg"] == "Nothing to save yet"
