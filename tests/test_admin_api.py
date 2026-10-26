"""Admin HTTP API: state, style CRUD, ordering, selection push, upload screening."""
import io

from PIL import Image

from conftest import is_json, recv_until
from server.styles import SAFETY_TAIL


def png(color, size=(64, 64), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def upload(client, data, label="Quilt", ctype="image/png", emoji="🧵"):
    return client.post("/api/admin/style-image", data={"label": label, "emoji": emoji},
                       files={"file": ("pic.png", data, ctype)})


def kid(client):
    ws = client.websocket_connect("/ws")
    ws.__enter__()
    ws.send_json({"t": "hello", "v": 1})
    recv_until(ws, is_json("ready"))
    return ws


def test_state(admin):
    state = admin.get("/api/admin/state").json()
    assert {"topics", "styles", "topic", "style", "view_mode", "backend", "stats"} <= set(state)
    assert any(s["prompt"] for s in state["styles"])  # admin sees prompts


def test_select_pushes_styles_to_kid(admin):
    ws = kid(admin)
    try:
        r = admin.post("/api/admin/select", json={"topic": "space", "style": "clay",
                                                  "view_mode": "split"})
        assert r.status_code == 200
        msg = recv_until(ws, is_json("styles"))[1]
        assert msg["topic"] == "space" and msg["style"] == "clay" and msg["view_mode"] == "split"
        assert "v" not in msg and msg["backend"] == "mock"
    finally:
        ws.__exit__(None, None, None)


def test_select_validation(admin):
    assert admin.post("/api/admin/select", json={"topic": "nope"}).status_code == 400
    assert admin.post("/api/admin/select", json={"view_mode": "huge"}).status_code == 400


def test_custom_style_appears_on_kid_socket(admin):
    ws = kid(admin)
    try:
        r = admin.post("/api/admin/styles",
                       json={"label": "Rainbow", "emoji": "🌈", "prompt": "rainbow colors"})
        assert r.status_code == 200
        style = r.json()["style"]
        msg = recv_until(ws, is_json("styles"))[1]
        assert style["id"] in [s["id"] for s in msg["styles"]]
        ws.send_json({"t": "style", "id": style["id"]})
        ws.send_json({"t": "ping"})
        recv_until(ws, is_json("pong"))
        session = admin.app.state.session
        cur = session.current()
        prompt = session.styles.build_prompt(cur["topic"], cur["style"])
        assert "rainbow colors" in prompt and prompt.endswith(SAFETY_TAIL)
    finally:
        ws.__exit__(None, None, None)


def test_custom_style_validation(admin):
    r = admin.post("/api/admin/styles", json={"label": "X", "prompt": "a" * 200})
    assert r.status_code == 400
    r = admin.post("/api/admin/styles", json={"label": "", "prompt": "ok"})
    assert r.status_code == 400


def test_patch_hide_order_delete(admin):
    assert admin.patch("/api/admin/items/topics/winter", json={"hidden": True}).status_code == 200
    assert admin.patch("/api/admin/items/topics/nope", json={"hidden": True}).status_code == 404
    assert admin.patch("/api/admin/items/bogus/x", json={"hidden": True}).status_code == 404
    assert admin.patch("/api/admin/items/styles/cartoon",
                       json={"prompt": "nope"}).status_code == 400
    r = admin.patch("/api/admin/items/styles/cartoon", json={"label": "Toons", "emoji": "📺"})
    assert r.json()["item"]["label"] == "Toons"

    assert admin.post("/api/admin/order",
                      json={"kind": "styles", "ids": ["clay", "cartoon"]}).status_code == 200
    state = admin.get("/api/admin/state").json()
    assert [s["id"] for s in state["styles"]][:2] == ["clay", "cartoon"]
    assert next(t for t in state["topics"] if t["id"] == "winter")["hidden"] is True
    assert admin.post("/api/admin/order",
                      json={"kind": "styles", "ids": ["nope"]}).status_code == 404

    sid = admin.post("/api/admin/styles", json={"label": "Tmp", "prompt": "tmp"}).json()["style"]["id"]
    assert admin.delete("/api/admin/items/styles/cartoon").status_code == 400
    assert admin.delete(f"/api/admin/items/styles/{sid}").status_code == 200
    assert admin.delete(f"/api/admin/items/styles/{sid}").status_code == 404


def test_upload_accepted_and_stored(admin, config):
    r = upload(admin, png((10, 200, 30)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["style"]["kind"] == "image"
    assert "phase 2" in body["note"]
    from pathlib import Path
    stored = Path(config.data_dir) / body["style"]["image"]
    assert stored.is_file()
    # Deleting the style removes the stored picture too.
    assert admin.delete(f"/api/admin/items/styles/{body['style']['id']}").status_code == 200
    assert not stored.exists()


def test_upload_rejected_by_screening(admin):
    r = upload(admin, png((255, 0, 255)))
    assert r.status_code == 422
    names = [s["label"] for s in admin.get("/api/admin/state").json()["styles"]]
    assert "Quilt" not in names


def test_upload_rejects_non_images(admin):
    assert upload(admin, b"hello", ctype="text/plain").status_code == 415
    assert upload(admin, b"not really a png").status_code == 400
    big = b"\x00" * (8 * 1024 * 1024 + 1)
    assert upload(admin, big).status_code == 413


def test_upload_applies_exif_orientation(admin, config):
    from pathlib import Path

    img = Image.new("RGB", (80, 40), (10, 120, 200))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    r = upload(admin, buf.getvalue(), ctype="image/jpeg")
    assert r.status_code == 200, r.text
    stored = Image.open(Path(config.data_dir) / r.json()["style"]["image"])
    assert stored.size == (40, 80)  # stored upright


def test_upload_rejects_exotic_formats(admin):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buf, "TIFF")
    assert upload(admin, buf.getvalue(), ctype="image/tiff").status_code == 400


def test_admin_adds_and_deletes_a_topic(admin):
    res = admin.post("/api/admin/topics", json={"label": "Pirates", "emoji": "🏴‍☠️",
                                                 "prompt": "a pirate ship on a sunny sea"})
    assert res.status_code == 200, res.text
    topic_id = res.json()["topic"]["id"]
    assert topic_id in [item["id"] for item in admin.get("/api/admin/state").json()["topics"]]
    assert admin.delete(f"/api/admin/items/topics/{topic_id}").status_code == 200
    assert topic_id not in [item["id"] for item in admin.get("/api/admin/state").json()["topics"]]
    assert admin.delete("/api/admin/items/topics/ocean").status_code == 400  # built-in: hide only
