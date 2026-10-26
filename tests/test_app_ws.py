"""Kid WebSocket end-to-end against the mock backend."""
import io
import time

from conftest import current_prompt, is_json, is_picture, recv, recv_until, sketch_frame
from PIL import Image

from server import protocol


def hello(ws):
    ws.send_json({"t": "hello", "v": 1})
    return recv_until(ws, is_json("ready"))[1]


def test_healthz_and_pages(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["backend"] == "mock"


def test_hello_ready(client):
    with client.websocket_connect("/ws") as ws:
        ready = hello(ws)
    assert ready["v"] == 1 and ready["backend"] == "mock"
    assert ready["topic"] == "ocean" and ready["style"] == "cartoon"
    assert ready["view_mode"] == "auto"
    assert "space" in [t["id"] for t in ready["topics"]]
    assert all("prompt" not in s for s in ready["styles"])


def test_sketch_to_picture_echoes_seq(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_bytes(sketch_frame(42))
        _, (msg_type, seq, jpeg) = recv_until(ws, is_picture())
    assert msg_type == protocol.PICTURE and seq == 42
    img = Image.open(io.BytesIO(jpeg))
    assert img.format == "JPEG" and img.size == (384, 256)  # landscape: canvas_dims(size=384)


def test_blank_sketch_gives_white_picture(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "clear"})
        ws.send_bytes(sketch_frame(7, blank=True))
        _, (_, seq, jpeg) = recv_until(ws, is_picture())
    assert seq == 7
    assert Image.open(io.BytesIO(jpeg)).convert("L").getextrema()[0] > 240


def test_latest_wins_under_burst(make_client):
    with make_client(delay_ms=150) as client, client.websocket_connect("/ws") as ws:
        hello(ws)
        for seq in range(1, 11):
            ws.send_bytes(sketch_frame(seq))
        seen = []
        while not seen or seen[-1] != 10:
            seen.append(recv_until(ws, is_picture())[1][1])
    assert seen[-1] == 10
    assert len(seen) < 10, f"frames were queued instead of overwritten: {seen}"
    assert seen == sorted(seen)


def test_ping_pong_and_ignored_messages(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_text("garbage")
        ws.send_json({"t": "whatever"})
        ws.send_bytes(b"\x01\x00")              # short header
        ws.send_bytes(b"\x02\x00\x00\x00\x01x")  # PICTURE type from client
        ws.send_json({"t": "ping"})
        assert recv_until(ws, is_json("pong"))[1] == {"t": "pong"}


def test_topic_style_selection_survives_reconnect(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "topic", "id": "space"})
        ws.send_json({"t": "style", "id": "crayon"})
        ws.send_json({"t": "style", "id": "does-not-exist"})
        ws.send_json({"t": "ping"})
        recv_until(ws, is_json("pong"))
    with client.websocket_connect("/ws") as ws:
        ready = hello(ws)
    assert ready["topic"] == "space" and ready["style"] == "crayon"


def test_kid_cannot_inject_prompt_text(client):
    app = client.app
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_json({"t": "style", "id": "crayon", "prompt": "INJECTED", "text": "INJECTED"})
        ws.send_json({"t": "prompt", "prompt": "INJECTED"})
        ws.send_json({"t": "ping"})
        recv_until(ws, is_json("pong"))
        ws.send_bytes(sketch_frame(1))
        recv_until(ws, is_picture(1))
    prompt = current_prompt(app)
    assert "INJECTED" not in prompt and "crayon" in prompt
    assert app.state.styles.get("styles", "crayon")["prompt"] in prompt


def test_new_kid_socket_replaces_old(client):
    with client.websocket_connect("/ws") as old:
        hello(old)
        with client.websocket_connect("/ws") as new:
            hello(new)
            recv_until(old, is_json("replaced"))
            kind, code = recv(old)
            assert kind == "close" and code == 4000
            # The new socket still works after the old one was closed.
            new.send_bytes(sketch_frame(5))
            assert recv_until(new, is_picture(5))[1][1] == 5


def test_admin_watcher_gets_pictures_and_stats(admin):
    with admin.websocket_connect("/ws/admin") as watch, admin.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_bytes(sketch_frame(9))
        assert recv_until(ws, is_picture(9))
        assert recv_until(watch, is_picture(9))[1][1] == 9
        started = time.monotonic()
        stats = recv_until(watch, lambda k, v: k == "json" and v.get("t") == "stats"
                           and time.monotonic() - started > 0.5)[1]
    assert stats["backend"] == "mock" and stats["frames"] >= 1 and "fps" in stats


def test_admin_watcher_requires_cookie(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/admin") as ws:
            ws.receive()


def test_new_admin_watcher_gets_last_picture(admin):
    with admin.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_bytes(sketch_frame(11))
        recv_until(ws, is_picture(11))
        with admin.websocket_connect("/ws/admin") as watch:
            assert recv_until(watch, is_picture(11))[1][1] == 11


def test_picture_from_replaced_socket_not_sent_to_new_one(make_client):
    """A frame still on the 'GPU' when its socket is replaced must not reach the new page
    (its seq is from the old page's numbering and would poison the new page's ordering)."""
    with make_client(delay_ms=600) as client, client.websocket_connect("/ws") as old:
        hello(old)
        old.send_bytes(sketch_frame(500))
        end = time.monotonic() + 5
        while client.app.state.mailbox.pending and time.monotonic() < end:
            time.sleep(0.005)  # until the worker has taken frame 500
        with client.websocket_connect("/ws") as new:
            hello(new)
            new.send_bytes(sketch_frame(1))
            seen = []
            while not seen or seen[-1] != 1:
                seen.append(recv_until(new, is_picture())[1][1])
    assert 500 not in seen


def test_replaced_message_survives_full_send_queue(client):
    """The 'replaced' notice and the close get a slot even when the queue is full."""
    from server.session import CLOSE_REPLACED, PEER_QUEUE_SIZE

    session = client.app.state.session
    with client.websocket_connect("/ws") as old:
        hello(old)
        client.portal.call(_fill_then_replace, session.kid, PEER_QUEUE_SIZE * 2, CLOSE_REPLACED)
        recv_until(old, is_json("replaced"))
        assert recv(old) == ("close", CLOSE_REPLACED)


async def _fill_then_replace(peer, n, code):
    # Runs in one event-loop step, so the sender task cannot drain in between.
    for i in range(n):
        peer.send_json({"t": "filler", "i": i})
    peer.close_soon(code, farewell={"t": "replaced"})


def test_clear_and_new_reset_the_worker(client):
    worker = client.app.state.runtime.worker
    resets = []
    real_reset = worker.reset
    worker.reset = lambda: (resets.append(1), real_reset())
    with client.websocket_connect("/ws") as ws:
        hello(ws)
        for msg in ("clear", "new"):
            ws.send_json({"t": msg})
        ws.send_json({"t": "ping"})
        recv_until(ws, is_json("pong"))
    assert len(resets) >= 2


def test_resolution_change_sends_the_kid_page_its_new_canvas(admin):
    with admin.websocket_connect("/ws") as ws:
        hello(ws)
        assert admin.patch("/api/admin/params", json={"seed": 3}).status_code == 200
        assert admin.patch("/api/admin/params", json={"resolution": 512}).status_code == 200
        msg = recv_until(ws, lambda kind, value: kind == "json" and "canvas" in value)[1]
    assert msg["t"] == "styles" and msg["canvas"] == {"landscape": [512, 384], "portrait": [384, 512]}
    assert "topics" in msg and "styles" in msg and msg["topic"] == "ocean"


def test_admin_watcher_closes_when_its_login_expires(admin):
    from conftest import ADMIN_PIN

    auth = admin.app.state.auth
    assert admin.post("/api/admin/login", json={"pin": ADMIN_PIN, "short": True}).status_code == 200
    with admin.websocket_connect("/ws/admin") as watch:
        recv_until(watch, is_json("stats"))
        real_clock = auth._clock
        auth._clock = lambda: real_clock() + 16 * 60  # past the 15-minute gate login
        try:
            for _ in range(200):
                kind, value = recv(watch)
                if kind == "close":
                    break
            assert (kind, value) == ("close", 1008)
        finally:
            auth._clock = real_clock


def test_a_slow_client_gets_the_newest_picture_not_a_backlog():
    import asyncio

    from server.session import Peer

    class SlowSocket:
        def __init__(self):
            self.sent, self.release = [], asyncio.Event()

        async def send_bytes(self, data):
            await self.release.wait()
            self.sent.append(data)

        async def send_text(self, text):
            self.sent.append(text)

    async def run():
        ws = SlowSocket()
        peer = Peer(ws, "admin")
        peer.send_picture(b"first")
        await asyncio.sleep(0)  # the sender picks up "first" and blocks on the socket
        for i in range(50):
            peer.send_picture(f"p{i}".encode())
        ws.release.set()
        await asyncio.sleep(0.05)
        await peer.finish()
        return ws.sent

    assert asyncio.run(run()) == [b"first", b"p49"]


def test_admin_change_redraws_the_drawing_on_screen(admin):
    with admin.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_bytes(sketch_frame(5))
        recv_until(ws, is_picture(5))
        assert admin.patch("/api/admin/params", json={"seed": 77}).status_code == 200
        # Nobody draws: the same drawing comes back with the new settings.
        _, (_, seq, _) = recv_until(ws, is_picture(5))
    assert seq == 5


def test_admin_change_after_clear_redraws_nothing(admin):
    with admin.websocket_connect("/ws") as ws:
        hello(ws)
        ws.send_bytes(sketch_frame(5))
        recv_until(ws, is_picture(5))
        ws.send_json({"t": "clear"})
        assert admin.patch("/api/admin/params", json={"seed": 78}).status_code == 200
        time.sleep(0.8)  # longer than the redraw delay
        ws.send_json({"t": "ping"})
        seen = []
        recv_until(ws, lambda kind, value: seen.append((kind, value)) or is_json("pong")(kind, value))
    assert not any(kind == "bin" for kind, _ in seen)
