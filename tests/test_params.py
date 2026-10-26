"""Live diffusion params: validation, persistence, admin API, effect on pictures."""
import json

import pytest

from conftest import is_json, is_picture, recv_until, sketch_frame
from server.config import Config
from server.params import GenParams, ParamsStore, apply_changes, defaults_from_config


def test_defaults_follow_config():
    cfg = Config()
    cfg.model.t_index_list = [20, 35]
    cfg.model.controlnet = None
    cfg.jpeg_quality = 99  # clamped into the live range
    d = defaults_from_config(cfg)
    assert d.t_index_list == (20, 35) and d.controlnet_enabled is False and d.jpeg_quality == 95


def test_clamping():
    p = apply_changes(GenParams(), {"t_index_list": [-5, 80], "seed": -1, "controlnet_scale": 9,
                                    "safety_threshold": 0, "line_threshold": 500,
                                    "line_dilate": 7, "jpeg_quality": 10})
    assert p.t_index_list == (0, 49) and p.seed == 0 and p.controlnet_scale == 2.0
    assert p.safety_threshold == 0.05 and p.line_threshold == 120
    assert p.line_dilate == 3 and p.jpeg_quality == 50


@pytest.mark.parametrize("bad", [
    {"seed": "7"}, {"seed": True}, {"seed": 1.5}, {"controlnet_enabled": 1},
    {"t_index_list": []}, {"t_index_list": [1, 2, 3, 4, 5, 6, 7, 8, 9]}, {"t_index_list": [30, 30]},
    {"t_index_list": [40, 20]}, {"t_index_list": "30"}, {"line_polarity": "rainbow"},
    {"controlnet_scale": float("nan")}, {"guidance_scale": 1.0}, [1, 2],
])
def test_invalid_rejected(bad):
    with pytest.raises(ValueError):
        apply_changes(GenParams(), bad)


def test_store_persists_and_ignores_bad_fields(tmp_path):
    store = ParamsStore(tmp_path, GenParams())
    store.patch({"seed": 42})
    assert ParamsStore(tmp_path, GenParams()).get().seed == 42
    (tmp_path / "params.json").write_text(json.dumps({"seed": "x", "line_dilate": 2}))
    loaded = ParamsStore(tmp_path, GenParams()).get()
    assert loaded.seed == GenParams().seed and loaded.line_dilate == 2


@pytest.mark.parametrize("method,url", [("get", "/api/admin/params"),
                                        ("patch", "/api/admin/params"),
                                        ("post", "/api/admin/params/reset")])
def test_params_require_login(client, method, url):
    kwargs = {"json": {"seed": 1}} if method == "patch" else {}
    assert getattr(client, method)(url, **kwargs).status_code == 401


def test_params_api_roundtrip(admin, config):
    r = admin.get("/api/admin/params")
    assert r.status_code == 200
    body = r.json()
    assert body["params"] == body["defaults"] and body["limits"]["seed"]["max"] == 2**31 - 1

    r = admin.patch("/api/admin/params", json={"seed": 77, "jpeg_quality": 999})
    assert r.status_code == 200
    assert r.json()["params"]["seed"] == 77 and r.json()["params"]["jpeg_quality"] == 95
    assert admin.patch("/api/admin/params", json={"seed": "x"}).status_code == 400
    assert admin.patch("/api/admin/params", content=b"{nope",
                       headers={"content-type": "application/json"}).status_code == 400

    r = admin.post("/api/admin/params/reset")
    assert r.json()["params"] == r.json()["defaults"]


def test_params_change_the_picture(admin):
    with admin.websocket_connect("/ws") as ws, admin.websocket_connect("/ws/admin") as watch:
        ws.send_json({"t": "hello", "v": 1})
        recv_until(ws, is_json("ready"))
        ws.send_bytes(sketch_frame(1))
        before = recv_until(ws, is_picture(1))[1][2]
        admin.patch("/api/admin/params", json={"t_index_list": [5, 40]})
        assert recv_until(watch, is_json("params"))[1]["params"]["t_index_list"] == [5, 40]
        ws.send_bytes(sketch_frame(2))
        after = recv_until(ws, is_picture(2))[1][2]
    assert before != after


def test_styles_save_failure_is_503_and_rolled_back(admin, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("server.styles.os.replace", boom)
    r = admin.post("/api/admin/styles", json={"label": "Rainbow", "prompt": "rainbow colors"})
    assert r.status_code == 503
    monkeypatch.undo()
    assert all(s["label"] != "Rainbow" for s in admin.get("/api/admin/state").json()["styles"])


def test_unauthenticated_bodies_are_not_read(client):
    """Auth runs before body parsing; oversized bodies are refused up front."""
    r = client.post("/api/admin/styles", content=b"{broken",
                    headers={"content-type": "application/json"})
    assert r.status_code == 401
    r = client.post("/api/admin/login", content=b"x" * (200 * 1024),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413

    def chunks():
        for _ in range(20):
            yield b"x" * (1024 * 1024)

    client.post("/api/admin/login", json={"pin": "4321"})
    r = client.post("/api/admin/style-image", content=chunks(),
                    headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413


def test_foreign_origin_kid_socket_refused(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive()
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"t": "ping"})
        assert recv_until(ws, is_json("pong"))


def test_init_mode_and_blur_validate():
    p = apply_changes(GenParams(), {"init_mode": "sketch", "init_blur": 200})
    assert p.init_mode == "sketch" and p.init_blur == 64
    assert GenParams().init_mode == "colors"
    with pytest.raises(ValueError):
        apply_changes(GenParams(), {"init_mode": "crayon"})


def test_picture_model_and_tcd_steps_validated():
    from server.params import GenParams, apply_changes

    changed = apply_changes(GenParams(), {"picture_model": "tcd", "tcd_steps": 99})
    assert changed.picture_model == "tcd" and changed.tcd_steps == 8
    import pytest

    with pytest.raises(ValueError):
        apply_changes(GenParams(), {"picture_model": "sdxl"})


def test_make_backend_picks_tcd_for_the_real_backend_only():
    from server.backends import MockBackend, make_backend
    from server.backends.tcd import TcdBackend
    from server.config import Config

    assert isinstance(make_backend(Config(backend="mock"), "tcd"), MockBackend)
    assert isinstance(make_backend(Config(backend="streamdiffusion"), "tcd"), TcdBackend)


def test_resolution_below_what_the_engines_take_is_refused():
    from server.params import GenParams, apply_changes
    with pytest.raises(ValueError):
        apply_changes(GenParams(), {"resolution": 384})  # short side 256: below the engines' 384
    assert apply_changes(GenParams(), {"resolution": 512}).resolution == 512


def test_saved_low_resolution_is_ignored(tmp_path):
    import json
    from server.params import GenParams, ParamsStore
    (tmp_path / "params.json").write_text(json.dumps({"resolution": 64, "seed": 9}))
    store = ParamsStore(tmp_path, GenParams())
    assert store.get().resolution == GenParams().resolution and store.get().seed == 9


def test_gpu_backend_needs_size_512(tmp_path):
    from server.config import load_config
    path = tmp_path / "config.yaml"
    path.write_text('backend: streamdiffusion\nadmin_pin: "2468"\nsize: 448\n')
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text('backend: mock\nadmin_pin: "2468"\nsize: 256\n')
    assert load_config(path).size == 256
