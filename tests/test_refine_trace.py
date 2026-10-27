"""Refinement traces: recorded per pause, scored, kept in a ring, saved as a batch folder."""
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import ADMIN_PIN, is_json, recv_until, sketch_frame, sketch_png
from fastapi.testclient import TestClient
from PIL import Image
from test_worker import RefineBackend, _start, wait_until

from server.app import create_app
from server.backends.mock import MockBackend
from server.config import Config
from server.params import defaults_from_config
from server.refine_trace import BATCHES_DIR, JUMP_STEP, RefineTrace, RefineTraces
from server.worker import FrameJob

SIZE = 384


def trace_for(traces, model="path"):
    return RefineTrace(1, model, sketch_png(size=64), "a prompt", "#102030", "landscape",
                       (384, 256), "mock", {"refine_model": model, "t_index_list": [6, 26]})


def jpeg_of(picture):
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG")
    return buffer.getvalue()


def gray(level, size=(384, 256)):
    return Image.new("RGB", size, (level, level, level))


# ----- worker ------------------------------------------------------------------------------

def test_a_pause_records_step_zero_and_every_refined_step():
    backend = RefineBackend()
    cfg = Config(backend="mock", size=SIZE)
    params = replace(defaults_from_config(cfg), t_index_list=(6,), refine_steps=4, refine_model="path")
    sent, metas = [], []
    worker, rec = _start(backend, cfg, params=params)
    worker.on_trace = metas.append
    original = worker.on_result
    worker.on_result = lambda seq, jpeg, owner: (sent.append(jpeg), original(seq, jpeg, owner))
    try:
        worker.mailbox.put(FrameJob(7, sketch_png(size=64), "a prompt", background="#000000"))
        rec.wait(4)  # the drawing picture, then 3 refined pictures
        assert backend.finished.wait(5)
        wait_until(lambda: worker.traces.list() and worker.traces.list()[0]["done"])
    finally:
        worker.stop()
    [meta] = worker.traces.list()
    assert meta["seq"] == 7 and meta["model"] == "path" and not meta["stopped"]
    assert [step["index"] for step in meta["steps"]] == [0, 1, 2, 3]
    first = meta["steps"][0]
    assert first["step"] is None and first["drift"] == 0 and first["ms"] == 0
    for step in meta["steps"][1:]:
        assert isinstance(step["step"], float) and isinstance(step["drift"], float)
        assert step["ms"] >= 0 and step["width"] == 384 and step["height"] == 256
    # Step 0 is the drawing picture as it was sent; the others are the refined pictures.
    trace = worker.traces.get(meta["id"])
    assert [step.jpeg for step in trace.steps] == sent[:4]
    assert metas and metas[-1]["done"] and len(metas[-1]["steps"]) == 4
    assert "jpeg" not in json.dumps(metas[-1])  # metadata only


def test_no_trace_without_refinement():
    backend = RefineBackend()
    cfg = Config(backend="mock", size=SIZE)
    worker, rec = _start(backend, cfg, params=replace(defaults_from_config(cfg), refine_steps=0))
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(1)
    finally:
        worker.stop()
    assert worker.traces.list() == []


# ----- the holder -----------------------------------------------------------------------

def test_ring_keeps_only_the_latest_traces():
    traces = RefineTraces(size=3)
    made = [traces.begin(trace_for(traces), gray(100), jpeg_of(gray(100))) for _ in range(5)]
    listed = [meta["id"] for meta in traces.list()]
    assert listed == [trace.id for trace in reversed(made[2:])]  # newest first
    assert traces.get(made[0].id) is None and traces.get(made[4].id) is made[4]


def test_scores_measure_change_to_previous_step_and_to_step_zero():
    traces = RefineTraces()
    trace = traces.begin(trace_for(traces), gray(100), jpeg_of(gray(100)))
    traces.add_step(trace, gray(102), jpeg_of(gray(102)))
    traces.add_step(trace, gray(140), jpeg_of(gray(140)))
    # An upscaled step (twice the size) is compared at step 0's small size.
    traces.add_step(trace, gray(141, (768, 512)), jpeg_of(gray(141, (768, 512))))
    traces.finish(trace)
    steps = traces.meta(trace)["steps"]
    assert [(step["step"], step["drift"]) for step in steps] == [
        (None, 0.0), (2.0, 2.0), (38.0, 40.0), (1.0, 41.0)]
    assert [step["jump"] for step in steps] == [False, False, True, True]
    assert traces.meta(trace)["jumps"] == 2 and traces.meta(trace)["max_drift"] == 41.0


def test_bad_ids_are_not_found():
    traces = RefineTraces()
    trace = traces.begin(trace_for(traces), gray(100), jpeg_of(gray(100)))
    assert traces.get("../etc") is None and traces.get("ABCDEF123456") is None
    assert traces.step_jpeg(trace.id, 1) is None and traces.step_jpeg(trace.id, -1) is None
    with pytest.raises(KeyError):
        traces.save("nope", "unused")


def test_save_writes_sketch_steps_and_settings(tmp_path):
    traces = RefineTraces()
    trace = trace_for(traces, "anchored")
    traces.begin(trace, gray(100), jpeg_of(gray(100)))
    traces.add_step(trace, gray(110), jpeg_of(gray(110)))
    traces.finish(trace)
    folder = traces.save(trace.id, tmp_path)
    assert folder.startswith(f"{BATCHES_DIR}/") and folder.endswith("-anchored")
    batch = tmp_path / folder
    assert (batch / "sketch.png").read_bytes() == trace.sketch_png
    assert (batch / "step-00.jpg").read_bytes() == trace.steps[0].jpeg
    assert (batch / "step-01.jpg").read_bytes() == trace.steps[1].jpeg
    settings = json.loads((batch / "settings.json").read_text(encoding="utf-8"))
    assert {"params", "prompt", "background", "orientation", "picture_size", "backend", "steps",
            "refine_model", "creativity_pct", "git_commit", "complete"} <= set(settings)
    assert settings["prompt"] == "a prompt" and settings["background"] == "#102030"
    assert settings["picture_size"] == [384, 256] and settings["creativity_pct"] == 88
    assert [step["file"] for step in settings["steps"]] == ["step-00.jpg", "step-01.jpg"]
    assert settings["steps"][1]["step"] == 10.0 and settings["steps"][1]["ms"] >= 0
    # Saving again with nothing new returns the same folder; no stray temp folders remain.
    assert traces.save(trace.id, tmp_path) == folder
    assert [path.name for path in (tmp_path / BATCHES_DIR).iterdir()] == [folder.split("/")[1]]


# ----- app: endpoints and the admin socket ------------------------------------------------

class PathRefineMock(MockBackend):
    """Continues the drawing's path in three pictures, each a little brighter."""

    def __init__(self):
        super().__init__(mock_delay_ms=0)

    def refine_steps(self, init_image, line_map, t_index, first_shown, noise="fresh"):
        base = self.generate(init_image, line_map)
        for level in (4, 8, 40):
            yield Image.eval(base, lambda value, shift=level: min(255, value + shift))


@pytest.fixture
def refine_client(config):
    params_file = Path(config.data_dir) / "params.json"
    params_file.parent.mkdir(parents=True, exist_ok=True)
    params_file.write_text(json.dumps({"refine_model": "path"}), encoding="utf-8")
    with TestClient(create_app(config, backend=PathRefineMock())) as client:
        yield client


def refine_once(client):
    """Draw one stroke and wait (on the admin socket) for its finished trace."""
    with client.websocket_connect("/ws/admin") as admin_ws, client.websocket_connect("/ws") as kid_ws:
        kid_ws.send_json({"t": "hello", "v": 1})
        recv_until(kid_ws, is_json("ready"))
        kid_ws.send_bytes(sketch_frame(3))
        _, message = recv_until(admin_ws, lambda kind, value: kind == "json" and value.get("t") == "refine_trace"
                                and value["trace"]["done"], limit=400)
    return message["trace"]


def test_admin_socket_and_endpoints_serve_the_trace(refine_client, config):
    client = refine_client
    assert client.post("/api/admin/login", json={"pin": ADMIN_PIN}).status_code == 200
    trace = refine_once(client)
    assert trace["model"] == "path" and len(trace["steps"]) == 4 and trace["jumps"] >= 1
    listing = client.get("/api/admin/refine-traces").json()
    assert listing["traces"][0]["id"] == trace["id"] and listing["thresholds"]["step"] == JUMP_STEP

    base = f"/api/admin/refine-traces/{trace['id']}"
    for index in range(4):
        response = client.get(f"{base}/step/{index}.jpg")
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        Image.open(io.BytesIO(response.content)).verify()
    sketch = client.get(f"{base}/sketch.png")
    assert sketch.status_code == 200 and sketch.content == sketch_png()

    saved = client.post(f"{base}/save").json()
    batch = Path(config.data_dir) / saved["folder"]
    assert saved["ok"] and saved["name"].endswith("-path")
    assert sorted(path.name for path in batch.iterdir()) == [
        "settings.json", "sketch.png", "step-00.jpg", "step-01.jpg", "step-02.jpg", "step-03.jpg"]
    settings = json.loads((batch / "settings.json").read_text(encoding="utf-8"))
    assert settings["params"]["refine_steps"] == 8 and settings["backend"] == "mock"
    assert client.get("/api/admin/refine-traces").json()["traces"][0]["saved_as"] == saved["folder"]


def test_bad_trace_ids_and_steps_are_404(refine_client):
    client = refine_client
    assert client.post("/api/admin/login", json={"pin": ADMIN_PIN}).status_code == 200
    trace = refine_once(client)
    base = f"/api/admin/refine-traces/{trace['id']}"
    for path in ["/api/admin/refine-traces/000000000000/step/0.jpg",
                 "/api/admin/refine-traces/..%2F..%2Fdreamer.db/sketch.png",
                 "/api/admin/refine-traces/nope/sketch.png",
                 f"{base}/step/9.jpg", f"{base}/step/-1.jpg", f"{base}/step/x.jpg",
                 f"{base}/step/0.png"]:
        assert client.get(path).status_code == 404, path
    assert client.post("/api/admin/refine-traces/000000000000/save").status_code == 404
    assert client.post("/api/admin/refine-traces/..%2Fx/save").status_code == 404


def test_refine_trace_endpoints_need_admin(client):
    for method, path in [("GET", "/api/admin/refine-traces"),
                         ("GET", "/api/admin/refine-traces/0123456789ab/step/0.jpg"),
                         ("GET", "/api/admin/refine-traces/0123456789ab/sketch.png"),
                         ("POST", "/api/admin/refine-traces/0123456789ab/save")]:
        assert client.request(method, path).status_code == 401, (method, path)


# ----- tools/replay_batch.py (on the mock backend; the real run needs the GPU) -------------

def test_replay_batch_reruns_a_saved_batch(tmp_path):
    from tools import replay_batch

    backend = RefineBackend()
    cfg = Config(backend="mock", size=SIZE)
    params = replace(defaults_from_config(cfg), t_index_list=(6,), refine_steps=4, refine_model="path")
    worker, _recorder = _start(backend, cfg, params=params)
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "a prompt", background="#000000"))
        wait_until(lambda: worker.traces.list() and worker.traces.list()[0]["done"])
    finally:
        worker.stop()
    folder = tmp_path / worker.traces.save(worker.traces.list()[0]["id"], tmp_path)

    pictures = replay_batch.replay(folder, cfg, RefineBackend(), saved_first=True)
    assert len(pictures) == 4 and pictures[0].size == (384, 256)
    assert sorted(path.name for path in folder.glob("replay-step-*.jpg")) == [
        f"replay-step-{index:02d}.jpg" for index in range(4)]
    lines = replay_batch.report(folder, pictures)
    # The mock makes the same pictures again: only JPEG noise between replay and original.
    assert len(lines) == 2 + 4 and float(lines[-1].split("|")[-1]) < 1


def test_replay_batch_leaves_out_params_dreamer_no_longer_has():
    from tools import replay_batch

    cfg = Config(backend="mock", size=SIZE)
    settings = {"params": {"seed": 9, "picture_model": "turbo", "tcd_steps": 4}}  # an older batch
    assert replay_batch.batch_params(cfg, settings) == replace(defaults_from_config(cfg), seed=9)
