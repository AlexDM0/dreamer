"""GpuWorker: params slot, error/ready state handling, owner pass-through."""
import threading
from dataclasses import replace
import time

from conftest import sketch_png
from server.backends.mock import MockBackend
from server.config import Config
from server.mailbox import Mailbox
from server.params import GenParams, defaults_from_config
from server.worker import FrameJob, GpuWorker


class Recorder:
    def __init__(self):
        self.results, self.statuses, self.reverts = [], [], []
        self.event = threading.Event()

    def on_result(self, seq, jpeg, owner):
        self.results.append((seq, owner))
        self.event.set()

    def on_status(self, state, msg):
        self.statuses.append((state, msg))

    def on_revert(self, params):
        self.reverts.append(params)

    def wait(self, n, timeout=5.0):
        end = time.monotonic() + timeout
        while len(self.results) < n and time.monotonic() < end:
            time.sleep(0.01)
        assert len(self.results) >= n, self.statuses


class SpyBackend(MockBackend):
    def __init__(self, fail_seed=None):
        super().__init__(mock_delay_ms=0)
        self.applied, self.fail_seed, self.frames = [], fail_seed, 0

    def set_params(self, params):
        if params.seed == self.fail_seed:
            raise RuntimeError("engine says no")
        super().set_params(params)
        self.applied.append(params)

    def generate(self, init_image, line_map):
        self.frames += 1
        return super().generate(init_image, line_map)


def _start(backend, cfg=None, **kw):
    rec = Recorder()
    cfg = cfg or Config(backend="mock", size=64)
    worker = GpuWorker(backend, Mailbox(), rec.on_result, rec.on_status, cfg,
                       on_params_error=rec.on_revert, **kw)
    worker.start()
    return worker, rec


def test_params_applied_before_next_frame_and_owner_passed_through():
    backend = SpyBackend()
    worker, rec = _start(backend)
    try:
        new = GenParams(seed=99)
        worker.set_params(new)
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p", owner="peer-a"))
        rec.wait(1)
        assert backend.applied[-1] == new and worker.params == new
        assert rec.results == [(1, "peer-a")]
    finally:
        worker.stop()


def test_failed_params_revert_and_report_error():
    backend = SpyBackend(fail_seed=13)
    worker, rec = _start(backend)
    try:
        before = worker.params
        worker.set_params(GenParams(seed=13))
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(1)
        assert worker.params == before
        assert rec.reverts == [before]
        assert any(s == "error" and "settings" in m for s, m in rec.statuses)
        assert rec.statuses[-1][0] == "ready"  # the next real frame recovered
    finally:
        worker.stop()


def test_persisted_params_given_to_fresh_backend():
    backend = SpyBackend()
    cfg = Config(backend="mock", size=64)
    start = replace(defaults_from_config(cfg), seed=5)
    worker, rec = _start(backend, cfg, params=start)
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(1)
        assert backend.applied and backend.applied[0].seed == 5
    finally:
        worker.stop()


def test_blank_frame_does_not_clear_error_state():
    def broken_factory():
        raise RuntimeError("no GPU here")

    worker, rec = _start(None, backend_factory=broken_factory)
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(blank=True, size=64), "p"))
        rec.wait(1)
        assert worker.state == "error"
        assert all(s != "ready" for s, _ in rec.statuses)
    finally:
        worker.stop()
