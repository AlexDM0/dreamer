"""GpuWorker: params slot, error/ready state handling, owner pass-through."""
import io
import threading
import time
from dataclasses import replace

from conftest import sketch_png
from PIL import Image

from server.backends.mock import MockBackend
from server.config import Config
from server.mailbox import Mailbox
from server.params import GenParams, defaults_from_config
from server.worker import MAX_CONSECUTIVE_FAILURES as MAX_FAILURES
from server.worker import FrameJob, GpuWorker

SIZE = 384  # the smallest picture size: 384x256 landscape, 256x384 portrait


def wait_until(predicate, timeout=5.0):
    """Poll ``predicate`` until it holds (fails the test after ``timeout`` seconds)."""
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "condition never became true"
        time.sleep(0.005)


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
    cfg = cfg or Config(backend="mock", size=SIZE)
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
    cfg = Config(backend="mock", size=SIZE)
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


class LaggyBackend(MockBackend):
    """Like StreamDiffusion with 2 steps: each picture is the one for the previous call's
    sketch, and the very first call returns black (empty pipeline)."""

    latency_frames = 1

    def __init__(self):
        super().__init__(mock_delay_ms=0)
        self.previous = None
        self.calls = 0

    def generate(self, init_image, line_map):
        self.calls += 1
        out = self.previous or Image.new("RGB", init_image.size, "black")
        self.previous = init_image.copy()
        return out


def _pixel(jpeg):
    # On the line: (32, 22) of the 64x64 sketch, stretched to 384x256.
    return Image.open(io.BytesIO(jpeg)).convert("RGB").getpixel((192, 88))


def _is_red(rgb):
    return rgb[0] > 150 and rgb[2] < 100


def _is_blue(rgb):
    return rgb[2] > 150 and rgb[0] < 100


def test_latency_backend_is_primed_and_settled():
    rec = Recorder()
    jpegs = []

    def on_result(seq, jpeg, owner):
        jpegs.append((seq, jpeg))
        rec.on_result(seq, jpeg, owner)

    backend = LaggyBackend()
    # Init from the sketch itself, so the stroke colour shows up on the line.
    cfg = Config(backend="mock", size=SIZE)
    worker = GpuWorker(backend, Mailbox(), on_result, rec.on_status, cfg,
                       params=replace(defaults_from_config(cfg), init_mode="sketch"))
    worker.start()
    try:
        red, blue = sketch_png(size=64, color="red"), sketch_png(size=64, color="blue")
        worker.mailbox.put(FrameJob(1, red, "p"))
        rec.wait(1)
        # Primed: the first picture already shows the red drawing, not the empty pipeline.
        assert jpegs[0][0] == 1 and _is_red(_pixel(jpegs[0][1]))
        rec.wait(2)  # the settle frame for seq 1 (same seq again)
        worker.mailbox.put(FrameJob(2, blue, "p"))
        rec.wait(3)
        assert jpegs[2][0] == 2 and _is_red(_pixel(jpegs[2][1]))  # one frame behind: still red
        rec.wait(4)  # drawing paused: settled
        assert jpegs[3][0] == 2 and _is_blue(_pixel(jpegs[3][1]))  # now blue
        assert worker._settle_left == 0 and len(jpegs) == 4  # settles only as often as the latency
    finally:
        worker.stop()
    assert backend.calls == 5  # 2 (primed) + 1 settle + 1 + 1 settle


class SizedBackend(MockBackend):
    def __init__(self):
        super().__init__(mock_delay_ms=0)
        self.sizes = []

    def set_size(self, width, height):
        self.sizes.append((width, height))


def test_worker_switches_backend_size_per_orientation():
    backend = SizedBackend()
    worker, rec = _start(backend, Config(backend="mock", size=SIZE))
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p", background="#000000",
                                    orientation="landscape"))
        rec.wait(1)
        assert backend.sizes == []  # backends start in landscape
        worker.mailbox.put(FrameJob(2, sketch_png(size=64), "p", background="#000000",
                                    orientation="portrait"))
        rec.wait(2)
        assert backend.sizes == [(256, 384)]
    finally:
        worker.stop()


class RefineBackend(MockBackend):
    """Refinement continues one path: yields a picture per step from ``first_shown`` on."""

    def __init__(self):
        super().__init__(mock_delay_ms=0)
        self.paths, self.refined = [], []
        self.finished = threading.Event()  # a refinement ran to its end or was stopped

    def refine_steps(self, init_image, line_map, t_index, first_shown, noise="fresh"):
        self.paths.append((tuple(t_index), first_shown))
        try:
            for step in range(first_shown, len(t_index) + 1):
                self.refined.append(step)
                yield self.generate(init_image, line_map)
        finally:
            self.finished.set()


def test_refine_path_continues_the_drawing_path():
    from server.refine import continue_path as refine_path

    # Draw 1 step + 3 more is exactly the evenly spread 4-step path (the admin page's too).
    assert refine_path((6,), 4) == (6, 20, 35, 49)
    assert refine_path((0,), 4) == (0, 16, 33, 49)
    path = refine_path((6, 26), 20)
    assert path[:2] == (6, 26) and path[-1] == 49 and len(path) == 20
    assert all(b > a for a, b in zip(path, path[1:]))
    assert refine_path((6, 26), 2) is None  # nothing to add
    assert refine_path((45,), 20) == (45, 46, 47, 48, 49)  # only so many t_index values left


def test_pause_continues_the_path_one_picture_per_step():
    backend = RefineBackend()
    cfg = Config(backend="mock", size=SIZE)
    params = replace(defaults_from_config(cfg), t_index_list=(6,), refine_steps=4, refine_model="path")
    worker, rec = _start(backend, cfg, params=params)
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(4)  # the drawing picture (step 1), then steps 2, 3 and 4 of the same path
        assert backend.finished.wait(5)
        assert [seq for seq, _ in rec.results] == [1, 1, 1, 1]
        assert backend.paths == [((6, 20, 35, 49), 2)] and backend.refined == [2, 3, 4]
    finally:
        worker.stop()


def test_refinement_off_sends_only_the_drawing_frame():
    backend = RefineBackend()
    cfg = Config(backend="mock", size=SIZE)
    worker, rec = _start(backend, cfg, params=replace(defaults_from_config(cfg), refine_steps=0))
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(1)
        assert not worker._refine_plan  # nothing left to do once drawing pauses
        assert backend.paths == [] and len(rec.results) == 1
    finally:
        worker.stop()


class SlowRefineBackend(RefineBackend):
    def refine_steps(self, init_image, line_map, t_index, first_shown, noise="fresh"):
        for picture in super().refine_steps(init_image, line_map, t_index, first_shown, noise):
            time.sleep(0.05)
            yield picture


def test_new_stroke_stops_the_refinement_path():
    backend = SlowRefineBackend()
    cfg = Config(backend="mock", size=SIZE)
    params = replace(defaults_from_config(cfg), t_index_list=(6,), refine_steps=20, refine_model="path")
    worker, rec = _start(backend, cfg, params=params)
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(3)  # drawing picture + a couple of refinement steps
        worker.mailbox.put(FrameJob(2, sketch_png(size=64, color="red"), "p"))
        end = time.monotonic() + 5
        while 2 not in [seq for seq, _ in rec.results] and time.monotonic() < end:
            time.sleep(0.01)
        assert 2 in [seq for seq, _ in rec.results]
        stopped_at = len([seq for seq, _ in rec.results if seq == 1])
        assert stopped_at < 20  # the first path was cut short, not run to the end
    finally:
        worker.stop()


class ClosableBackend(SpyBackend):
    """Remembers whether the worker closed it (gave its GPU memory back)."""

    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


def test_reset_stops_settling_and_refining_the_old_drawing():
    backend = SlowRefineBackend()
    cfg = Config(backend="mock", size=SIZE)
    params = replace(defaults_from_config(cfg), t_index_list=(6,), refine_steps=20, refine_model="path")
    worker, rec = _start(backend, cfg, params=params)
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(3)  # refining
        worker.reset()
        assert backend.finished.wait(5)  # the refinement was cut short
        wait_until(lambda: worker._settle_job is None)
        assert len(rec.results) < 20 and not worker._refine_plan and worker._settle_left == 0
    finally:
        worker.stop()


def test_settle_frame_gives_way_to_a_waiting_sketch():
    backend = LaggyBackend()
    rec = Recorder()
    worker = GpuWorker(backend, Mailbox(), rec.on_result, rec.on_status, Config(backend="mock", size=SIZE))
    job = FrameJob(1, sketch_png(size=64), "p")
    worker.mailbox.put(FrameJob(2, sketch_png(size=64, color="red"), "p"))
    worker._process_frame(job, settling=True)  # on this thread; the worker is not started
    assert backend.calls == 0 and rec.results == []


def test_init_image_is_composed_only_without_a_line_map(monkeypatch):
    from server import imaging

    worker = GpuWorker(MockBackend(mock_delay_ms=0), Mailbox(), lambda *args: None,
                       lambda *args: None, Config(backend="mock", size=SIZE))
    sketch = imaging.decode_sketch(sketch_png(size=64), (384, 256))
    job = FrameJob(1, b"", "p")
    calls = []
    real_compose = imaging.compose
    monkeypatch.setattr(imaging, "compose", lambda *args: calls.append(args) or real_compose(*args))
    _init, line_map = worker._frame_inputs(sketch, job, worker.params)  # ControlNet on
    assert line_map is not None and calls == []
    worker._frame_inputs(sketch, job, replace(worker.params, controlnet_enabled=False))
    assert len(calls) == 1


def test_upscaler_is_made_without_upscaling_a_picture(monkeypatch):
    from server import upscale

    made, upscaled = [], []

    class FakeUpscaler:
        available = True

        def __init__(self, model_path):
            made.append(model_path)

        def upscale(self, picture, factor=2):
            upscaled.append(factor)
            return picture

    monkeypatch.setattr(upscale, "Upscaler", FakeUpscaler)
    worker = GpuWorker(MockBackend(mock_delay_ms=0), Mailbox(), lambda *args: None,
                       lambda *args: None, Config(backend="mock", size=SIZE))
    assert worker._ensure_upscaler() is worker._ensure_upscaler()
    assert len(made) == 1 and upscaled == []


class FailingBackend(ClosableBackend):
    def generate(self, init_image, line_map):
        raise RuntimeError("CUDA out of memory")


def test_rebuild_after_failures_closes_the_old_backend_first():
    built = []

    def factory():
        built.append(FailingBackend() if not built else ClosableBackend())
        return built[-1]

    cfg = Config(backend="mock", size=SIZE)
    worker, _rec = _start(None, cfg, backend_factory=factory,
                          params=replace(defaults_from_config(cfg), refine_steps=0))
    try:
        for seq in range(1, MAX_FAILURES + 1):
            worker.mailbox.put(FrameJob(seq, sketch_png(size=64), "p"))
            wait_until(lambda failures=seq: worker._errors >= failures or len(built) > 1)
        wait_until(lambda: len(built) == 2)
        assert built[0].closed  # its GPU memory was given back before the new one loaded
    finally:
        worker.stop()


class PickyBackend(ClosableBackend):
    """Refuses the safety filter on (like a backend whose config has it off)."""

    def set_params(self, params):
        if params.safety_checker:
            raise RuntimeError("the safety checker needs a restart")
        super().set_params(params)


def test_one_refused_saved_param_keeps_the_other_saved_params():
    cfg = Config(backend="mock", size=SIZE)
    cfg.model.safety_checker = False
    backend = PickyBackend()
    saved = replace(defaults_from_config(cfg), seed=5, jpeg_quality=70, safety_checker=True)
    worker, rec = _start(backend, cfg, params=saved)
    try:
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        rec.wait(1)
        assert worker.params == replace(saved, safety_checker=False)
        assert backend.applied[-1] == worker.params
        assert rec.reverts == [worker.params]  # persisted and shown on the admin page
        assert any(state == "error" and "safety_checker" in msg for state, msg in rec.statuses)
    finally:
        worker.stop()


def test_no_line_map_without_a_controlnet_in_config():
    from server import imaging

    cfg = Config(backend="mock", size=SIZE)
    cfg.model.controlnet = None  # SD-Turbo without ControlNet
    worker = GpuWorker(MockBackend(mock_delay_ms=0), Mailbox(), lambda *args: None,
                       lambda *args: None, cfg)
    sketch = imaging.decode_sketch(sketch_png(size=64), (384, 256))
    job = FrameJob(1, b"", "p")
    params = replace(worker.params, controlnet_enabled=True)
    assert worker._frame_inputs(sketch, job, params)[1] is None
    cfg.model.controlnet = ""
    assert worker._frame_inputs(sketch, job, params)[1] is None


class CrashingBackend(SpyBackend):
    """The GPU crashed: every frame fails with CUDA's illegal-memory-access error."""

    def generate(self, init_image, line_map):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


def test_gpu_crash_asks_for_a_process_restart_once():
    fatal = []
    worker, rec = _start(CrashingBackend(), Config(backend="mock", size=SIZE), on_fatal=fatal.append)
    try:
        for seq in (1, 2):
            worker.mailbox.put(FrameJob(seq, sketch_png(size=64), "p"))
        wait_until(lambda: fatal and any("restarting" in msg for _, msg in rec.statuses))
        assert len(fatal) == 1 and "illegal memory access" in fatal[0]
    finally:
        worker.stop()


def test_ordinary_failures_do_not_restart_the_process():
    fatal = []
    worker, rec = _start(SpyBackend(fail_seed=13), Config(backend="mock", size=SIZE), on_fatal=fatal.append)
    try:
        worker.set_params(replace(worker.params, seed=13))
        worker.mailbox.put(FrameJob(1, sketch_png(size=64), "p"))
        wait_until(lambda: rec.reverts)
        assert fatal == []
    finally:
        worker.stop()
