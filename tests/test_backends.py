"""Tests for server.backends (mock + StreamDiffusion wiring with fakes; no torch)."""
import subprocess
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest
from PIL import Image, ImageDraw

from server.backends import Backend, MockBackend, make_backend
from server.imaging import make_line_map


@dataclass
class Params:
    t_index_list: list = field(default_factory=lambda: [30, 42])
    seed: int = 2
    controlnet_enabled: bool = True
    controlnet_scale: float = 0.8
    safety_checker: bool = True
    safety_threshold: float = 0.5


@dataclass
class Model:
    base: str = "stabilityai/sd-turbo"
    t_index_list: list = field(default_factory=lambda: [30, 42])
    controlnet: object = "thibaud/controlnet-sd21-scribble-diffusers"
    controlnet_scale: float = 0.8
    engine_dir: str = "engines"
    safety_checker: bool = True
    safety_threshold: float = 0.5


@dataclass
class Cfg:
    backend: str = "mock"
    size: int = 64
    model: Model = field(default_factory=Model)


def _sketch(size=64):
    img = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(img).ellipse([10, 10, 50, 50], outline="black", width=3)
    return img


def _gen(backend, sketch=None):
    sketch = sketch or _sketch()
    return np.asarray(backend.generate(sketch, make_line_map(sketch)))


# -- MockBackend ------------------------------------------------------------------

def test_mock_satisfies_protocol_and_name():
    b = MockBackend(mock_delay_ms=0)
    assert isinstance(b, Backend)
    assert b.name == "mock"
    b.warmup()


def test_mock_is_deterministic_and_same_size():
    a, b = MockBackend(0), MockBackend(0)
    for m in (a, b):
        m.set_prompt("ocean, crayon")
    out = a.generate(_sketch(), None)
    assert out.size == (64, 64) and out.mode == "RGB"
    assert np.array_equal(_gen(a), _gen(b))


def test_mock_output_depends_on_prompt():
    b = MockBackend(0)
    b.set_prompt("space")
    first = _gen(b)
    b.set_prompt("dinosaurs")
    assert not np.array_equal(first, _gen(b))


@pytest.mark.parametrize("change", [
    {"t_index_list": [10, 42]}, {"seed": 99}, {"controlnet_enabled": False}, {"controlnet_scale": 2.0},
])
def test_mock_output_changes_with_params(change):
    b = MockBackend(0)
    b.set_params(Params())
    base = _gen(b)
    b.set_params(Params(**change))
    assert not np.array_equal(base, _gen(b))


def test_mock_rejects_empty_t_index():
    with pytest.raises(ValueError):
        MockBackend(0).set_params(Params(t_index_list=[]))


def test_mock_safety_hook():
    b = MockBackend(0)
    assert b.check_image_safe(Image.new("RGB", (8, 8), "white"))
    assert not b.check_image_safe(Image.new("RGB", (8, 8), (255, 0, 255)))
    almost = Image.new("RGB", (8, 8), (255, 0, 255))
    almost.putpixel((0, 0), (0, 0, 0))
    assert b.check_image_safe(almost)


# -- factory ------------------------------------------------------------------------

def test_make_backend_mock_and_unknown():
    assert isinstance(make_backend(Cfg(backend="mock")), MockBackend)
    with pytest.raises(ValueError):
        make_backend(Cfg(backend="dalle"))


def test_importing_backends_does_not_import_torch():
    code = ("import sys, server.backends, server.backends.streamdiffusion, server.imaging; "
            "assert 'torch' not in sys.modules and 'streamdiffusion' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(__import__('pathlib').Path(__file__).parents[1]))


def test_streamdiffusion_missing_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamdiffusion", None)
    with pytest.raises(RuntimeError, match="requirements-gpu.md"):
        make_backend(Cfg(backend="streamdiffusion"))


# -- StreamDiffusionBackend wiring (fake wrapper) ----------------------------------

class FakeWrapper:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeWrapper.instances.append(self)

    def prepare(self, **kwargs):
        self.calls.append(("prepare", kwargs))

    def update_control_image(self, index, image):
        self.calls.append(("control", index, image.size))

    def update_stream_params(self, **kwargs):
        if kwargs.get("t_index_list") == [1, 2, 3, 4]:
            raise RuntimeError("engine batch range exceeded")
        self.calls.append(("params", kwargs))

    def __call__(self, image, prompt=None):
        self.calls.append(("call", prompt))
        return [image.copy()]


@pytest.fixture
def fake_sd(monkeypatch):
    pkg = types.ModuleType("streamdiffusion")
    mod = types.ModuleType("streamdiffusion.wrapper")
    mod.StreamDiffusionWrapper = FakeWrapper
    pkg.wrapper = mod
    monkeypatch.setitem(sys.modules, "streamdiffusion", pkg)
    monkeypatch.setitem(sys.modules, "streamdiffusion.wrapper", mod)
    FakeWrapper.instances.clear()
    return FakeWrapper


def test_streamdiffusion_flow(fake_sd):
    b = make_backend(Cfg(backend="streamdiffusion"))
    assert b.name == "streamdiffusion" and not fake_sd.instances  # nothing built yet
    b.warmup()
    w = fake_sd.instances[0]
    assert w.kwargs["acceleration"] == "tensorrt" and w.kwargs["use_controlnet"] is True
    assert w.kwargs["controlnet_config"][0]["preprocessor"] == "passthrough"
    assert w.calls[0][0] == "prepare" and w.calls[0][1]["num_inference_steps"] == 50

    b.set_prompt("ocean")
    out = b.generate(_sketch(100), make_line_map(_sketch(100)))
    assert isinstance(out, Image.Image) and out.size == (64, 64)
    assert w.calls[-2] == ("control", 0, (64, 64))
    assert w.calls[-1] == ("call", "ocean")
    b.generate(_sketch(), None)
    assert w.calls[-1] == ("call", None)  # prompt sent once only


def test_streamdiffusion_params(fake_sd):
    b = make_backend(Cfg(backend="streamdiffusion"))
    b.set_params(Params(seed=7))  # before warmup: deferred
    b.warmup()
    w = fake_sd.instances[0]
    assert w.calls[-1] == ("params", {"seed": 7})

    b.set_params(Params(seed=7, controlnet_enabled=False, safety_threshold=0.3))
    kw = w.calls[-1][1]
    assert kw["controlnet_config"][0]["enabled"] is False and kw["safety_checker_threshold"] == 0.3

    n = len(w.calls)
    b.set_params(Params(seed=7, controlnet_enabled=False, safety_threshold=0.3))
    assert len(w.calls) == n  # no-op when unchanged

    b.generate(_sketch(), make_line_map(_sketch()))
    assert not any(c[0] == "control" for c in w.calls)  # ControlNet disabled

    with pytest.raises(RuntimeError):
        b.set_params(Params(t_index_list=[1, 2, 3, 4], seed=7))
    b.set_params(Params(seed=7, controlnet_enabled=False, safety_threshold=0.3))
    assert len(w.calls) == n + 1  # failed change was not committed; only the generate call


def test_streamdiffusion_upload_screening(fake_sd, monkeypatch):
    b = make_backend(Cfg(backend="streamdiffusion"))
    # No transformers available -> allowed with a warning.
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert b.check_image_safe(Image.new("RGB", (8, 8)))

    b2 = make_backend(Cfg(backend="streamdiffusion"))
    tf = types.ModuleType("transformers")

    def pipeline(task, model, device):
        assert task == "image-classification" and model == "Freepik/nsfw_image_detector"
        return lambda img, top_k: (
            [{"label": "high", "score": 0.9}, {"label": "neutral", "score": 0.1}]
            if img.getpixel((0, 0)) == (255, 0, 255)
            else [{"label": "neutral", "score": 0.95}, {"label": "medium", "score": 0.05}]
        )

    tf.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", tf)
    assert b2.check_image_safe(Image.new("RGB", (8, 8), "white"))
    assert not b2.check_image_safe(Image.new("RGB", (8, 8), (255, 0, 255)))
