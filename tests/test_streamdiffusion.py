"""StreamDiffusionBackend refine paths with a fake wrapper and stream (CPU torch, no GPU).

The fake stream's "VAE" is an 8x average pool (and nearest-neighbour enlargement back), its
"UNet" returns the sample unchanged; enough to run latent_ladder and the hires tiles end to
end and to check what reaches the wrapper.
"""
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from server import refine
from server.backends.streamdiffusion import StreamDiffusionBackend


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
    backend: str = "streamdiffusion"
    size: int = 128  # 128x64 landscape: not square, so swapped sizes show
    model: Model = field(default_factory=Model)


@dataclass
class Params:
    t_index_list: list = field(default_factory=lambda: [30, 42])
    seed: int = 2
    controlnet_enabled: bool = True
    controlnet_scale: float = 0.8
    safety_checker: bool = True
    safety_threshold: float = 0.5


class FakeVae:
    dtype = torch.float32
    config = types.SimpleNamespace(scaling_factor=1.0)

    def __init__(self):
        self.encodes = 0

    def encode(self, pixels):
        self.encodes += 1
        pooled = F.avg_pool2d(pixels, 8)
        return types.SimpleNamespace(latents=torch.cat([pooled, torch.zeros_like(pooled[:, :1])], dim=1))


class FakeStream:
    device = "cpu"
    dtype = torch.float32
    frame_bff_size = 1
    do_add_noise = True
    generator = None

    def __init__(self, width, height):
        self.use_denoising_batch = True
        self.vae = FakeVae()
        self.init_noise = torch.zeros(1, 4, height // 8, width // 8)
        self.sub_timesteps_tensor = torch.tensor([30, 42])
        self.alpha_prod_t_sqrt = [torch.tensor(1.0)] * 50
        self.beta_prod_t_sqrt = [torch.tensor(0.0)] * 50
        self.unet_gain = 1.0

    def unet_step(self, sample, timestep, idx):
        return sample * self.unet_gain, None

    def decode_image(self, latent):
        return F.interpolate(latent[:, :3], scale_factor=8, mode="nearest")


class FakeWrapper:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.stream = FakeStream(kwargs["width"], kwargs["height"])
        self.use_safety_checker = kwargs["use_safety_checker"]
        self.safety_checker_threshold = kwargs["safety_checker_threshold"]
        self.safety_checker = lambda tensor, threshold: False  # loaded: nothing flagged
        # Like the fork (wrapper.py set_nsfw_fallback_img): black, width and height swapped.
        self.nsfw_fallback_img = Image.new("RGB", (kwargs["height"], kwargs["width"]))
        self.fail_t_index = None
        FakeWrapper.instances.append(self)

    def prepare(self, **kwargs):
        self.calls.append(("prepare", kwargs))

    def update_prompt(self, prompt):
        self.calls.append(("prompt", prompt))

    def update_control_image(self, index, image):
        if not self.kwargs["use_controlnet"]:  # the fork raises without a ControlNet
            raise RuntimeError("ControlNet is not enabled")
        self.calls.append(("control", index, image.size))

    def update_stream_params(self, **kwargs):
        if self.fail_t_index is not None and kwargs.get("t_index_list") == self.fail_t_index:
            raise RuntimeError("engine batch range exceeded")
        if "t_index_list" in kwargs:
            self.stream.sub_timesteps_tensor = torch.tensor(kwargs["t_index_list"])
        self.calls.append(("params", kwargs))

    def preprocess_image(self, picture):
        array = np.asarray(picture.convert("RGB"), dtype=np.float32) / 127.5 - 1
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    def postprocess_image(self, tensor, output_type="pil"):
        array = ((tensor[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
        return Image.fromarray(array)

    def __call__(self, image, prompt=None):
        self.calls.append(("call", prompt))
        return image.copy()


@pytest.fixture
def fake_sd(monkeypatch):
    package = types.ModuleType("streamdiffusion")
    module = types.ModuleType("streamdiffusion.wrapper")
    module.StreamDiffusionWrapper = FakeWrapper
    package.wrapper = module
    monkeypatch.setitem(sys.modules, "streamdiffusion", package)
    monkeypatch.setitem(sys.modules, "streamdiffusion.wrapper", module)
    # latent_ladder imports diffusers' retrieve_latents: a stand-in that skips the import.
    names = ["diffusers", "diffusers.pipelines", "diffusers.pipelines.stable_diffusion",
             "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].retrieve_latents = lambda output, generator: output.latents
    FakeWrapper.instances.clear()
    return FakeWrapper


def _backend(**model_fields):
    backend = StreamDiffusionBackend(Cfg(model=Model(**model_fields)))
    backend.warmup()
    return backend, FakeWrapper.instances[-1]


def _picture(color=(200, 80, 40), size=(128, 64)):
    return Image.new("RGB", size, color)


# -- latent_ladder ----------------------------------------------------------------------

def test_outlines_model_without_controlnet_does_not_push_control_images(fake_sd):
    backend, wrapper = _backend(controlnet=None)
    pictures = list(refine.run("anchored_edges", backend, _picture(), _picture(), None, (30,), 4))
    assert len(pictures) == 3 and all(picture.size == (128, 64) for picture in pictures)
    assert not any(call[0] == "control" for call in wrapper.calls)


def test_outlines_model_pushes_control_images_while_controlnet_is_on(fake_sd):
    backend, wrapper = _backend()
    list(refine.run("anchored_edges", backend, _picture(), _picture(), None, (30,), 4))
    assert [call for call in wrapper.calls if call[0] == "control"] == [("control", 0, (128, 64))] * 3

    backend.set_params(Params(controlnet_enabled=False))
    wrapper.calls.clear()
    list(refine.run("anchored_edges", backend, _picture(), _picture(), None, (30,), 4))
    assert not any(call[0] == "control" for call in wrapper.calls)


def test_latent_ladder_applies_a_pending_prompt(fake_sd):
    backend, wrapper = _backend()
    backend.set_prompt("a dragon")
    next(backend.latent_ladder(_picture(), None, [40], chain=True, anchor_sigma=2.0, gamma=0.0))
    assert ("prompt", "a dragon") in wrapper.calls
    wrapper.calls.clear()
    next(backend.latent_ladder(_picture(), None, [40], chain=True, anchor_sigma=2.0, gamma=0.0))
    assert not any(call[0] == "prompt" for call in wrapper.calls)  # sent once only


def test_latent_ladder_without_chain_starts_every_rung_from_first(fake_sd):
    backend, wrapper = _backend()
    first, anchor = _picture((200, 80, 40)), _picture((20, 20, 220))
    # A UNet that halves the latent, no noise: chained rungs would keep halving; restarting
    # from first gives the same picture every rung (also with another anchor picture).
    wrapper.stream.unet_gain = 0.5
    pictures = list(backend.latent_ladder(first, None, [40, 44], chain=False, anchor_sigma=None,
                                          gamma=0.0, anchor_image=anchor))
    assert np.array_equal(np.asarray(pictures[0]), np.asarray(pictures[1]))


# -- safety fallbacks -----------------------------------------------------------------

def test_flagged_refine_picture_falls_back_to_black_at_the_right_size(fake_sd):
    backend, wrapper = _backend()
    wrapper.safety_checker = lambda tensor, threshold: True
    picture = next(backend.latent_ladder(_picture(), None, [40], chain=True, anchor_sigma=None, gamma=0.0))
    # The fork's initial fallback is 64x128; the picture is 128x64.
    assert picture.size == (128, 64) and picture.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_flagged_refine_picture_falls_back_to_the_previous_safe_one(fake_sd):
    backend, wrapper = _backend()
    wrapper.safety_checker = lambda tensor, threshold: False
    safe = next(backend.latent_ladder(_picture(), None, [40], chain=True, anchor_sigma=None, gamma=0.0))
    assert wrapper.nsfw_fallback_img is safe
    wrapper.safety_checker = lambda tensor, threshold: True
    assert next(backend.latent_ladder(_picture((0, 255, 0)), None, [40], chain=True,
                                       anchor_sigma=None, gamma=0.0)) is safe


def test_wrong_size_fallback_frame_from_the_wrapper_becomes_black(fake_sd):
    backend, wrapper = _backend()
    wrapper.__class__ = type("FlaggingWrapper", (FakeWrapper,), {
        "__call__": lambda self, image, prompt=None: self.nsfw_fallback_img})
    picture = backend.generate(_picture(), None)
    assert picture.size == (128, 64)


def test_hires_tiles_check_the_composite_once(fake_sd):
    backend, wrapper = _backend()
    checked = []
    backend._checker_input = lambda picture: picture
    wrapper.safety_checker = lambda picture, threshold: checked.append(picture.size) or False
    fallback_before = wrapper.nsfw_fallback_img = _picture((1, 2, 3))
    first = _picture()
    pictures = list(refine.run("hires_tiles", backend, first, first, None, (30,), 8))
    assert len(pictures) == 3 and all(picture.size == (192, 96) for picture in pictures)
    assert checked == [(192, 96)] * 3  # one check per rung, of the whole picture
    assert wrapper.nsfw_fallback_img is fallback_before  # still a full-size stream frame
    # 4 tiles: each tile encoded every rung, its anchor crop only on the first rung.
    assert wrapper.stream.vae.encodes == 4 * 3 + 4


def test_flagged_hires_composite_falls_back_to_the_enlarged_first_picture(fake_sd):
    backend, wrapper = _backend()
    backend._checker_input = lambda picture: picture
    wrapper.safety_checker = lambda picture, threshold: True
    first = _picture((10, 200, 30))
    pictures = list(refine.run("hires_tiles", backend, first, first, None, (30,), 8))
    enlarged = first.resize((192, 96), Image.Resampling.LANCZOS)
    assert all(np.array_equal(np.asarray(picture), np.asarray(enlarged)) for picture in pictures)


def test_hires_composite_is_not_checked_with_the_live_filter_off(fake_sd):
    backend, wrapper = _backend()
    backend._checker_input = lambda picture: picture
    wrapper.safety_checker = lambda picture, threshold: pytest.fail("checked")
    backend.set_params(Params(safety_checker=False))
    wrapper.use_safety_checker = False  # what update_stream_params does in the fork
    assert len(list(refine.run("hires_tiles", backend, _picture(), _picture(), None, (30,), 8))) == 3


# -- upload screening -------------------------------------------------------------------

def test_uploads_are_screened_with_the_live_filter_off(fake_sd):
    backend, wrapper = _backend()
    backend._checker_input = lambda picture: picture
    wrapper.safety_checker = lambda picture, threshold: picture.getpixel((0, 0)) == (255, 0, 255)
    backend.set_params(Params(safety_checker=False))
    assert not backend.check_image_safe(_picture((255, 0, 255)))
    assert backend.check_image_safe(_picture((0, 0, 0)))


# -- stream steps -----------------------------------------------------------------------

def test_failed_mode_switch_forgets_the_stream_steps(fake_sd):
    backend, wrapper = _backend(t_index_list=[30, 42])
    assert backend._stream_t_index == [30, 42] and backend._stream_batched
    wrapper.fail_t_index = [30, 42]
    with pytest.raises(RuntimeError):
        backend._ensure_stream([30, 42], False)  # pass-through [30] succeeds, the real update fails
    assert backend._stream_t_index == [] and wrapper.stream.use_denoising_batch is True
    wrapper.fail_t_index = None
    wrapper.calls.clear()
    backend._ensure_stream([30, 42], True)  # the stream really runs [30] now: set the steps again
    assert [call[1]["t_index_list"] for call in wrapper.calls if call[0] == "params"] == [[30, 42]]


def test_step_change_reseeds_the_stream(fake_sd):
    backend, wrapper = _backend()
    backend.set_params(Params(t_index_list=[20, 30, 42], seed=7))
    backend.set_params(Params(t_index_list=[20, 42], seed=7))
    assert wrapper.calls[-1] == ("params", {"t_index_list": [20, 42], "seed": 7})


def test_prepare_inputs_keeps_a_ready_picture(fake_sd):
    backend, wrapper = _backend()
    ready = _picture()
    assert backend._prepare_inputs(ready, None) is ready
    resized = backend._prepare_inputs(_picture(size=(64, 64)), None)
    assert resized.size == (128, 64)
    assert backend._prepare_inputs(ready.convert("RGBA"), None).mode == "RGB"
