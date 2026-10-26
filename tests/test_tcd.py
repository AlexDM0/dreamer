"""TcdBackend's pure-Python parts with fakes (no models loaded, CPU torch only)."""
import types

import pytest
import torch
from PIL import Image

from server.backends import tcd
from server.backends.common import previous_or_black


def _backend(**model_fields):
    return tcd.TcdBackend(types.SimpleNamespace(model=types.SimpleNamespace(**model_fields)))


class Rendering(tcd.TcdBackend):
    """Records the step counts instead of running the pipeline."""

    def __init__(self):
        super().__init__(types.SimpleNamespace())
        self.rendered = []

    def _render(self, init, line_map, steps):
        self.rendered.append(steps)
        return init


def test_config_defaults_and_overrides():
    backend = _backend(tcd_strength=0.8, safety_threshold=0.3)
    assert backend._strength == 0.8 and backend._safety_threshold == 0.3
    assert backend._gamma == 0.3 and backend._seed == tcd.DEFAULT_SEED
    assert tcd.TcdBackend(types.SimpleNamespace())._strength == 1.0  # no model section


@pytest.mark.parametrize("draw_steps, budget, ladder", [
    (4, 8, [6, 8]),
    (4, 20, [6, 8, 12, 16, 20]),
    (4, 4, []),             # nothing above the drawing's own steps
    (8, 15, [12]),          # only rungs above the drawing's steps, up to the budget
    (1, 6, [6]),
])
def test_refine_own_ladder(draw_steps, budget, ladder):
    backend = Rendering()
    backend._steps = draw_steps
    init = Image.new("RGB", (64, 64))
    assert list(backend.refine_own(init, None, budget)) == [init] * len(ladder)
    assert backend.rendered == ladder


def test_checked_falls_back_to_the_last_safe_picture_of_the_same_size():
    backend = _backend()
    flags = []
    backend._nsfw = lambda picture: flags.pop(0)
    safe = Image.new("RGB", (640, 448), "green")
    flags[:] = [False, True]
    assert backend._checked(safe) is safe
    assert backend._checked(Image.new("RGB", (640, 448), "red")) is safe


def test_checked_after_an_orientation_change_falls_back_to_black():
    backend = _backend()
    backend._nsfw = lambda picture: picture.getpixel((0, 0)) == (255, 0, 0)
    backend._checked(Image.new("RGB", (640, 448), "green"))  # e.g. the warm-up picture
    flagged = backend._checked(Image.new("RGB", (448, 640), "red"))
    assert flagged.size == (448, 640) and flagged.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_checked_passes_everything_with_the_filter_off():
    backend = _backend(safety_checker=False)
    backend._nsfw = lambda picture: pytest.fail("checked")
    picture = Image.new("RGB", (64, 64))
    assert backend._checked(picture) is picture


def test_previous_or_black():
    previous = Image.new("RGB", (8, 4), "white")
    assert previous_or_black(previous, (8, 4)) is previous
    assert previous_or_black(previous, (4, 8)).size == (4, 8)
    assert previous_or_black(None, (4, 8)).getpixel((0, 0)) == (0, 0, 0)


def test_checker_sees_the_whole_picture_squashed_to_448():
    # A landscape picture with a white band at the far right edge: the old short-side resize +
    # centre crop cut it off; squashing keeps it.
    picture = Image.new("RGB", (640, 448), "black")
    picture.paste((255, 255, 255), (600, 0, 640, 448))
    pixels = tcd.checker_pixels(picture, device="cpu")
    assert pixels.shape == (1, 3, 448, 448) and pixels.dtype == torch.float32
    mean, std = torch.tensor(tcd.CHECKER_MEAN), torch.tensor(tcd.CHECKER_STD)
    white = (1 - mean) / std
    black = -mean / std
    assert torch.allclose(pixels[0, :, 224, -5], white, atol=0.05)
    assert torch.allclose(pixels[0, :, 224, 5], black, atol=0.05)
    portrait = tcd.checker_pixels(Image.new("RGB", (448, 640)), device="cpu")
    assert portrait.shape == (1, 3, 448, 448)


def test_nsfw_runs_the_classifier_on_the_squashed_picture(monkeypatch):
    backend = _backend(safety_threshold=0.5)
    seen = []

    def classifier(pixel_values):
        seen.append(pixel_values.shape)
        # Class 0 is "neutral"; the rest are NSFW classes.
        return types.SimpleNamespace(logits=torch.tensor([[0.0, 3.0, 0.0, 0.0]]))

    backend._checker = classifier
    monkeypatch.setattr(tcd, "checker_pixels", lambda image: torch.zeros(1, 3, 448, 448))
    assert backend._nsfw(Image.new("RGB", (640, 448)))
    assert seen == [(1, 3, 448, 448)]
