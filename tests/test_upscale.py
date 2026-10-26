"""The AI upscaler (server/upscale.py): missing weights, and the network on CPU."""
import torch
from PIL import Image

from server import refine
from server.upscale import Upscaler, _build_network


def test_missing_weights_are_not_available_and_leave_the_picture(tmp_path):
    upscaler = Upscaler(tmp_path / "missing.pth")
    picture = Image.new("RGB", (32, 16), "red")
    assert not upscaler.available
    assert upscaler.upscale(picture, 2) is picture  # nothing loaded, no GPU touched
    assert upscaler._network is None


def test_hires_refinement_enlarges_without_the_upscaler(tmp_path):
    # Hires models with missing weights fall back to a plain resize to the hires size.
    picture = Image.new("RGB", (64, 32), "blue")
    enlarged = refine._enlarged(Upscaler(tmp_path / "none.pth"), picture, refine._hires_size(picture))
    assert enlarged.size == (96, 48)


def test_network_is_rebuilt_from_the_weights():
    # A tiny SRVGGNetCompact state: conv in, 1 body conv, conv out (3 * 4 * 4 channels: 4x).
    features = 4
    state = {
        "body.0.weight": torch.zeros(features, 3, 3, 3), "body.0.bias": torch.zeros(features),
        "body.1.weight": torch.zeros(features),
        "body.2.weight": torch.zeros(features, features, 3, 3), "body.2.bias": torch.zeros(features),
        "body.3.weight": torch.zeros(features),
        "body.4.weight": torch.zeros(48, features, 3, 3), "body.4.bias": torch.zeros(48),
    }
    network = _build_network(state)
    assert network.scale == 4
    image = torch.rand(1, 3, 8, 6)
    with torch.no_grad():
        out = network(image)
    # Zero weights add no detail: the output is the nearest-neighbour enlargement.
    assert out.shape == (1, 3, 32, 24)
    assert torch.equal(out, torch.nn.functional.interpolate(image, scale_factor=4, mode="nearest"))
