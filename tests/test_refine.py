"""Refinement models' helpers (server/refine.py)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from server import params, refine


def _picture(color, size=(128, 96)):
    image = Image.new("RGB", size, color)
    ImageDraw.Draw(image).rectangle((30, 20, 90, 70), fill=(250, 250, 250))
    return image


def test_models_match_the_params_options():
    assert set(refine.MODELS) == set(params.REFINE_MODELS)
    assert refine.DEFAULT_MODEL in refine.MODELS


def test_continue_path_extends_the_drawing_path():
    assert refine.continue_path((6,), 4) == (6, 20, 35, 49)
    assert refine.continue_path((6, 26), 2) is None


def test_lock_keeps_broad_terms_of_the_first_picture():
    first = _picture((10, 10, 40))     # night
    later = _picture((200, 220, 255))  # drifted to day
    locked = refine.lock_broad_terms(first, later, radius=8)
    blur = lambda image: np.asarray(image.filter(ImageFilter.GaussianBlur(8)), dtype=np.float32)
    assert np.abs(blur(locked) - blur(first)).mean() < np.abs(blur(later) - blur(first)).mean() / 4


def test_edge_map_marks_outlines_only():
    edges = np.asarray(refine.edge_map(_picture((0, 0, 0))).convert("L"))
    assert edges[45, 30] == 255 and edges[45, 60] == 0 and edges[5, 5] == 0


def test_rungs_follow_the_budget():
    assert refine._rungs(refine.ANCHORED_RUNGS, (6,), 4) == list(refine.ANCHORED_RUNGS[:3])
    assert refine._rungs(refine.ANCHORED_RUNGS, (6, 26), 2) == [refine.ANCHORED_RUNGS[0]]
