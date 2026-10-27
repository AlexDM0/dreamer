"""Config loading: admin PIN validation and YAML number pitfalls."""
import logging
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from server.config import Config, ModelConfig, load_config


def _load(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_defaults_without_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.port == 7860 and cfg.admin_pin == "1234"


@pytest.mark.parametrize("line,expected", [
    ('admin_pin: "0420"', "0420"),
    ("admin_pin: 0420", "0420"),        # YAML would read this as octal 272
    ("admin_pin: 0000  # comment", "0000"),
    ("admin_pin: 98765", "98765"),
])
def test_pin_kept_exactly_as_typed(tmp_path, line, expected):
    assert _load(tmp_path, f"backend: mock\n{line}\n").admin_pin == expected


@pytest.mark.parametrize("pin", ['"dreamer"', '"123"', '"1234567890123"', '"12 34"', '""'])
def test_pin_must_be_typeable_on_pin_pad(tmp_path, pin):
    with pytest.raises(ValueError, match="admin_pin"):
        _load(tmp_path, f"admin_pin: {pin}\n")


EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.yaml"


def test_example_config_matches_the_defaults():
    """config.example.yaml documents the defaults: every key it lists has the default value."""
    listed = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    loaded, defaults = load_config(EXAMPLE), Config()
    model_listed = listed.pop("model")
    assert set(listed) <= {item.name for item in fields(Config)}
    assert set(model_listed) <= {item.name for item in fields(ModelConfig)}
    for key in listed:
        assert getattr(loaded, key) == getattr(defaults, key), key
    for key in model_listed:
        assert getattr(loaded.model, key) == getattr(defaults.model, key), f"model.{key}"


@pytest.mark.parametrize("text", [
    "size: 2048", "size: 320", "size: 400", "kid_theme: pink",
    "model:\n  max_batch_steps: 3", "model:\n  max_steps: 9", "model:\n  max_steps: 0",
])
def test_invalid_values_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        _load(tmp_path, text + "\n")


def test_smallest_and_largest_size_load(tmp_path):
    assert _load(tmp_path, "size: 384\n").size == 384
    assert _load(tmp_path, "size: 1024\n").size == 1024


def test_old_config_with_the_dropped_tcd_keys_still_loads(tmp_path, caplog):
    # config.yaml from before SD 1.5 + TCD was removed: its keys are ignored with a warning.
    text = ("backend: mock\nmodel:\n  tcd_base: stable-diffusion-v1-5/stable-diffusion-v1-5\n"
            "  tcd_strength: 1.0\n  safety_threshold: 0.4\n")
    with caplog.at_level(logging.WARNING, logger="dreamer.config"):
        cfg = _load(tmp_path, text)
    assert cfg.model.safety_threshold == 0.4 and not hasattr(cfg.model, "tcd_base")
    assert "model.tcd_base" in caplog.text and "model.tcd_strength" in caplog.text
