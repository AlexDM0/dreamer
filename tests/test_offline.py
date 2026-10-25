"""Offline switch once the models are cached (server/offline.py)."""
import dataclasses
import os

import pytest

from server import offline
from server.config import Config


@pytest.fixture
def config(tmp_path, monkeypatch):
    for name in offline.OFFLINE_ENV:
        monkeypatch.setenv(name, "")  # empty = not set; restored after the test
    base = Config()
    return dataclasses.replace(base, model=dataclasses.replace(base.model, engine_dir=str(tmp_path / "engines")))


def test_online_until_the_models_loaded_once(config):
    assert offline.apply(config) is False
    assert not any(os.environ[name] for name in offline.OFFLINE_ENV)


def test_offline_after_mark_ready(config):
    offline.mark_ready(config)
    assert offline.apply(config) is True
    assert all(os.environ[name] == "1" for name in offline.OFFLINE_ENV)


def test_changed_model_goes_online_again(config):
    offline.mark_ready(config)
    changed = dataclasses.replace(config, model=dataclasses.replace(config.model, controlnet="other/controlnet"))
    assert offline.apply(changed) is False
