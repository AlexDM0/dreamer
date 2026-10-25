"""Config loading: admin PIN validation and YAML number pitfalls."""
import pytest

from server.config import load_config


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
