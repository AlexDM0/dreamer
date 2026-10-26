import json

import pytest

from server.styles import PROMPT_HEAD, SAFETY_TAIL, StyleStore


@pytest.fixture
def store(tmp_path):
    return StyleStore(tmp_path)


def ids(items):
    return [it["id"] for it in items]


def test_presets_present(store):
    lists = store.admin_list()
    for topic in ["ocean", "nature", "space", "dinosaurs", "castle", "city", "jungle", "winter"]:
        assert topic in ids(lists["topics"])
    for style in ["cartoon", "crayon", "watercolor", "clay", "storybook", "pixel"]:
        assert style in ids(lists["styles"])
    assert all(it["kind"] == "preset" for it in lists["topics"])


def test_public_list_has_no_prompts_or_paths(store):
    store.add_image_style("Granny's quilt", "🧵", "style_images/x.png")
    for items in store.public_list().values():
        for it in items:
            assert set(it) == {"id", "label", "emoji", "kind"}


def test_build_prompt_format_and_safety_tail_last(store):
    p = store.build_prompt("ocean", "crayon")
    assert p.startswith(PROMPT_HEAD + ", ")
    assert p.endswith(", " + SAFETY_TAIL)
    assert "under the sea" in p and "crayon" in p


def test_build_prompt_unknown_ids_fall_back_to_first_visible(store):
    assert store.build_prompt("nope", "nope") == store.build_prompt("ocean", "cartoon")
    store.update_item("topics", "ocean", hidden=True)
    assert store.build_prompt("ocean", None) == store.build_prompt("nature", "cartoon")


def test_custom_style_crud_and_persistence(tmp_path):
    store = StyleStore(tmp_path)
    item = store.add_custom_style("  Rainbow   Glitter ", "🌈", "rainbow\n glitter   sparkles")
    assert item["kind"] == "custom"
    assert item["label"] == "Rainbow Glitter"
    assert item["prompt"] == "rainbow glitter sparkles"
    assert item["id"].startswith("rainbow-glitter-")
    store.update_item("styles", item["id"], prompt="shiny stickers", label="Shiny")
    store.update_item("styles", "crayon", hidden=True, label="Crayons!")

    reloaded = StyleStore(tmp_path)
    got = reloaded.get("styles", item["id"])
    assert got["prompt"] == "shiny stickers" and got["label"] == "Shiny"
    crayon = reloaded.get("styles", "crayon")
    assert crayon["hidden"] and crayon["label"] == "Crayons!"
    assert "crayon" not in ids(reloaded.public_list()["styles"])
    assert "shiny stickers" in reloaded.build_prompt("ocean", item["id"])
    assert reloaded.build_prompt("ocean", item["id"]).endswith(SAFETY_TAIL)

    reloaded.delete_item("styles", item["id"])
    assert StyleStore(tmp_path).get("styles", item["id"]) is None


def test_file_is_written_atomically_and_presets_stored_as_overrides(tmp_path):
    store = StyleStore(tmp_path)
    store.update_item("topics", "space", hidden=True)
    data = json.loads((tmp_path / "styles.json").read_text(encoding="utf-8"))
    space = next(e for e in data["topics"] if e["id"] == "space")
    assert space == {"id": "space", "hidden": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_unique_ids(store):
    a = store.add_custom_style("Same", None, "one")
    b = store.add_custom_style("Same", None, "two")
    assert a["id"] != b["id"]


@pytest.mark.parametrize("prompt", ["", "   ", "x" * 121, "bad\x00char", "bell\x07"])
def test_custom_prompt_validation(store, prompt):
    with pytest.raises(ValueError):
        store.add_custom_style("Label", "🙂", prompt)


def test_prompt_max_length_allowed(store):
    assert len(store.add_custom_style("Long", None, "y" * 120)["prompt"]) == 120


def test_prompt_editable_only_for_custom(store):
    with pytest.raises(ValueError):
        store.update_item("styles", "cartoon", prompt="something else")
    with pytest.raises(ValueError):
        store.update_item("topics", "ocean", prompt="something else")


def test_presets_cannot_be_deleted(store):
    with pytest.raises(ValueError):
        store.delete_item("styles", "cartoon")
    with pytest.raises(KeyError):
        store.delete_item("styles", "missing")


def test_cannot_hide_everything(store):
    for topic in ids(store.admin_list()["topics"])[:-1]:
        store.update_item("topics", topic, hidden=True)
    last = ids(store.admin_list()["topics"])[-1]
    with pytest.raises(ValueError):
        store.update_item("topics", last, hidden=True)


def test_order(store):
    store.set_order("topics", ["winter", "space"])
    order = ids(store.admin_list()["topics"])
    assert order[:3] == ["winter", "space", "ocean"]
    assert len(order) == 8
    assert StyleStore(store._dir).admin_list()["topics"][0]["id"] == "winter"
    with pytest.raises(KeyError):
        store.set_order("topics", ["nope"])
    with pytest.raises(ValueError):
        store.set_order("topics", ["space", "space"])
    with pytest.raises(ValueError):
        store.set_order("bogus", [])


def test_corrupt_file_falls_back_to_presets(tmp_path):
    (tmp_path / "styles.json").write_text("{not json", encoding="utf-8")
    assert "ocean" in ids(StyleStore(tmp_path).admin_list()["topics"])


def test_image_style(store):
    item = store.add_image_style("My Picture", "🖼️", "style_images/abc.png")
    assert item["kind"] == "image" and item["image"] == "style_images/abc.png"
    assert store.build_prompt("ocean", item["id"]).endswith(SAFETY_TAIL)


def test_failed_save_rolls_back_memory(tmp_path, monkeypatch):
    """If styles.json cannot be written, memory must keep matching what is on disk."""
    store = StyleStore(tmp_path)

    def boom(*args, **kwargs):
        raise PermissionError("file is locked")

    monkeypatch.setattr("server.fsutil.os.replace", boom)
    monkeypatch.setattr("server.fsutil.time.sleep", lambda seconds: None)
    before = store.admin_list()
    with pytest.raises(OSError):
        store.add_custom_style("Rainbow", None, "rainbow colors")
    with pytest.raises(OSError):
        store.update_item("styles", "cartoon", hidden=True)
    with pytest.raises(OSError):
        store.set_order("topics", ["space"])
    assert store.admin_list() == before


def test_custom_topics_can_be_added_used_and_deleted(tmp_path):
    store = StyleStore(tmp_path)
    topic = store.add_custom_topic("Pirates", "🏴‍☠️", "a pirate ship on a sunny sea")
    assert topic["kind"] == "custom" and topic["id"].startswith("pirates")
    prompt = store.build_prompt(topic["id"], None)
    assert "a pirate ship on a sunny sea" in prompt and prompt.endswith(SAFETY_TAIL)
    assert any(item["id"] == topic["id"] for item in StyleStore(tmp_path).public_list()["topics"])  # persisted
    store.update_item("topics", topic["id"], prompt="a pirate ship at night")
    assert "a pirate ship at night" in store.build_prompt(topic["id"], None)
    store.delete_item("topics", topic["id"])
    assert all(item["id"] != topic["id"] for item in StyleStore(tmp_path).admin_list()["topics"])


def test_built_in_topics_cannot_be_deleted(tmp_path):
    store = StyleStore(tmp_path)
    with pytest.raises(ValueError):
        store.delete_item("topics", "ocean")
