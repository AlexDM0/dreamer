"""StyleStore: the topic and style lists the kid picks from, and the prompt builder.

Presets live in code. ``data/styles.json`` stores one ordered list per kind: presets
appear there only as ``{"id", "hidden", "label"?, "emoji"?}`` overrides, while custom and
image styles are stored in full. Presets added to the code later are appended; presets
removed from the code are dropped. Writes are atomic (temp file + os.replace).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger("dreamer.styles")

KINDS = ("topics", "styles")
STYLES_FILE = "styles.json"
PROMPT_MAX = 120
LABEL_MAX = 40
EMOJI_MAX = 16  # code points; ZWJ emoji sequences can be long
REPLACE_RETRIES = 5  # Windows: os.replace fails while antivirus/indexer holds the file
SAFETY_TAIL = "for children, friendly, bright colors, wholesome"
PROMPT_HEAD = "a child's drawing turned into a picture"

# (id, label, emoji, prompt fragment)
PRESET_TOPICS = [
    ("ocean", "Ocean", "🌊", "under the sea with fish, coral and bubbles"),
    ("nature", "Nature", "🌳", "a sunny meadow with trees and flowers"),
    ("space", "Space", "🚀", "outer space with planets, stars and rockets"),
    ("dinosaurs", "Dinosaurs", "🦕", "a land of friendly dinosaurs and ferns"),
    ("castle", "Castle", "🏰", "a fairytale castle with flags and towers"),
    ("city", "City", "🏙️", "a busy city with houses, cars and streets"),
    ("jungle", "Jungle", "🐒", "a lush jungle with vines, parrots and monkeys"),
    ("winter", "Winter", "❄️", "a snowy winter wonderland"),
]
PRESET_STYLES = [
    ("cartoon", "Cartoon", "🖍️", "cartoon style, bold outlines, flat colors"),
    ("crayon", "Crayon", "🖍", "crayon drawing, waxy texture"),
    ("watercolor", "Watercolor", "🎨", "soft watercolor painting"),
    ("clay", "Clay", "🧸", "claymation, plasticine models"),
    ("storybook", "Storybook", "📖", "storybook illustration"),
    ("pixel", "Pixel Art", "👾", "pixel art, retro video game"),
]
_PRESETS = {"topics": PRESET_TOPICS, "styles": PRESET_STYLES}
DEFAULT_EMOJI = {"topics": "✨", "styles": "🎨"}


def _preset_items(kind: str) -> list[dict[str, Any]]:
    return [
        {"id": i, "label": label, "emoji": emoji, "prompt": prompt,
         "kind": "preset", "hidden": False, "image": None}
        for i, label, emoji, prompt in _PRESETS[kind]
    ]


def clean_text(value: Any, max_len: int, field: str, allow_empty: bool = False) -> str:
    """Strip, collapse whitespace, enforce length, reject control characters."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    text = " ".join(value.split())
    if not text and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    if len(text) > max_len:
        raise ValueError(f"{field} is too long (max {max_len} characters)")
    if any(unicodedata.category(ch) in ("Cc", "Cs") for ch in text):
        raise ValueError(f"{field} contains control characters")
    return text


def clean_emoji(value: Any, kind: str) -> str:
    """Emoji may use ZWJ / variation selectors, so only Cc/Cs are rejected."""
    if value is None:
        return DEFAULT_EMOJI[kind]
    text = clean_text(value, EMOJI_MAX, "emoji", allow_empty=True).replace(" ", "")
    return text or DEFAULT_EMOJI[kind]


def _slug(label: str) -> str:
    norm = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-")[:24].strip("-")
    return slug or "style"


class StyleStore:
    """Thread-safe topics/styles store backed by ``data_dir/styles.json``."""

    def __init__(self, data_dir: str | os.PathLike) -> None:
        self._dir = Path(data_dir)
        self._path = self._dir / STYLES_FILE
        self._lock = threading.RLock()
        self._items: dict[str, list[dict[str, Any]]] = {k: _preset_items(k) for k in KINDS}
        self._load()

    # ----- persistence -------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.error("could not read %s (%s); using presets only", self._path, exc)
            return
        for kind in KINDS:
            stored = data.get(kind) if isinstance(data, dict) else None
            if isinstance(stored, list):
                self._items[kind] = self._merge(kind, stored)

    def _merge(self, kind: str, stored: list[Any]) -> list[dict[str, Any]]:
        presets = {it["id"]: it for it in _preset_items(kind)}
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in stored:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                continue
            item_id = entry["id"]
            if item_id in seen:
                continue
            if item_id in presets:
                item = presets[item_id]
                for key in ("label", "emoji"):
                    if isinstance(entry.get(key), str):
                        item[key] = entry[key]
            elif kind == "styles" and entry.get("kind") in ("custom", "image"):
                item = {
                    "id": item_id,
                    "label": str(entry.get("label") or item_id),
                    "emoji": str(entry.get("emoji") or DEFAULT_EMOJI[kind]),
                    "prompt": str(entry.get("prompt") or ""),
                    "kind": entry["kind"],
                    "hidden": False,
                    "image": entry.get("image") if isinstance(entry.get("image"), str) else None,
                }
            else:
                continue  # preset removed from code, or junk
            item["hidden"] = bool(entry.get("hidden", False))
            merged.append(item)
            seen.add(item_id)
        merged.extend(it for i, it in presets.items() if i not in seen)
        return merged

    def _serialise(self, kind: str) -> list[dict[str, Any]]:
        defaults = {it["id"]: it for it in _preset_items(kind)}
        out = []
        for item in self._items[kind]:
            if item["kind"] == "preset":
                entry: dict[str, Any] = {"id": item["id"], "hidden": item["hidden"]}
                for key in ("label", "emoji"):
                    if item[key] != defaults[item["id"]][key]:
                        entry[key] = item[key]
                out.append(entry)
            else:
                out.append(dict(item))
        return out

    def _save(self) -> None:
        """Atomically write styles.json (temp file, fsync, replace). Raises OSError."""
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, **{k: self._serialise(k) for k in KINDS}}
        tmp = self._path.with_name(self._path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(REPLACE_RETRIES):
            try:
                os.replace(tmp, self._path)
                return
            except PermissionError:
                if attempt == REPLACE_RETRIES - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))

    @contextmanager
    def _mutation(self):
        """Hold the lock for a change; if saving fails, roll memory back to match disk."""
        with self._lock:
            backup = copy.deepcopy(self._items)
            try:
                yield
                self._save()
            except BaseException:
                self._items = backup
                raise

    # ----- reads -------------------------------------------------------------------

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")

    def _find(self, kind: str, item_id: str) -> dict[str, Any]:
        self._check_kind(kind)
        for item in self._items[kind]:
            if item["id"] == item_id:
                return item
        raise KeyError(item_id)

    def get(self, kind: str, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            try:
                return copy.deepcopy(self._find(kind, item_id))
            except KeyError:
                return None

    def public_list(self) -> dict[str, list[dict[str, Any]]]:
        """Visible items for the kid page: no prompts, no file paths."""
        with self._lock:
            return {
                kind: [{"id": it["id"], "label": it["label"], "emoji": it["emoji"], "kind": it["kind"]}
                       for it in self._items[kind] if not it["hidden"]]
                for kind in KINDS
            }

    def admin_list(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {kind: copy.deepcopy(self._items[kind]) for kind in KINDS}

    def resolve(self, kind: str, item_id: str | None) -> str:
        """Return ``item_id`` if it names a visible item, else the first visible item."""
        with self._lock:
            items = self._items[kind]
            visible = [it["id"] for it in items if not it["hidden"]]
            if item_id in visible:
                return item_id
            return visible[0] if visible else items[0]["id"]

    def build_prompt(self, topic_id: str | None, style_id: str | None) -> str:
        """Compose the diffusion prompt; the safety tail always comes last."""
        with self._lock:
            topic = self._find("topics", self.resolve("topics", topic_id))["prompt"]
            style = self._find("styles", self.resolve("styles", style_id))["prompt"]
        parts = [PROMPT_HEAD] + [p for p in (topic, style) if p] + [SAFETY_TAIL]
        return ", ".join(parts)

    # ----- mutations ---------------------------------------------------------------

    def _new_id(self, kind: str, label: str) -> str:
        taken = {it["id"] for it in self._items[kind]}
        base = _slug(label)
        while True:
            candidate = f"{base}-{secrets.token_hex(2)}"
            if candidate not in taken:
                return candidate

    def _add_style(self, label: Any, emoji: Any, prompt: str, kind: str, image: str | None) -> dict[str, Any]:
        label = clean_text(label, LABEL_MAX, "label")
        with self._mutation():
            item = {
                "id": self._new_id("styles", label),
                "label": label,
                "emoji": clean_emoji(emoji, "styles"),
                "prompt": prompt,
                "kind": kind,
                "hidden": False,
                "image": image,
            }
            self._items["styles"].append(item)
        return copy.deepcopy(item)

    def add_custom_style(self, label: Any, emoji: Any, prompt: Any) -> dict[str, Any]:
        return self._add_style(label, emoji, clean_text(prompt, PROMPT_MAX, "prompt"), "custom", None)

    def add_image_style(self, label: Any, emoji: Any, image: str) -> dict[str, Any]:
        """Image styles are not applied until phase 2; their label doubles as the prompt."""
        label_text = clean_text(label, LABEL_MAX, "label")
        return self._add_style(label_text, emoji, label_text.lower(), "image", image)

    def update_item(self, kind: str, item_id: str, *, hidden: bool | None = None,
                    label: Any = None, emoji: Any = None, prompt: Any = None) -> dict[str, Any]:
        """Patch an item. Raises KeyError (unknown id), ValueError (invalid change) or
        OSError (could not save; nothing changed)."""
        with self._mutation():
            item = self._find(kind, item_id)
            changes: dict[str, Any] = {}
            if label is not None:
                changes["label"] = clean_text(label, LABEL_MAX, "label")
            if emoji is not None:
                changes["emoji"] = clean_emoji(emoji, kind)
            if prompt is not None:
                if item["kind"] != "custom":
                    raise ValueError("only custom styles have an editable prompt")
                changes["prompt"] = clean_text(prompt, PROMPT_MAX, "prompt")
            if hidden is not None:
                if not isinstance(hidden, bool):
                    raise ValueError("hidden must be true/false")
                others_visible = any(not it["hidden"] for it in self._items[kind] if it is not item)
                if hidden and not others_visible:
                    raise ValueError(f"at least one of the {kind} must stay visible")
                changes["hidden"] = hidden
            item.update(changes)
            result = copy.deepcopy(item)
        return result

    def delete_style(self, item_id: str) -> dict[str, Any]:
        """Remove a custom/image style and return it (caller deletes any image file)."""
        with self._mutation():
            item = self._find("styles", item_id)
            if item["kind"] == "preset":
                raise ValueError("preset styles can be hidden but not deleted")
            if not item["hidden"] and not any(
                    not it["hidden"] for it in self._items["styles"] if it is not item):
                raise ValueError("at least one style must stay visible")
            self._items["styles"].remove(item)
        return copy.deepcopy(item)

    def set_order(self, kind: str, ids: list[str]) -> None:
        """Move the given ids to the front in that order; unlisted items keep their order."""
        with self._mutation():
            self._check_kind(kind)
            if not isinstance(ids, list) or len(set(ids)) != len(ids):
                raise ValueError("ids must be a list without duplicates")
            by_id = {it["id"]: it for it in self._items[kind]}
            unknown = [i for i in ids if i not in by_id]
            if unknown:
                raise KeyError(unknown[0])
            listed = set(ids)
            self._items[kind] = [by_id[i] for i in ids] + [
                it for it in self._items[kind] if it["id"] not in listed]
