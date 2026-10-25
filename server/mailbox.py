"""Single-slot, thread-safe mailbox: "latest frame wins"."""
from __future__ import annotations

import threading
from typing import Any


class Mailbox:
    """Holds at most one item. ``put`` overwrites; ``take`` empties the slot."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Any = None
        self._full = False

    def put(self, item: Any) -> None:
        with self._cond:
            self._item = item
            self._full = True
            self._cond.notify()

    def take(self, timeout: float | None = None) -> Any:
        """Return the item (emptying the slot), or None if nothing arrived in time."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._full, timeout=timeout):
                return None
            item, self._item, self._full = self._item, None, False
            return item

    def clear(self) -> None:
        with self._cond:
            self._item, self._full = None, False

