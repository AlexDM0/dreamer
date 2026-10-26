"""Socket bookkeeping on the event loop: the single kid session and admin watchers.

Every socket gets a ``Peer`` with its own outgoing queue and sender task, so results
arriving from the worker thread never interleave sends on one socket and a slow
client cannot stall anyone else. All methods here run on the event loop thread.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket

from . import protocol
from .gallery import LastFrame
from .styles import StyleStore

log = logging.getLogger("dreamer.session")

PEER_QUEUE_SIZE = 32
CLOSE_REPLACED = 4000  # app-specific close code: a newer kid socket took over
CLOSE_TIMEOUT = 2.0

_CLOSE = object()  # sentinel queued to make the sender task close the socket
_PICTURE = object()  # placeholder in the queue for the peer's newest picture


class Peer:
    """One WebSocket plus a bounded outgoing queue drained by a sender task."""

    def __init__(self, ws: WebSocket, label: str) -> None:
        self.ws = ws
        self.label = label
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=PEER_QUEUE_SIZE)
        # The newest picture and whether a _PICTURE slot for it is queued (send_picture).
        self._latest_picture: bytes | None = None
        self._picture_queued = False
        self._task = asyncio.create_task(self._pump(), name=f"peer-{label}")
        self.closing = False

    def send_json(self, msg: dict[str, Any]) -> None:
        self._enqueue(protocol.dumps(msg))

    def send_bytes(self, data: bytes) -> None:
        self._enqueue(data)

    def send_picture(self, frame: bytes) -> None:
        """Queue a PICTURE frame, replacing one still waiting: a client that falls behind
        (a phone on weak Wi-Fi) gets the newest picture next, not a backlog of old ones."""
        if self.closing:
            return
        self._latest_picture = frame
        if not self._picture_queued:
            self._picture_queued = True
            self._enqueue(_PICTURE)

    def _enqueue(self, item: Any) -> None:
        if self.closing:
            return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # The client is not keeping up: drop this message (pictures never pile up here,
            # see send_picture).
            log.debug("%s: send queue full; dropping message", self.label)
            if item is _PICTURE:
                self._picture_queued = False

    def close_soon(self, code: int = 1000, farewell: dict[str, Any] | None = None) -> None:
        """Send everything already queued (then ``farewell``, if given) and close.

        The farewell message and the close always get a queue slot: stale queued
        frames are dropped to make room, since they matter less than telling the page
        why it is being closed.
        """
        if self.closing:
            return
        self.closing = True
        tail: list[Any] = [protocol.dumps(farewell)] if farewell is not None else []
        tail.append((_CLOSE, code))
        while self._queue.maxsize - self._queue.qsize() < len(tail):
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        for item in tail:
            self._queue.put_nowait(item)

    async def finish(self) -> None:
        """Stop the sender task (used when the receive loop ends)."""
        self.closing = True
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _pump(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                if item is _PICTURE:
                    item, self._latest_picture = self._latest_picture, None
                    self._picture_queued = False
                    if item is None:
                        continue
                if isinstance(item, tuple) and item and item[0] is _CLOSE:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(self.ws.close(code=item[1]), CLOSE_TIMEOUT)
                    return
                if isinstance(item, str):
                    await self.ws.send_text(item)
                else:
                    await self.ws.send_bytes(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # socket went away mid-send
            log.debug("%s: send failed: %s", self.label, exc)
            self.closing = True


@dataclass(frozen=True)
class FrameOrigin:
    """The ``FrameJob.owner`` tag for kid sketches: which socket sent it, plus what the
    picture will be made from, so the result can become the session's ``last_frame``.
    ``epoch`` is the session's drawing epoch when the sketch arrived: a picture finished
    after the kid cleared / started anew / opened another drawing belongs to the old
    drawing and must not become ``last_frame``."""
    peer: Peer
    sketch_png: bytes
    prompt: str
    topic_id: str | None
    style_id: str | None
    epoch: int = 0
    # The page's canvas when the sketch was sent (None: an older page sending opaque
    # sketches on white); see SessionManager.canvas.
    canvas: dict[str, str] | None = None


@dataclass
class Selection:
    topic: str | None
    style: str | None
    view_mode: str
    kid_theme: str = "dark"


class SessionManager:
    """Holds the active kid Peer, admin watcher Peers, and the current selection.

    The selection is server-side so it survives kid reconnects.
    """

    def __init__(self, styles: StyleStore, default_view_mode: str,
                 backend_name: Callable[[], str], kid_theme: str = "dark") -> None:
        self.styles = styles
        self.selection = Selection(None, None, default_view_mode, kid_theme)
        self._backend_name = backend_name
        self.kid: Peer | None = None
        self.admins: set[Peer] = set()
        self.last_status: dict[str, Any] = protocol.status_msg("busy", "starting")
        # Newest PICTURE frame, so a freshly opened admin page shows it straight away.
        self.last_picture: bytes | None = None
        # Gallery (SPEC-addendum-gallery): what is on the kid's screen right now (Save uses
        # exactly this, no regeneration), and the saved artwork being continued, if any.
        self.last_frame: LastFrame | None = None
        self.continuing_id: str | None = None
        # Bumped whenever the canvas starts holding a different drawing (clear, new,
        # continue, draft restore, a new page); see FrameOrigin.epoch.
        self.epoch = 0
        # {"background": "#rrggbb", "orientation": ...} as the page last reported it
        # ({"t":"canvas"}); None until then, meaning an older page with opaque sketches.
        self.canvas: dict[str, str] | None = None

    def new_drawing(self, continuing_id: str | None = None) -> None:
        """The canvas now holds another drawing: forget what is on screen, ignore pictures
        still being made from the old one, and remember which artwork (if any) it is."""
        self.epoch += 1
        self.last_frame = None
        self.continuing_id = continuing_id

    # ----- kid -----------------------------------------------------------------------

    def attach_kid(self, peer: Peer) -> None:
        """Make ``peer`` the kid session; the previous one is told and closed."""
        old, self.kid = self.kid, peer
        if old is not None:
            log.info("new kid socket; replacing the old one")
            old.close_soon(CLOSE_REPLACED, farewell={"t": "replaced"})

    def detach_kid(self, peer: Peer) -> None:
        if self.kid is peer:
            self.kid = None

    def current(self) -> dict[str, Any]:
        """Selection resolved against the visible lists (hidden items fall back)."""
        return {
            "topic": self.styles.resolve("topics", self.selection.topic),
            "style": self.styles.resolve("styles", self.selection.style),
            "view_mode": self.selection.view_mode,
            "kid_theme": self.selection.kid_theme,
        }

    def lists_msg(self, t: str) -> dict[str, Any]:
        """The ``ready``/``styles`` payload for the kid page."""
        msg: dict[str, Any] = {"t": t}
        if t == "ready":
            msg["v"] = protocol.PROTOCOL_VERSION
        msg.update(self.styles.public_list())
        msg.update(self.current())
        msg["backend"] = self._backend_name()
        return msg

    def select(self, *, topic: str | None = None, style: str | None = None,
               view_mode: str | None = None, kid_theme: str | None = None) -> None:
        if topic is not None:
            self.selection.topic = topic
        if style is not None:
            self.selection.style = style
        if view_mode is not None:
            self.selection.view_mode = view_mode
        if kid_theme is not None:
            self.selection.kid_theme = kid_theme

    def broadcast_styles(self) -> None:
        if self.kid is not None:
            self.kid.send_json(self.lists_msg("styles"))

    # ----- fan-out ---------------------------------------------------------------------

    def _everyone(self) -> list[Peer]:
        peers = list(self.admins)
        if self.kid is not None:
            peers.append(self.kid)
        return peers

    def send_picture(self, seq: int, jpeg: bytes, owner: Peer | None = None) -> None:
        """Fan a picture out to admins, and to the kid if it made the sketch.

        ``owner`` is the kid Peer whose sketch this picture came from. A picture that was
        still being made when that socket got replaced is not sent to the new socket:
        its seq belongs to the old page's numbering and would confuse the new one.
        """
        frame = protocol.pack_picture(seq, jpeg)
        self.last_picture = frame
        for peer in list(self.admins):
            peer.send_picture(frame)
        if self.kid is not None and (owner is None or owner is self.kid):
            self.kid.send_picture(frame)

    def broadcast_status(self, state: str, msg: str = "") -> None:
        self.last_status = protocol.status_msg(state, msg)
        for peer in self._everyone():
            peer.send_json(self.last_status)

    def broadcast_admin(self, msg: dict[str, Any]) -> None:
        for peer in list(self.admins):
            peer.send_json(msg)

    # ----- admins ----------------------------------------------------------------------

    def add_admin(self, peer: Peer) -> None:
        self.admins.add(peer)

    def remove_admin(self, peer: Peer) -> None:
        self.admins.discard(peer)
