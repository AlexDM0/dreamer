"""FastAPI application: kid WebSocket, admin API + watcher socket, static pages."""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import io
import logging
import queue
import secrets
import sqlite3
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable

from fastapi import (APIRouter, Depends, FastAPI, File, Form, HTTPException, Request,
                     Response, UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import protocol
from .auth import COOKIE_MAX_AGE, COOKIE_NAME, AdminAuth, LockedOut, load_or_create_secret
from .config import DEFAULT_PIN, ORIENTATIONS, VIEW_MODES, Config, canvas_dims
from .gallery import FILES, DraftAutosaver, GalleryStore, LastFrame, is_blank_sketch, valid_id
from .imaging import COLOR_RE
from .mailbox import Mailbox
from .params import LIMITS, GenParams, ParamsStore, defaults_from_config
from .session import FrameOrigin, Peer, SessionManager
from .styles import KINDS, LABEL_MAX, StyleStore, clean_text
from .worker import FrameJob, GpuWorker

log = logging.getLogger("dreamer.app")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STYLE_IMAGES_DIR = "style_images"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
# Phone photos go up to ~48 MP; JPEGs are draft-decoded at reduced size anyway.
MAX_UPLOAD_PIXELS = 50_000_000
UPLOAD_FORMATS = ["PNG", "JPEG", "WEBP", "GIF"]
# Request body caps, enforced before anything reads the body (see AdminGate).
UPLOAD_PATH = "/api/admin/style-image"
UPLOAD_BODY_MAX = MAX_UPLOAD_BYTES + 1024 * 1024  # file + multipart overhead
JSON_BODY_MAX = 64 * 1024
PUBLIC_ADMIN_PATHS = ("/api/admin/login", "/api/admin/logout")
STORED_IMAGE_MAX_SIDE = 1024
THEMES = ("dark", "light")  # kid page look, picked on the admin page
SCREEN_TIMEOUT = 60.0
STATS_INTERVAL = 1.0
IMAGE_STYLE_NOTE = ("Style pictures need phase 2 (SD 1.5 + IP-Adapter); "
                    "stored but not applied yet.")
NO_CACHE = "no-cache"
IMMUTABLE = "public, max-age=31536000, immutable"  # artwork file URLs carry ?v=updated_at
GALLERY_PAGE = 24
DRAFT_INTERVAL = 10.0  # seconds between draft autosaves while the kid draws
NOTHING_TO_SAVE = "Nothing to save yet"


class LoginBody(BaseModel):
    pin: str


class NewStyleBody(BaseModel):
    label: str
    emoji: str | None = None
    prompt: str


class PatchBody(BaseModel):
    hidden: bool | None = None
    label: str | None = None
    emoji: str | None = None
    prompt: str | None = None


class OrderBody(BaseModel):
    kind: str
    ids: list[str]


class SelectBody(BaseModel):
    topic: str | None = None
    style: str | None = None
    view_mode: str | None = None
    theme: str | None = None


class TitleBody(BaseModel):
    title: str | None = None


class _Runtime:
    """Things that exist only while the app is running (created in lifespan)."""

    worker: GpuWorker | None = None
    drafts: DraftAutosaver | None = None


def _bad_request(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(404, detail="not found")
    return HTTPException(400, detail=str(exc))


def _load_upload(data: bytes) -> Image.Image:
    """Decode an admin upload into an upright, bounded-size RGB image.

    Only common photo formats are decoded (no exotic Pillow plugins). Raises ValueError
    if the data is not such a picture.
    """
    try:
        img = Image.open(io.BytesIO(data), formats=UPLOAD_FORMATS)
        if img.width * img.height > MAX_UPLOAD_PIXELS:
            raise ValueError("picture is too big")
        # JPEG: let the decoder downscale by up to 8x while decoding (much less memory).
        img.draft("RGB", (STORED_IMAGE_MAX_SIDE, STORED_IMAGE_MAX_SIDE))
        img.load()
        # Phone photos store the sensor orientation plus an EXIF tag; bake it in, since
        # the stored PNG keeps no EXIF.
        img = ImageOps.exif_transpose(img)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("that file is not a picture") from exc
    img = img.convert("RGB")
    img.thumbnail((STORED_IMAGE_MAX_SIDE, STORED_IMAGE_MAX_SIDE))
    return img


class AdminGate:
    """ASGI middleware for ``/api/*``: auth and body-size checks before the body is read.

    FastAPI parses JSON / multipart bodies before running route dependencies, so without
    this an unauthenticated client could make the server spool arbitrarily large uploads.
    Here protected admin routes answer 401 without reading the body, a too-large
    Content-Length gets 413 straight away, and a streamed (chunked) body is cut off with
    413 as soon as it passes the cap.
    """

    def __init__(self, app: ASGIApp, auth: AdminAuth) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        if path.startswith("/api/admin/") and path not in PUBLIC_ADMIN_PATHS:
            if not self.auth.verify_cookie(_cookie(headers.get("cookie", ""), COOKIE_NAME)):
                await JSONResponse({"detail": "Not logged in"}, 401)(scope, receive, send)
                return
        cap = UPLOAD_BODY_MAX if path == UPLOAD_PATH else JSON_BODY_MAX
        length = headers.get("content-length", "")
        if length.isdigit() and int(length) > cap:
            await JSONResponse({"detail": "Request is too big"}, 413)(scope, receive, send)
            return

        seen = 0

        async def capped_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > cap:
                    # FastAPI re-raises HTTPException from body parsing as-is -> 413.
                    raise HTTPException(413, detail="Request is too big")
            return message

        await self.app(scope, capped_receive, send)


def _cookie(header: str, name: str) -> str | None:
    """Value of cookie ``name`` from a Cookie header (no quoting needed for ours)."""
    for part in header.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key == name:
            return value
    return None


def create_app(config: Config, backend: Any = None) -> FastAPI:
    """Build the app. With ``backend=None`` it is created by ``make_backend(config)``
    on the GPU worker thread (and rebuilt there after repeated failures)."""
    if config.admin_pin == DEFAULT_PIN:
        log.warning("admin_pin is the default %r; set admin_pin in config.yaml", DEFAULT_PIN)

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    styles = StyleStore(data_dir)
    auth = AdminAuth(config.admin_pin, load_or_create_secret(config.secret_key, data_dir))
    params_store = ParamsStore(data_dir, defaults_from_config(config))
    gallery = GalleryStore(data_dir)
    mailbox = Mailbox()
    runtime = _Runtime()

    def backend_name() -> str:
        if runtime.worker is not None:
            return runtime.worker.backend_name
        return backend.name if backend is not None else config.backend

    session = SessionManager(styles, config.default_view_mode, backend_name)

    def worker_stats() -> dict[str, Any]:
        if runtime.worker is None:
            return {"state": "stopped", "backend": backend_name()}
        return runtime.worker.stats()

    async def stats_loop() -> None:
        while True:
            await asyncio.sleep(STATS_INTERVAL)
            if session.admins:
                session.broadcast_admin({"t": "stats", **worker_stats(), **session.current()})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()

        def threadsafe(fn: Callable[..., None]) -> Callable[..., None]:
            """Wrap an event-loop-only callable so the worker thread can call it."""
            def call(*args: Any) -> None:
                try:
                    loop.call_soon_threadsafe(fn, *args)
                except RuntimeError:  # loop already closed during shutdown
                    pass
            return call

        def params_reverted(previous: GenParams) -> None:
            """The worker could not apply a change: persist and show what is live."""
            try:
                params_store.set(previous)
            except OSError:
                log.exception("could not save reverted params")
            session.broadcast_admin(params_msg(previous))

        factory = None
        if backend is None:
            from .backends import make_backend
            factory = partial(make_backend, config)
        runtime.drafts = DraftAutosaver(gallery, lambda: session.last_frame,
                                        lambda: session.continuing_id, DRAFT_INTERVAL)
        try:  # files a crash or a locked delete left behind
            swept = await asyncio.to_thread(gallery.sweep)
            if swept:
                log.info("removed %d stray gallery files/folders", swept)
        except (OSError, sqlite3.Error) as exc:
            log.error("could not tidy the gallery folder: %s", exc)
        worker = GpuWorker(backend, mailbox, threadsafe(on_picture),
                           threadsafe(session.broadcast_status), config,
                           backend_factory=factory, params=params_store.get(),
                           on_params_error=threadsafe(params_reverted))
        runtime.worker = worker
        worker.start()
        stats_task = asyncio.create_task(stats_loop())
        log.info("Dreamer started (backend=%s)", backend_name())
        try:
            yield
        finally:
            stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stats_task
            await runtime.drafts.flush()  # keep the drawing for the next start
            await asyncio.to_thread(worker.stop)
            runtime.worker = None
            log.info("Dreamer stopped")

    app = FastAPI(title="Dreamer", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.styles = styles
    app.state.session = session
    app.state.mailbox = mailbox
    app.state.auth = auth
    app.state.runtime = runtime
    app.state.params = params_store
    app.state.gallery = gallery
    app.add_middleware(AdminGate, auth=auth)

    # ----- caching: always revalidate pages and modules so the iPad picks up updates -----

    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path in ("/", "/admin") or path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", NO_CACHE)
        return response

    def page(name: str) -> FileResponse:
        path = STATIC_DIR / name
        if not path.is_file():
            raise HTTPException(404, detail=f"{name} is missing")
        return FileResponse(path, media_type="text/html", headers={"Cache-Control": NO_CACHE})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return page("index.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_page() -> FileResponse:
        return page("admin.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        path = STATIC_DIR / "favicon.svg"
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/svg+xml")

    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        state = runtime.worker.state if runtime.worker is not None else "stopped"
        return {"ok": True, "backend": backend_name(), "state": state}

    # ----- artworks: public (kid LAN) -------------------------------------------------

    def page_args(limit: int, offset: int) -> tuple[int, int]:
        return max(1, min(limit, 100)), max(0, offset)

    def artwork_file(art: Any, name: str, request: Request,
                     download: bool = False) -> FileResponse:
        if art is None or name not in FILES:
            raise HTTPException(404, detail="not found")
        try:
            path = gallery.file_path(art, name)
        except (KeyError, ValueError):
            raise HTTPException(404, detail="not found")
        if not path.is_file():
            raise HTTPException(404, detail="not found")
        # Versioned URLs (?v=updated_at) never change content; bare ones must revalidate,
        # and so must an outdated ?v (it gets the current files, not that version's).
        current = request.query_params.get("v") == art.version
        headers = {"Cache-Control": IMMUTABLE if current else NO_CACHE}
        filename = None
        if download:
            stem = "".join(c if c.isalnum() else "-" for c in (art.title or "").lower()).strip("-")
            filename = f"dreamer-{stem or art.id}-{name}"
        return FileResponse(path, media_type=FILES[name], headers=headers, filename=filename)

    @app.get("/api/artworks")
    async def list_artworks(limit: int = GALLERY_PAGE, offset: int = 0) -> dict[str, Any]:
        limit, offset = page_args(limit, offset)
        rows = await asyncio.to_thread(gallery.list, limit + 1, offset)
        return {"artworks": [a.public() for a in rows[:limit]], "more": len(rows) > limit,
                "offset": offset}

    @app.get("/api/artworks/{artwork_id}")
    async def get_artwork(artwork_id: str) -> dict[str, Any]:
        art = await asyncio.to_thread(gallery.get, artwork_id)
        if art is None or art.is_draft:
            raise HTTPException(404, detail="not found")
        return art.public()

    @app.get("/api/artworks/{artwork_id}/{name}")
    async def get_artwork_file(artwork_id: str, name: str, request: Request) -> FileResponse:
        # Drafts are served too: the kid page restores its draft sketch from here.
        art = await asyncio.to_thread(gallery.get, artwork_id)
        return artwork_file(art, name, request)

    # ----- kid socket ----------------------------------------------------------------

    def on_picture(seq: int, jpeg: bytes, owner: Any) -> None:
        """Worker result (on the loop): fan out, and remember it as what is on screen."""
        origin = owner if isinstance(owner, FrameOrigin) else None
        session.send_picture(seq, jpeg, origin.peer if origin else owner)
        if origin is None or origin.peer is not session.kid or origin.epoch != session.epoch:
            return  # another socket's, or made from a drawing the kid has left behind
        worker = runtime.worker
        params = worker.params.to_dict() if worker is not None else {}
        if origin.canvas is not None:
            params = {**params, "canvas": origin.canvas}  # stored with the artwork
        session.last_frame = LastFrame(seq, origin.sketch_png, jpeg, origin.prompt,
                                       origin.topic_id, origin.style_id, params)
        if runtime.drafts is not None:
            runtime.drafts.touch()

    def kid_error(peer: Peer, msg: str) -> None:
        """A one-off error for the kid page, then the real backend state again."""
        peer.send_json(protocol.status_msg("error", msg))
        peer.send_json(session.last_status)

    def draft_info() -> dict[str, Any] | None:
        art = runtime.drafts.current() if runtime.drafts is not None else None
        if art is None:
            return None
        return {"id": art.id, "sketch": art.file_url("sketch.png"), "source": art.source_id,
                **art.canvas()}

    def live_artwork(artwork_id: Any) -> str | None:
        """``artwork_id`` if it names a saved (non-draft, not deleted) artwork, else None."""
        art = gallery.get(artwork_id) if valid_id(artwork_id) else None
        return art.id if art is not None and not art.is_draft else None

    async def kid_hello(peer: Peer, continuing: Any) -> None:
        drafts = runtime.drafts
        if drafts is not None:
            await drafts.flush()  # a reload must get the drawing it just left
        # What is being continued belongs to the page's canvas, not to the server: a
        # reconnecting page says which saved artwork it holds (if any); a freshly loaded
        # page holds nothing until it restores the draft (see kid_restored). Until the
        # page's own sketch comes back, nothing is on screen.
        session.new_drawing(await asyncio.to_thread(live_artwork, continuing))
        msg = session.lists_msg("ready")
        msg["canvas"] = {name: list(dims) for name, dims in canvas_dims(config).items()}
        msg["draft"] = await asyncio.to_thread(draft_info)
        peer.send_json(msg)
        peer.send_json(session.last_status)

    async def kid_save(peer: Peer) -> None:
        frame = session.last_frame
        if frame is None or await asyncio.to_thread(is_blank_sketch, frame.sketch_png):
            kid_error(peer, NOTHING_TO_SAVE)
            return
        drafts = runtime.drafts
        lock = drafts.lock if drafts is not None else asyncio.Lock()
        try:
            async with lock:
                art = None
                if session.continuing_id is not None:
                    try:
                        art = await asyncio.to_thread(gallery.update, session.continuing_id,
                                                      frame.sketch_png, frame.picture_jpeg,
                                                      frame.meta())
                    except KeyError:
                        art = None  # deleted meanwhile: save it as a new artwork
                if art is None:
                    art = await asyncio.to_thread(gallery.save, frame.sketch_png,
                                                  frame.picture_jpeg, frame.meta())
        except (OSError, ValueError, sqlite3.Error) as exc:
            log.error("could not save artwork: %s", exc)
            kid_error(peer, "Could not save the picture. Try again.")
            return
        # Further saves of this drawing update the same artwork instead of piling up copies.
        session.continuing_id = art.id
        if drafts is not None:
            drafts.touch()  # the draft now records which artwork it continues
        log.info("saved artwork %s", art.id)
        peer.send_json({"t": "saved", "artwork": art.public()})
        session.broadcast_admin({"t": "gallery"})

    async def kid_fresh(continuing_id: str | None = None) -> None:
        """Clear / New drawing (or a saved one put on the canvas): nothing on screen, no
        draft of the old drawing, and only ``continuing_id`` being continued."""
        mailbox.clear()
        session.new_drawing(continuing_id)  # before any await: stale pictures are ignored
        if runtime.drafts is not None:
            await runtime.drafts.discard()

    async def kid_continue(peer: Peer, artwork_id: Any) -> None:
        """"Keep drawing" asked: send the sketch to load. Nothing changes on the server until
        the page says it has the sketch on its canvas ({"t":"continuing"}), so a sketch
        that fails to load cannot make the next Save overwrite that artwork."""
        art = await asyncio.to_thread(gallery.get, artwork_id) if valid_id(artwork_id) else None
        if art is None or art.is_draft:
            kid_error(peer, "That picture is not there any more")
            return
        for t, kind, item_id in (("topic", "topics", art.topic_id), ("style", "styles", art.style_id)):
            item = styles.get(kind, item_id) if item_id else None
            if item is not None and not item["hidden"]:
                session.select(**{t: item_id})
        cur = session.current()
        peer.send_json({"t": "continue", "id": art.id, "sketch": art.file_url("sketch.png"),
                        "topic": cur["topic"], "style": cur["style"], **art.canvas()})

    async def kid_continuing(artwork_id: Any) -> None:
        """The page loaded a saved artwork's sketch: Save updates that artwork from now on
        (or saves a new one, if it was deleted meanwhile). The previous drawing's draft
        goes, like on New: the kid already agreed to leave that drawing."""
        await kid_fresh(await asyncio.to_thread(live_artwork, artwork_id))

    async def kid_restored(draft_id: Any) -> None:
        """The page put the draft back on its canvas: it continues what the draft continued."""
        mailbox.clear()
        drafts = runtime.drafts
        draft = await asyncio.to_thread(drafts.current) if drafts is not None else None
        source = draft.source_id if draft is not None and draft.id == draft_id else None
        session.new_drawing(await asyncio.to_thread(live_artwork, source) if source else None)

    def kid_canvas(background: Any, orientation: Any) -> None:
        """The page's background layer colour and drawing orientation. From now on its
        sketches are stroke layers (transparent background). Bad values are ignored."""
        if not isinstance(background, str) or not COLOR_RE.fullmatch(background):
            return
        if orientation not in ORIENTATIONS:
            return
        session.canvas = {"background": background.lower(), "orientation": orientation}

    async def handle_kid_text(peer: Peer, text: str) -> None:
        msg = protocol.parse_client_text(text)
        if msg is None:
            return
        t = msg["t"]
        if t == "hello":
            await kid_hello(peer, msg.get("continuing"))
        elif t in ("topic", "style"):
            kind = "topics" if t == "topic" else "styles"
            item_id = msg.get("id")
            item = styles.get(kind, item_id) if isinstance(item_id, str) else None
            if item is not None and not item["hidden"]:
                session.select(**{t: item_id})
        elif t in ("clear", "new"):
            await kid_fresh()
        elif t == "save":
            await kid_save(peer)
        elif t == "continue":
            await kid_continue(peer, msg.get("id"))
        elif t == "continuing":
            await kid_continuing(msg.get("id"))
        elif t == "restored":
            await kid_restored(msg.get("id"))
        elif t == "canvas":
            kid_canvas(msg.get("background"), msg.get("orientation"))
        elif t == "ping":
            peer.send_json({"t": "pong"})
        # Unknown types are ignored on purpose (forward compatibility).

    def handle_kid_bytes(peer: Peer, data: bytes) -> None:
        if len(data) > protocol.HEADER_SIZE + protocol.MAX_MESSAGE_BYTES:
            log.warning("ignoring oversized frame (%d bytes)", len(data))
            return
        try:
            msg_type, seq, payload = protocol.unpack(data)
        except protocol.ProtocolError as exc:
            log.warning("ignoring malformed frame: %s", exc)
            return
        if msg_type != protocol.SKETCH:
            log.warning("ignoring frame type 0x%02x from kid", msg_type)
            return
        # The prompt is built server-side from the selection; the kid never sends text.
        cur = session.current()
        prompt = styles.build_prompt(cur["topic"], cur["style"])
        canvas = session.canvas
        origin = FrameOrigin(peer, payload, prompt, cur["topic"], cur["style"], session.epoch,
                             canvas)
        mailbox.put(FrameJob(seq, payload, prompt, owner=origin,
                             background=canvas["background"] if canvas else None,
                             orientation=canvas["orientation"] if canvas else None))

    def same_origin(ws: WebSocket) -> bool:
        """Browsers always send Origin on WebSockets: refuse sockets opened by pages
        served from elsewhere (a cross-site page could otherwise take over the session).
        Non-browser clients without an Origin header are allowed."""
        origin = ws.headers.get("origin")
        if not origin:
            return True
        host = ws.headers.get("host", "")
        return origin.split("://", 1)[-1].rstrip("/").lower() == host.lower()

    @app.websocket("/ws")
    async def kid_ws(ws: WebSocket) -> None:
        if not same_origin(ws):
            log.warning("refusing kid socket from foreign origin %r", ws.headers.get("origin"))
            await ws.close(code=1008)
            return
        await ws.accept()
        peer = Peer(ws, "kid")
        session.attach_kid(peer)
        mailbox.clear()  # a stale sketch from the previous socket is not wanted
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if session.kid is not peer:
                    continue  # replaced: ignore anything the old page still sends
                if message.get("bytes") is not None:
                    handle_kid_bytes(peer, message["bytes"])
                elif message.get("text") is not None:
                    await handle_kid_text(peer, message["text"])
        except (WebSocketDisconnect, RuntimeError):
            pass  # RuntimeError: receive after the socket was closed
        finally:
            session.detach_kid(peer)
            await peer.finish()
            if runtime.drafts is not None and session.kid is None:
                with contextlib.suppress(Exception):
                    await runtime.drafts.flush()  # page closed: keep what was drawn

    # ----- admin watcher socket ------------------------------------------------------

    @app.websocket("/ws/admin")
    async def admin_ws(ws: WebSocket) -> None:
        if not auth.verify_cookie(ws.cookies.get(COOKIE_NAME)):
            await ws.close(code=1008)  # before accept -> HTTP 403
            return
        await ws.accept()
        peer = Peer(ws, "admin")
        session.add_admin(peer)
        peer.send_json(session.last_status)
        peer.send_json({"t": "stats", **worker_stats(), **session.current()})
        if session.last_picture is not None:
            peer.send_bytes(session.last_picture)
        try:
            while True:  # read-only: drain and ignore anything the client sends
                if (await ws.receive())["type"] == "websocket.disconnect":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session.remove_admin(peer)
            await peer.finish()

    # ----- admin HTTP API ------------------------------------------------------------

    @app.post("/api/admin/login")
    async def login(body: LoginBody, request: Request, response: Response) -> dict[str, Any]:
        client = request.client.host if request.client else ""
        try:
            ok = auth.check_pin(body.pin, client)
        except LockedOut as exc:
            raise HTTPException(429, detail="Too many tries. Wait a little.",
                                headers={"Retry-After": str(int(exc.retry_after) + 1)})
        if not ok:
            raise HTTPException(401, detail="Wrong PIN")
        response.set_cookie(COOKIE_NAME, auth.make_cookie(), max_age=COOKIE_MAX_AGE,
                            httponly=True, samesite="strict", path="/")
        return {"ok": True}

    @app.post("/api/admin/logout")
    async def logout(response: Response) -> dict[str, Any]:
        response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="strict")
        return {"ok": True}

    def require_admin(request: Request) -> None:
        if not auth.verify_cookie(request.cookies.get(COOKIE_NAME)):
            raise HTTPException(401, detail="Not logged in")

    admin = APIRouter(prefix="/api/admin", dependencies=[Depends(require_admin)])

    def mutated() -> None:
        """Push the new lists/defaults to the kid page after any admin change."""
        session.broadcast_styles()

    def params_msg(params: GenParams) -> dict[str, Any]:
        return {"t": "params", "params": params.to_dict()}

    def save_failed(exc: Exception) -> HTTPException:
        log.error("could not save settings: %s", exc)
        return HTTPException(503, detail="Could not save the change on the computer. Try again.")

    @admin.get("/state")
    async def get_state() -> dict[str, Any]:
        return {**styles.admin_list(), **session.current(), "backend": backend_name(),
                "stats": worker_stats()}

    @admin.post("/styles")
    async def add_style(body: NewStyleBody) -> dict[str, Any]:
        try:
            item = styles.add_custom_style(body.label, body.emoji, body.prompt)
        except ValueError as exc:
            raise _bad_request(exc)
        except OSError as exc:
            raise save_failed(exc)
        mutated()
        return {"ok": True, "style": item}

    @admin.patch("/items/{kind}/{item_id}")
    async def patch_item(kind: str, item_id: str, body: PatchBody) -> dict[str, Any]:
        if kind not in KINDS:
            raise HTTPException(404, detail="not found")
        try:
            item = styles.update_item(kind, item_id, hidden=body.hidden, label=body.label,
                                      emoji=body.emoji, prompt=body.prompt)
        except (KeyError, ValueError) as exc:
            raise _bad_request(exc)
        except OSError as exc:
            raise save_failed(exc)
        mutated()
        return {"ok": True, "item": item}

    @admin.delete("/items/styles/{item_id}")
    async def delete_style(item_id: str) -> dict[str, Any]:
        try:
            item = styles.delete_style(item_id)
        except (KeyError, ValueError) as exc:
            raise _bad_request(exc)
        except OSError as exc:
            raise save_failed(exc)
        if item.get("image"):
            images = (data_dir / STYLE_IMAGES_DIR).resolve()
            path = (data_dir / item["image"]).resolve()
            if path.parent == images:  # never delete outside the images folder
                path.unlink(missing_ok=True)
        mutated()
        return {"ok": True}

    @admin.post("/order")
    async def set_order(body: OrderBody) -> dict[str, Any]:
        try:
            styles.set_order(body.kind, body.ids)
        except (KeyError, ValueError) as exc:
            raise _bad_request(exc)
        except OSError as exc:
            raise save_failed(exc)
        mutated()
        return {"ok": True}

    @admin.post("/select")
    async def select(body: SelectBody) -> dict[str, Any]:
        if body.topic is not None and styles.get("topics", body.topic) is None:
            raise HTTPException(400, detail="unknown topic")
        if body.style is not None and styles.get("styles", body.style) is None:
            raise HTTPException(400, detail="unknown style")
        if body.view_mode is not None and body.view_mode not in VIEW_MODES:
            raise HTTPException(400, detail=f"view_mode must be one of {VIEW_MODES}")
        if body.theme is not None and body.theme not in THEMES:
            raise HTTPException(400, detail=f"theme must be one of {THEMES}")
        session.select(topic=body.topic, style=body.style, view_mode=body.view_mode,
                       theme=body.theme)
        mutated()
        return {"ok": True, **session.current()}

    @admin.post("/style-image")
    async def style_image(file: UploadFile = File(...), label: str = Form(...),
                          emoji: str = Form("")) -> dict[str, Any]:
        try:
            clean_text(label, LABEL_MAX, "label")
        except ValueError as exc:
            raise _bad_request(exc)
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(415, detail="Please choose a picture file")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="Picture is too big (max 8 MB)")
        try:
            img = await asyncio.to_thread(_load_upload, data)
        except ValueError as exc:
            raise _bad_request(exc)

        worker = runtime.worker
        if worker is None:
            raise HTTPException(503, detail="Server is not ready")
        try:
            future = worker.submit_upload(img)
        except queue.Full:
            raise HTTPException(503, detail="Busy, try again in a moment")
        try:
            safe = await asyncio.wait_for(asyncio.wrap_future(future), SCREEN_TIMEOUT)
        except (asyncio.TimeoutError, concurrent.futures.CancelledError):
            future.cancel()
            raise HTTPException(503, detail="Checking the picture took too long")
        except Exception as exc:
            log.warning("upload screening error: %s", exc)
            raise HTTPException(503, detail="Could not check the picture")
        if not safe:
            raise HTTPException(422, detail="That picture can't be used as a style")

        rel = f"{STYLE_IMAGES_DIR}/{secrets.token_hex(8)}.png"
        try:
            (data_dir / STYLE_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(img.save, data_dir / rel, "PNG")
            item = styles.add_image_style(label, emoji or None, rel)
        except (ValueError, OSError) as exc:
            (data_dir / rel).unlink(missing_ok=True)  # no orphaned picture files
            raise save_failed(exc) if isinstance(exc, OSError) else _bad_request(exc)
        mutated()
        result: dict[str, Any] = {"ok": True, "style": item}
        if not getattr(worker.backend, "supports_image_styles", False):
            result["note"] = IMAGE_STYLE_NOTE
        return result

    # ----- live diffusion params (SPEC-addendum-params) ------------------------------

    def params_payload(params: GenParams) -> dict[str, Any]:
        return {"params": params.to_dict(), "limits": LIMITS,
                "defaults": params_store.defaults.to_dict()}

    def params_changed(params: GenParams) -> None:
        if runtime.worker is not None:
            runtime.worker.set_params(params)
        session.broadcast_admin(params_msg(params))  # keeps other admin phones in sync

    @admin.get("/params")
    async def get_params() -> dict[str, Any]:
        return params_payload(params_store.get())

    @admin.patch("/params")
    async def patch_params(request: Request) -> dict[str, Any]:
        try:
            changes = await request.json()
        except ValueError:
            raise HTTPException(400, detail="expected a JSON object")
        try:
            params = params_store.patch(changes)
        except ValueError as exc:
            raise _bad_request(exc)
        except OSError as exc:
            raise save_failed(exc)
        params_changed(params)
        return params_payload(params)

    @admin.post("/params/reset")
    async def reset_params() -> dict[str, Any]:
        try:
            params = params_store.set(params_store.defaults)
        except OSError as exc:
            raise save_failed(exc)
        params_changed(params)
        return params_payload(params)

    # ----- artworks: admin (SPEC-addendum-gallery) -----------------------------------

    def gallery_changed() -> None:
        """Tell the kid page (and other admin phones) to refresh their galleries."""
        if session.kid is not None:
            session.kid.send_json({"t": "gallery"})
        session.broadcast_admin({"t": "gallery"})

    def artwork_404() -> HTTPException:
        return HTTPException(404, detail="not found")

    @admin.get("/artworks")
    async def admin_list_artworks(trash: bool = False, limit: int = GALLERY_PAGE,
                                  offset: int = 0) -> dict[str, Any]:
        limit, offset = page_args(limit, offset)

        def read() -> dict[str, Any]:
            rows = gallery.list(limit + 1, offset, deleted=trash)
            return {"artworks": [a.admin() for a in rows[:limit]], "more": len(rows) > limit,
                    "offset": offset, "total": gallery.count(),
                    "trash": gallery.count(deleted=True)}
        return await asyncio.to_thread(read)

    @admin.get("/artworks/{artwork_id}/{name}")
    async def admin_artwork_file(artwork_id: str, name: str, request: Request,
                                 download: bool = False) -> FileResponse:
        art = await asyncio.to_thread(gallery.get, artwork_id, True)
        return artwork_file(art, name, request, download)

    @admin.patch("/artworks/{artwork_id}")
    async def rename_artwork(artwork_id: str, body: TitleBody) -> dict[str, Any]:
        try:
            art = await asyncio.to_thread(gallery.rename, artwork_id, body.title)
        except KeyError:
            raise artwork_404()
        except ValueError as exc:
            raise _bad_request(exc)
        except (OSError, sqlite3.Error) as exc:
            raise save_failed(exc)
        gallery_changed()
        return {"ok": True, "artwork": art.admin()}

    @admin.delete("/artworks/{artwork_id}")
    async def delete_artwork(artwork_id: str) -> dict[str, Any]:
        try:
            art = await asyncio.to_thread(gallery.soft_delete, artwork_id)
        except KeyError:
            raise artwork_404()
        except (OSError, sqlite3.Error) as exc:
            raise save_failed(exc)
        gallery_changed()
        return {"ok": True, "artwork": art.admin()}

    @admin.post("/artworks/{artwork_id}/restore")
    async def restore_artwork(artwork_id: str) -> dict[str, Any]:
        try:
            art = await asyncio.to_thread(gallery.restore, artwork_id)
        except KeyError:
            raise artwork_404()
        except (OSError, sqlite3.Error) as exc:
            raise save_failed(exc)
        gallery_changed()
        return {"ok": True, "artwork": art.admin()}

    @admin.post("/artworks/purge")
    async def purge_artworks() -> dict[str, Any]:
        try:
            purged = await asyncio.to_thread(gallery.purge_deleted)
        except (OSError, sqlite3.Error) as exc:
            raise save_failed(exc)
        log.info("emptied the gallery trash (%d artworks)", purged)
        return {"ok": True, "purged": purged}

    app.include_router(admin)
    return app
