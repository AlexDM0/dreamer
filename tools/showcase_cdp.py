"""A small Chrome DevTools Protocol client for driving headless Microsoft Edge.

Used by ``tools/make_showcase.py``. Only needs the ``websockets`` package (already in
.venv); no Playwright, no downloads.

    edge = Edge.launch()                       # headless Edge with a temp profile
    cdp = await Cdp.connect(edge.ws_url)
    page = await Page.create(cdp, IPAD_LANDSCAPE)
    await page.navigate("http://127.0.0.1:7880/")
    await page.tap_selector("#save")
    await page.screenshot()                     # PNG bytes
    await cdp.close(); edge.stop()
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


class CdpError(RuntimeError):
    """A CDP command failed, a wait timed out, or the browser went away."""


# ---------------------------------------------------------------------------- devices

@dataclass(frozen=True)
class Device:
    name: str
    width: int
    height: int
    scale: float = 2.0
    mobile: bool = True
    touch: bool = True

    @property
    def orientation(self) -> str:
        return "landscape" if self.width >= self.height else "portrait"


IPAD_LANDSCAPE = Device("ipad-landscape", 1180, 820)
IPAD_PORTRAIT = Device("ipad-portrait", 820, 1180)
PHONE = Device("phone", 390, 844)
DESKTOP = Device("desktop", 1280, 900, mobile=False, touch=False)


# ---------------------------------------------------------------------------- browser

class Edge:
    """A headless Edge process with its own temporary profile."""

    def __init__(self, proc: subprocess.Popen, profile: str, ws_url: str) -> None:
        self.proc = proc
        self.profile = profile
        self.ws_url = ws_url

    @classmethod
    def launch(cls, exe: str = EDGE_PATH, timeout: float = 30.0,
               extra_args: list[str] | None = None) -> Edge:
        if not Path(exe).is_file():
            raise CdpError(f"Microsoft Edge not found at {exe}")
        profile = tempfile.mkdtemp(prefix="dreamer-edge-")
        args = [
            exe, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}",
            "--hide-scrollbars", "--force-device-scale-factor=2", "--no-first-run",
            "--no-default-browser-check", "--disable-extensions", "--disable-sync",
            "--disable-features=msEdgeSidebarV2,EdgeCollections,msUndersideButton",
            "--mute-audio", *(extra_args or []), "about:blank",
        ]
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = Path(profile) / "DevToolsActivePort"
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if proc.poll() is not None:
                shutil.rmtree(profile, ignore_errors=True)
                raise CdpError(f"Edge exited at start-up with code {proc.returncode}")
            try:
                lines = port_file.read_text().splitlines()
                if len(lines) >= 2:
                    return cls(proc, profile, f"ws://127.0.0.1:{lines[0]}{lines[1]}")
            except OSError:
                pass
            time.sleep(0.1)
        edge = cls(proc, profile, "")
        edge.stop()
        raise CdpError("Edge did not open its DevTools port in time")

    def stop(self) -> None:
        """Kill Edge and every process using its profile, then delete the temp profile."""
        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=3)  # Browser.close may already be in progress
            except subprocess.TimeoutExpired:
                pass
        if self.proc.poll() is None:
            self.proc.kill()
        # Utility processes can outlive the browser process; find them by the profile path.
        profile = Path(self.profile).name
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
             f"Where-Object {{ $_.CommandLine -like '*{profile}*' }} | "
             "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        for _attempt in range(40):  # file handles are released a moment after the kill
            shutil.rmtree(self.profile, ignore_errors=True)
            if not os.path.exists(self.profile):
                return
            time.sleep(0.25)
        print(f"warning: could not delete the Edge profile {self.profile}", flush=True)


# ---------------------------------------------------------------------------- protocol

class Cdp:
    """One browser-level WebSocket; pages are flattened sessions on it."""

    def __init__(self, socket: Any) -> None:
        self.socket = socket
        self.next_id = 0
        self.pending: dict[int, asyncio.Future] = {}
        self.waiters: list[tuple[str, str | None, Callable[[dict], bool], asyncio.Future]] = []
        self.listeners: list[Callable[[str, str | None, dict], None]] = []
        self.reader = asyncio.get_running_loop().create_task(self._read())
        self.closed = False

    @classmethod
    async def connect(cls, url: str) -> Cdp:
        socket = await websockets.connect(url, max_size=None, ping_interval=None,
                                          open_timeout=15)
        return cls(socket)

    async def _read(self) -> None:
        try:
            async for raw in self.socket:
                message = json.loads(raw)
                if "id" in message:
                    future = self.pending.pop(message["id"], None)
                    if future and not future.done():
                        future.set_result(message)
                    continue
                method = message.get("method", "")
                session = message.get("sessionId")
                params = message.get("params", {})
                for listener in list(self.listeners):
                    listener(method, session, params)
                for waiter in list(self.waiters):
                    wanted, wanted_session, predicate, future = waiter
                    if (method == wanted and (wanted_session is None or wanted_session == session)
                            and not future.done() and predicate(params)):
                        future.set_result(params)
                        self.waiters.remove(waiter)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.closed = True
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(CdpError("browser connection closed"))
            for _method, _session, _predicate, future in self.waiters:
                if not future.done():
                    future.set_exception(CdpError("browser connection closed"))

    async def send(self, method: str, params: dict | None = None, session: str | None = None,
                   timeout: float = 30.0) -> dict:
        if self.closed:
            raise CdpError(f"{method}: browser connection is closed")
        self.next_id += 1
        message: dict[str, Any] = {"id": self.next_id, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = session
        future = asyncio.get_running_loop().create_future()
        self.pending[self.next_id] = future
        await self.socket.send(json.dumps(message))
        try:
            reply = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self.pending.pop(message["id"], None)
            raise CdpError(f"{method}: no answer from the browser within {timeout:.0f} s")
        if "error" in reply:
            raise CdpError(f"{method}: {reply['error'].get('message')} {reply['error'].get('data', '')}")
        return reply.get("result", {})

    def expect(self, method: str, session: str | None = None,
               predicate: Callable[[dict], bool] = lambda _params: True) -> asyncio.Future:
        """Register a wait for an event *before* triggering it."""
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((method, session, predicate, future))
        return future

    async def close(self) -> None:
        try:
            await self.send("Browser.close", timeout=5)
        except CdpError:
            pass
        try:
            await self.socket.close()
        except Exception:
            pass
        self.reader.cancel()


# ---------------------------------------------------------------------------- pages

class Page:
    """A page (target) in its own browser context, emulating a device."""

    def __init__(self, cdp: Cdp, target_id: str, session: str, context_id: str) -> None:
        self.cdp = cdp
        self.target_id = target_id
        self.session = session
        self.context_id = context_id
        self.device: Device | None = None
        self.errors: list[str] = []
        cdp.listeners.append(self._on_event)

    @classmethod
    async def create(cls, cdp: Cdp, device: Device, context_id: str | None = None) -> Page:
        if context_id is None:  # a fresh context: its own cookies and localStorage
            context_id = (await cdp.send("Target.createBrowserContext",
                                         {"disposeOnDetach": True}))["browserContextId"]
        target_id = (await cdp.send("Target.createTarget", {
            "url": "about:blank", "browserContextId": context_id, "newWindow": True,
        }))["targetId"]
        session = (await cdp.send("Target.attachToTarget",
                                  {"targetId": target_id, "flatten": True}))["sessionId"]
        page = cls(cdp, target_id, session, context_id)
        for domain in ("Page", "Runtime", "Log"):
            await page.send(f"{domain}.enable")
        await page.send("Page.setLifecycleEventsEnabled", {"enabled": True})
        await page.set_device(device)
        return page

    def _on_event(self, method: str, session: str | None, params: dict) -> None:
        if session != self.session:
            return
        if method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            text = details.get("exception", {}).get("description") or details.get("text", "")
            self.errors.append(f"exception: {text.splitlines()[0] if text else '?'}")
        elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
            args = " ".join(str(arg.get("value", arg.get("description", ""))) for arg in params.get("args", []))
            self.errors.append(f"console.error: {args}")
        elif method == "Log.entryAdded" and params.get("entry", {}).get("level") == "error":
            entry = params["entry"]
            self.errors.append(f"log: {entry.get('text', '')} {entry.get('url', '')}".strip())

    async def send(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        return await self.cdp.send(method, params, self.session, timeout)

    # ---- emulation

    async def set_device(self, device: Device) -> None:
        self.device = device
        await self.send("Emulation.setDeviceMetricsOverride", {
            "width": device.width, "height": device.height, "deviceScaleFactor": device.scale,
            "mobile": device.mobile,
            "screenWidth": device.width, "screenHeight": device.height,
            "screenOrientation": {
                "type": "landscapePrimary" if device.orientation == "landscape" else "portraitPrimary",
                "angle": 90 if device.orientation == "landscape" else 0,
            },
        })
        await self.send("Emulation.setTouchEmulationEnabled",
                        {"enabled": device.touch, "maxTouchPoints": 5 if device.touch else 1})

    async def set_color_scheme(self, scheme: str) -> None:
        await self.send("Emulation.setEmulatedMedia",
                        {"features": [{"name": "prefers-color-scheme", "value": scheme}]})

    # ---- navigation and script

    async def navigate(self, url: str, timeout: float = 30.0) -> None:
        loaded = self.cdp.expect("Page.loadEventFired", self.session)
        result = await self.send("Page.navigate", {"url": url})
        if result.get("errorText"):
            loaded.cancel()
            raise CdpError(f"could not open {url}: {result['errorText']}")
        try:
            await asyncio.wait_for(loaded, timeout)
        except asyncio.TimeoutError:
            raise CdpError(f"{url} did not finish loading within {timeout:.0f} s")

    async def eval(self, expression: str, timeout: float = 30.0) -> Any:
        result = await self.send("Runtime.evaluate", {
            "expression": expression, "awaitPromise": True, "returnByValue": True,
        }, timeout=timeout)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            text = details.get("exception", {}).get("description") or details.get("text")
            raise CdpError(f"script failed: {text}\n  in: {expression[:200]}")
        return result.get("result", {}).get("value")

    async def wait_for(self, expression: str, what: str, timeout: float = 15.0,
                       interval: float = 0.1) -> Any:
        """Poll a JS expression until it is truthy; a clear error names `what` on timeout."""
        end = time.monotonic() + timeout
        last_error = ""
        while True:
            try:
                value = await self.eval(expression)
                if value:
                    return value
            except CdpError as exc:
                last_error = str(exc)
            if time.monotonic() > end:
                hint = f" (last error: {last_error})" if last_error else ""
                raise CdpError(f"timed out after {timeout:.0f} s waiting for {what}{hint}")
            await asyncio.sleep(interval)

    async def rect(self, selector: str) -> dict:
        found = await self.eval(
            f"(() => {{ const el = document.querySelector({json.dumps(selector)});"
            " if (!el) return null; const r = el.getBoundingClientRect();"
            " return {x: r.left, y: r.top, width: r.width, height: r.height}; })()")
        if not found:
            raise CdpError(f"element {selector} not found")
        return found

    async def scroll_into_view(self, selector: str, block: str = "center",
                               inline: str = "center", offset: float = 0) -> None:
        await self.eval(
            f"(() => {{ const el = document.querySelector({json.dumps(selector)});"
            f" el.scrollIntoView({{block: {json.dumps(block)}, inline: {json.dumps(inline)}, behavior: 'instant'}});"
            f" if ({offset}) window.scrollBy(0, {offset}); }})()")
        await asyncio.sleep(0.15)

    # ---- input (real touch / mouse events, so the page's own handlers run)

    async def touch(self, kind: str, x: float | None = None, y: float | None = None) -> None:
        points = [] if kind == "touchEnd" else [
            {"x": x, "y": y, "radiusX": 6, "radiusY": 6, "force": 1, "id": 1}]
        await self.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": points})

    async def tap(self, x: float, y: float) -> None:
        if self.device and self.device.touch:
            await self.touch("touchStart", x, y)
            await asyncio.sleep(0.05)
            await self.touch("touchEnd")
        else:
            for kind in ("mousePressed", "mouseReleased"):
                await self.send("Input.dispatchMouseEvent",
                                {"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1})

    async def mouse_click(self, x: float, y: float) -> None:
        await self.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        await self.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y,
                                                     "button": "left", "buttons": 1, "clickCount": 1})
        await self.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y,
                                                     "button": "left", "buttons": 0, "clickCount": 1})
        await asyncio.sleep(0.05)

    async def tap_selector(self, selector: str, scroll: bool = True) -> None:
        if scroll:
            await self.eval(
                f"(() => {{ const el = document.querySelector({json.dumps(selector)});"
                " if (!el) return; const r = el.getBoundingClientRect();"
                " if (r.top < 0 || r.left < 0 || r.bottom > innerHeight || r.right > innerWidth)"
                "   el.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'}); })()")
        box = await self.rect(selector)
        if box["width"] <= 0 or box["height"] <= 0:
            raise CdpError(f"element {selector} is not visible")
        await self.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await asyncio.sleep(0.12)

    async def drag(self, points: list[tuple[float, float]], step_delay: float = 0.005) -> None:
        """One finger stroke through `points` (CSS px, already densified)."""
        if not points:
            return
        await self.touch("touchStart", *points[0])
        for point in points[1:]:
            await self.touch("touchMove", *point)
            if step_delay:
                await asyncio.sleep(step_delay)
        await self.touch("touchEnd")

    # ---- capture

    async def screenshot(self, clip: dict | None = None) -> bytes:
        params: dict[str, Any] = {"format": "png", "fromSurface": True}
        if clip:
            params["clip"] = {**clip, "scale": 1}
            params["captureBeyondViewport"] = True
        result = await self.send("Page.captureScreenshot", params, timeout=60)
        return base64.b64decode(result["data"])

    async def close(self) -> None:
        try:
            await self.cdp.send("Target.closeTarget", {"targetId": self.target_id}, timeout=5)
        except CdpError:
            pass


def densify(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    """Insert points so neighbours are at most `spacing` apart (a finger moves smoothly)."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing))
        for i in range(1, steps + 1):
            out.append((x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps))
    return out
