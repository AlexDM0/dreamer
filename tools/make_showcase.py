"""Regenerate every marketing / documentation screenshot of Dreamer in one go.

    .venv\\Scripts\\python.exe tools\\make_showcase.py              everything
    .venv\\Scripts\\python.exe tools\\make_showcase.py --only kid   a subset (kid, admin, help, examples)

What it does:
  1. Starts Dreamer (``python -m server``, the real GPU backend) with a temporary config
     derived from config.yaml: test PIN, a free port, a fresh temporary data folder. The
     repo's data/ folder is never touched; the temp folder is deleted at the end.
  2. Drives headless Microsoft Edge over the Chrome DevTools Protocol (tools/showcase_cdp.py)
     with real touch input: it picks paper, topic, style, colours and brush like a child,
     draws kid-like strokes, waits for the refined picture and saves each artwork.
  3. Captures the kid page (iPad landscape/portrait, dark/light), the admin page on a phone
     (dark/light), the help page, and the raw before/after example pairs.
  4. Writes screenshots/manifest.json describing every file.

Groups (--only, comma separated or repeated):
  kid       kid page shots (drawing, pictures, split view, paper picker, minimised toolbar,
            light theme, gallery, detail, keep drawing, portrait)
  admin     admin page on a phone, dark and light
  help      the picture-controls help page
  examples  screenshots/examples/<name>-sketch.png / -picture.jpg
The artworks are drawn whenever kid, admin or examples is selected (they need them).
"""
from __future__ import annotations

import argparse
import asyncio
import http.cookiejar
import io
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from showcase_cdp import (
    DESKTOP,
    IPAD_LANDSCAPE,
    IPAD_PORTRAIT,
    PHONE,
    Cdp,
    CdpError,
    Edge,
    Page,
    densify,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"
EXAMPLES = OUT / "examples"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TEST_PIN = "4321"            # the temporary server's PIN; config.yaml's PIN is never read
GROUPS = ("kid", "admin", "help", "examples")
MAX_PNG_BYTES = 1_000_000    # bigger shots are stored as high-quality JPEG
JPEG_QUALITY = 90

# Pen colours (toolbar.js SWATCHES) and paper colours (paper.js PAPERS).
BLACK, WHITE, RED, ORANGE, YELLOW = "#22223b", "#ffffff", "#ff4d5e", "#ff9f1c", "#ffd23f"
GREEN, BLUE, PURPLE, PINK = "#3ecf6e", "#2fa8ff", "#8b5cf6", "#ff7eb6"
PAPER_NAMES = {"#000000": "black", "#ffffff": "white", "#1e2a5a": "night blue",
               "#8fd3ff": "sky blue", "#4caf50": "grass green", "#ffd23f": "sunny yellow",
               "#ff9ec7": "pink", "#c9b6f2": "lilac"}


class ShowcaseError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ============================================================================ drawings
# Strokes are in drawing pixels (640x448 landscape, 448x640 portrait) and get a little
# wobble before they are drawn, like a child's finger.

@dataclass
class Stroke:
    color: str
    width: int
    points: list[tuple[float, float]]


@dataclass
class Artwork:
    name: str
    title: str
    topic: str
    style: str
    paper: str
    orientation: str
    final_modes: list[str]       # view modes captured after refinement
    strokes: list[Stroke] = field(default_factory=list)
    live_after: int = 0          # capture a "live" split shot after this many strokes (0: no)
    artwork_id: str = ""


def circle(cx: float, cy: float, rx: float, ry: float | None = None, turns: float = 1.08,
           start: float = -1.9) -> list[tuple[float, float]]:
    ry = rx if ry is None else ry
    count = max(16, int(2 * math.pi * max(rx, ry) * turns / 8))
    return [(cx + rx * math.cos(start + 2 * math.pi * turns * i / count),
             cy + ry * math.sin(start + 2 * math.pi * turns * i / count)) for i in range(count + 1)]


def spiral(cx: float, cy: float, radius: float, turns: float = 2.5) -> list[tuple[float, float]]:
    """A colouring-in scribble that fills a round shape."""
    count = int(turns * 24)
    return [(cx + radius * (1 - i / count) * math.cos(i / 24 * 2 * math.pi),
             cy + radius * (1 - i / count) * math.sin(i / 24 * 2 * math.pi)) for i in range(count + 1)]


def zigzag(x0: float, x1: float, y_top: float, y_bottom: float, teeth: int) -> list[tuple[float, float]]:
    """Up-down scribble from x0 to x1 (grass, flames, colouring-in)."""
    return [(x0 + (x1 - x0) * i / teeth, y_top if i % 2 else y_bottom) for i in range(teeth + 1)]


def fill_rows(x0: float, x1: float, y0: float, y1: float, rows: int) -> list[tuple[float, float]]:
    """Back-and-forth colouring-in of a rectangle."""
    points = []
    for row in range(rows + 1):
        y = y0 + (y1 - y0) * row / rows
        points += [(x0, y), (x1, y)] if row % 2 == 0 else [(x1, y), (x0, y)]
    return points


def rays(cx: float, cy: float, inner: float, outer: float, count: int) -> list[list[tuple[float, float]]]:
    return [[(cx + inner * math.cos(angle), cy + inner * math.sin(angle)),
             (cx + outer * math.cos(angle), cy + outer * math.sin(angle))]
            for angle in (2 * math.pi * i / count + 0.2 for i in range(count))]


def flower(cx: float, cy: float, petal_color: str) -> list[Stroke]:
    petals = [Stroke(petal_color, 12, circle(cx + 22 * math.cos(angle), cy + 22 * math.sin(angle), 14))
              for angle in (2 * math.pi * i / 5 for i in range(5))]
    return petals + [Stroke(ORANGE, 14, spiral(cx, cy, 10, 1.5))]


def house() -> Artwork:
    art = Artwork("house", "Sunny house", "nature", "storybook", "#000000", "landscape", ["auto"])
    art.strokes = [
        Stroke(WHITE, 16, [(170, 392), (172, 240), (332, 238), (330, 392), (170, 392)]),
        Stroke(RED, 16, [(150, 250), (252, 148), (352, 248), (150, 250)]),
        Stroke(RED, 14, fill_rows(200, 300, 215, 238, 2)),
        Stroke(ORANGE, 14, [(228, 390), (228, 310), (272, 308), (272, 390)]),
        Stroke(BLUE, 10, [(190, 265), (230, 265), (230, 300), (190, 300), (190, 265)]),
        Stroke(BLUE, 8, [(210, 265), (210, 300)]),
        Stroke(YELLOW, 16, circle(535, 88, 40)),
        Stroke(YELLOW, 14, spiral(535, 88, 30)),
        *[Stroke(YELLOW, 10, ray) for ray in rays(535, 88, 55, 82, 9)],
        Stroke(ORANGE, 18, [(452, 395), (450, 300)]),
        Stroke(GREEN, 16, circle(452, 255, 52)),
        Stroke(GREEN, 14, spiral(452, 255, 40)),
        Stroke(GREEN, 14, zigzag(20, 620, 395, 430, 34)),
        Stroke(WHITE, 12, circle(110, 80, 34, 18)),
        Stroke(WHITE, 12, circle(150, 72, 30, 18)),
    ]
    return art


def cat() -> Artwork:
    art = Artwork("cat", "Swimming cat", "ocean", "cartoon", "#1e2a5a", "landscape", ["split"])
    art.strokes = [
        Stroke(ORANGE, 16, circle(300, 195, 70)),
        Stroke(ORANGE, 14, [(248, 158), (250, 92), (292, 132)]),
        Stroke(ORANGE, 14, [(308, 132), (352, 92), (354, 158)]),
        Stroke(GREEN, 12, circle(274, 186, 9)),
        Stroke(GREEN, 12, circle(326, 186, 9)),
        Stroke(PINK, 12, [(292, 210), (308, 210), (300, 222), (292, 210)]),
        Stroke(WHITE, 6, [(230, 214), (282, 220)]), Stroke(WHITE, 6, [(232, 232), (282, 226)]),
        Stroke(WHITE, 6, [(318, 220), (370, 214)]), Stroke(WHITE, 6, [(318, 226), (368, 232)]),
        Stroke(PINK, 8, [(284, 240), (300, 250), (316, 240)]),
        Stroke(ORANGE, 16, circle(300, 345, 95, 65)),
        Stroke(ORANGE, 14, [(392, 360), (440, 330), (470, 280), (460, 240), (440, 250)]),
        Stroke(YELLOW, 14, circle(530, 140, 42, 24)),
        Stroke(YELLOW, 14, [(488, 140), (462, 118), (464, 162), (488, 140)]),
        Stroke(BLUE, 10, circle(546, 134, 5)),
        Stroke(WHITE, 8, circle(470, 70, 10)), Stroke(WHITE, 8, circle(500, 40, 7)),
        Stroke(WHITE, 8, circle(160, 90, 9)),
        Stroke(GREEN, 14, [(70, 440), (90, 390), (60, 340), (88, 290), (64, 240), (84, 200)]),
        Stroke(GREEN, 14, [(120, 440), (136, 400), (112, 360), (130, 320)]),
        Stroke(PINK, 12, zigzag(500, 620, 400, 440, 8)),
    ]
    return art


def rocket() -> Artwork:
    art = Artwork("rocket", "Space rocket", "space", "watercolor", "#000000", "landscape",
                  ["split"], live_after=6)
    art.strokes = [
        Stroke(WHITE, 16, [(290, 340), (290, 150), (322, 70), (354, 150), (354, 340), (290, 340)]),
        Stroke(BLUE, 12, circle(322, 190, 18)),
        Stroke(RED, 14, [(290, 280), (250, 352), (290, 340)]),
        Stroke(RED, 14, [(354, 280), (394, 352), (354, 340)]),
        Stroke(ORANGE, 16, zigzag(298, 346, 350, 410, 6)),
        Stroke(YELLOW, 12, zigzag(306, 338, 352, 390, 4)),
        Stroke(PURPLE, 16, circle(130, 140, 55)),
        Stroke(PURPLE, 14, spiral(130, 140, 42)),
        Stroke(YELLOW, 10, circle(130, 140, 98, 22, turns=1.0, start=0.1)),
        Stroke(BLUE, 16, circle(520, 320, 62)),
        Stroke(GREEN, 14, spiral(505, 305, 22, 1.5)),
        Stroke(GREEN, 14, spiral(545, 345, 16, 1.5)),
        *[Stroke(YELLOW, 8, [(x - 9, y), (x + 9, y)]) for x, y in ((470, 80), (580, 150), (210, 300), (80, 380), (420, 230))],
        *[Stroke(YELLOW, 8, [(x, y - 9), (x, y + 9)]) for x, y in ((470, 80), (580, 150), (210, 300), (80, 380), (420, 230))],
        Stroke(PINK, 12, circle(590, 60, 22, turns=0.6, start=1.2)),
    ]
    return art


def dinosaur() -> Artwork:
    art = Artwork("dinosaur", "Dino friend", "dinosaurs", "crayon", "#ffffff", "landscape", ["auto"])
    art.strokes = [
        Stroke(GREEN, 16, circle(290, 290, 130, 68)),
        Stroke(GREEN, 16, [(392, 252), (430, 200), (452, 150), (470, 120)]),
        Stroke(GREEN, 16, [(410, 290), (446, 220), (480, 140)]),
        Stroke(GREEN, 16, circle(500, 112, 38, 24)),
        Stroke(BLACK, 10, circle(510, 104, 5)),
        Stroke(GREEN, 16, [(165, 280), (110, 300), (50, 340), (110, 330), (170, 310)]),
        Stroke(GREEN, 16, [(215, 340), (212, 410), (240, 410), (245, 350)]),
        Stroke(GREEN, 16, [(335, 345), (338, 410), (366, 410), (365, 340)]),
        Stroke(GREEN, 12, fill_rows(200, 380, 250, 320, 5)),
        Stroke(PURPLE, 12, circle(250, 270, 12)), Stroke(PURPLE, 12, circle(300, 250, 10)),
        Stroke(PURPLE, 12, circle(340, 280, 12)),
        Stroke(BLACK, 8, [(488, 126), (500, 130), (512, 126)]),
        Stroke(YELLOW, 16, circle(90, 80, 40)),
        Stroke(YELLOW, 14, spiral(90, 80, 30)),
        Stroke(ORANGE, 14, [(560, 410), (600, 300), (636, 410)]),
        Stroke(RED, 12, [(590, 300), (600, 270), (612, 300)]),
        Stroke(GREEN, 12, zigzag(10, 560, 415, 440, 30)),
    ]
    return art


def castle() -> Artwork:
    art = Artwork("castle", "Fairy-tale castle", "castle", "clay", "#c9b6f2", "landscape", ["split"])
    battlements = [(210, 225)]
    for i in range(6):
        x = 210 + i * 38
        battlements += [(x, 200), (x + 19, 200), (x + 19, 225), (x + 38, 225)]
    art.strokes = [
        Stroke(PURPLE, 16, [(210, 400), (210, 225)]),
        Stroke(PURPLE, 16, battlements),
        Stroke(PURPLE, 16, [(438, 225), (438, 400)]),
        Stroke(PINK, 16, [(140, 400), (140, 160), (210, 160), (210, 400)]),
        Stroke(PINK, 16, [(438, 400), (438, 160), (508, 160), (508, 400)]),
        Stroke(RED, 14, [(132, 165), (175, 82), (218, 165)]),
        Stroke(RED, 14, [(430, 165), (473, 82), (516, 165)]),
        Stroke(BLACK, 8, [(175, 82), (175, 40)]), Stroke(YELLOW, 12, [(175, 42), (205, 52), (175, 62)]),
        Stroke(BLACK, 8, [(473, 82), (473, 40)]), Stroke(YELLOW, 12, [(473, 42), (503, 52), (473, 62)]),
        Stroke(ORANGE, 16, [(292, 400), (292, 345), (300, 325), (324, 315), (348, 325), (356, 345), (356, 400)]),
        Stroke(BLUE, 12, circle(175, 230, 14)), Stroke(BLUE, 12, circle(473, 230, 14)),
        Stroke(BLUE, 12, [(258, 270), (258, 300), (284, 300), (284, 270), (258, 270)]),
        Stroke(BLUE, 12, [(364, 270), (364, 300), (390, 300), (390, 270), (364, 270)]),
        Stroke(GREEN, 16, zigzag(10, 630, 405, 440, 36)),
        Stroke(WHITE, 12, circle(560, 70, 40, 18)), Stroke(WHITE, 12, circle(80, 60, 34, 16)),
        Stroke(YELLOW, 14, circle(320, 120, 22)),
    ]
    return art


def meadow() -> Artwork:
    art = Artwork("meadow", "Flower meadow", "nature", "watercolor", "#8fd3ff", "portrait",
                  ["auto", "split"])
    art.strokes = [
        Stroke(GREEN, 18, zigzag(10, 440, 440, 470, 22)),
        Stroke(GREEN, 16, zigzag(10, 440, 500, 530, 22)),
        Stroke(GREEN, 16, zigzag(10, 440, 560, 600, 22)),
        Stroke(GREEN, 14, zigzag(10, 440, 610, 635, 22)),
        *[Stroke(GREEN, 10, [(x, 560), (x + 6, 470), (x, 380)]) for x in (80, 190, 300, 390)],
        *flower(80, 360, RED), *flower(190, 330, PINK), *flower(300, 370, PURPLE), *flower(390, 340, YELLOW),
        Stroke(YELLOW, 16, circle(360, 95, 45)),
        Stroke(YELLOW, 14, spiral(360, 95, 34)),
        *[Stroke(YELLOW, 10, ray) for ray in rays(360, 95, 60, 85, 8)],
        Stroke(WHITE, 12, circle(110, 90, 44, 22)), Stroke(WHITE, 12, circle(160, 80, 36, 20)),
        Stroke(PURPLE, 10, circle(150, 220, 18, 12, start=0)), Stroke(PURPLE, 10, circle(186, 220, 18, 12, start=3)),
        Stroke(BLACK, 6, [(168, 206), (168, 236)]),
    ]
    return art


ARTWORKS = (house, cat, rocket, dinosaur, castle, meadow)


def wobble(points: list[tuple[float, float]], rng: random.Random) -> list[tuple[float, float]]:
    """A child's hand: corners land a bit off, lines waver slowly, then a fine tremor."""
    jittered = [(x + rng.uniform(-3, 3), y + rng.uniform(-3, 3)) for x, y in points]
    dense = densify(jittered, 6.0)
    phase_x, phase_y = rng.uniform(0, 6.3), rng.uniform(0, 6.3)
    return [(x + 1.8 * math.sin(i / 9 + phase_x) + rng.uniform(-0.5, 0.5),
             y + 1.8 * math.sin(i / 11 + phase_y) + rng.uniform(-0.5, 0.5))
            for i, (x, y) in enumerate(dense)]


# ============================================================================ server

def free_port(start: int = 7880) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise ShowcaseError("no free port found")


def temp_config(port: int, data_dir: Path, path: Path, mock: bool = False) -> None:
    """config.yaml with the test PIN, port, data_dir and host replaced. Lines are dropped by
    key only; the real PIN's value is never looked at."""
    source = ROOT / "config.yaml"
    if not source.is_file():
        source = ROOT / "config.example.yaml"
    replaced = ("admin_pin", "port", "data_dir", "host") + (("backend", "mock_delay_ms") if mock else ())
    kept = [line for line in source.read_text(encoding="utf-8").splitlines()
            if not re.match(rf"^({'|'.join(replaced)})\s*:", line)]
    kept += ["", "# showcase overrides", f'admin_pin: "{TEST_PIN}"', f"port: {port}",
             "host: 127.0.0.1", f"data_dir: {json.dumps(str(data_dir))}"]
    if mock:
        kept += ["backend: mock", "mock_delay_ms: 30"]
    kept.append("")
    path.write_text("\n".join(kept), encoding="utf-8")


class Server:
    def __init__(self, mock: bool = False) -> None:
        self.mock = mock
        self.temp = Path(tempfile.mkdtemp(prefix="dreamer-showcase-"))
        self.data_dir = self.temp / "data"
        self.config = self.temp / "config.yaml"
        self.log_path = Path(tempfile.gettempdir()) / f"dreamer-showcase-server-{os.getpid()}.log"
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.log_file: Any = None
        self.cookies = http.cookiejar.CookieJar()
        self.http = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookies))

    def start(self, timeout: float = 300.0) -> None:
        if not PYTHON.is_file():
            raise ShowcaseError(f"{PYTHON} not found; run setup.ps1 first")
        temp_config(self.port, self.data_dir, self.config, self.mock)
        self.log_file = open(self.log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115 (closed in stop())
        env = {**os.environ, "SKETCH_CONFIG": str(self.config), "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen([str(PYTHON), "-m", "server"], cwd=ROOT, env=env,
                                     stdout=self.log_file, stderr=subprocess.STDOUT)
        log(f"server starting on {self.base} (log: {self.log_path})")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                raise ShowcaseError(f"server exited with code {self.proc.returncode} during start-up")
            try:
                health = self.get("/healthz", timeout=2)
                if health.get("state") == "ready":
                    log(f"server ready (backend {health.get('backend')})")
                    return
                if health.get("state") == "error":
                    raise ShowcaseError("server reports backend state 'error'")
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(0.5)
        raise ShowcaseError(f"server not ready after {timeout:.0f} s")

    def request(self, method: str, path: str, body: Any = None, timeout: float = 15.0) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with self.http.open(req, timeout=timeout) as res:
                raw = res.read()
                kind = res.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raise ShowcaseError(f"{method} {path} -> HTTP {exc.code}: {exc.read()[:300]!r}") from None
        return json.loads(raw) if "json" in kind else raw

    def get(self, path: str, timeout: float = 15.0) -> Any:
        return self.request("GET", path, timeout=timeout)

    def login(self) -> None:
        self.request("POST", "/api/admin/login", {"pin": TEST_PIN})

    def tail(self, lines: int = 60) -> str:
        try:
            if self.log_file:
                self.log_file.flush()
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no server log)"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            # .venv's python.exe is a launcher with the real interpreter as its child:
            # kill the whole tree.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self.log_file:
            self.log_file.close()
        for _attempt in range(20):  # SQLite / image files can stay locked for a moment
            shutil.rmtree(self.temp, ignore_errors=True)
            if not self.temp.exists():
                break
            time.sleep(0.25)
        if self.temp.exists():
            log(f"warning: could not delete {self.temp}")

    def remove_log(self) -> None:
        try:
            self.log_path.unlink()
        except OSError:
            pass


# ============================================================================ output

class Output:
    """Saves screenshots (PNG, or JPEG when large) and collects the manifest."""

    def __init__(self, groups: set[str]) -> None:
        self.groups = groups
        self.entries: list[dict[str, Any]] = []
        self.written: set[str] = set()
        OUT.mkdir(parents=True, exist_ok=True)
        EXAMPLES.mkdir(parents=True, exist_ok=True)

    def save(self, group: str, stem: str, png: bytes, *, title: str, caption: str, device: str,
             theme: str, features: list[str], **extra: Any) -> None:
        if group not in self.groups:
            return
        image = Image.open(io.BytesIO(png))
        # Keep an existing shot's format so links in the docs survive a re-run; the size rule
        # only picks the format of a new shot.
        previous = [old.suffix for old in OUT.glob(f"{stem}.*")]
        for old in OUT.glob(f"{stem}.*"):
            old.unlink()
        if previous[:1] == [".jpg"] or (not previous and len(png) > MAX_PNG_BYTES):
            path = OUT / f"{stem}.jpg"
            image.convert("RGB").save(path, "JPEG", quality=JPEG_QUALITY, optimize=True,
                                      progressive=True, subsampling=0)
        else:
            path = OUT / f"{stem}.png"
            path.write_bytes(png)
        self._add(group, path, image.size, title=title, caption=caption, device=device,
                  theme=theme, features=features, **extra)

    def save_example(self, path: Path, data: bytes, size: tuple[int, int], **meta: Any) -> None:
        path.write_bytes(data)
        self._add("examples", path, size, **meta)

    def _add(self, group: str, path: Path, size: tuple[int, int], **meta: Any) -> None:
        relative = path.relative_to(OUT).as_posix()
        self.written.add(relative)
        self.entries.append({"file": relative, "group": group, **meta,
                             "width": size[0], "height": size[1], "bytes": path.stat().st_size})
        log(f"  saved {relative} ({size[0]}x{size[1]}, {path.stat().st_size // 1024} KB)")

    def finish(self, full_run: bool) -> None:
        manifest_path = OUT / "manifest.json"
        previous: list[dict[str, Any]] = []
        if not full_run and manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
            except (ValueError, OSError):
                previous = []
        kept = [entry for entry in previous
                if entry.get("file") not in self.written and entry.get("group") not in self.groups
                and (OUT / entry.get("file", "")).is_file()]
        if full_run:  # drop shots that no longer exist in this script
            for path in [*OUT.glob("*.png"), *OUT.glob("*.jpg"), *EXAMPLES.glob("*")]:
                if path.relative_to(OUT).as_posix() not in self.written:
                    path.unlink()
        entries = sorted(kept + self.entries, key=lambda entry: entry["file"])
        manifest = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "generator": "tools/make_showcase.py",
            "note": "Real pictures made on the RTX 3080 PC (StreamDiffusion SD-Turbo + scribble "
                    "ControlNet). UI shots are 2x device pixel ratio.",
            "files": entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        total = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
        log(f"manifest: {len(entries)} files, folder total {total / 1e6:.1f} MB")


# ============================================================================ kid page

KID_CLEAN = """(() => {
  const hidden = (sel) => { const el = document.querySelector(sel); return !el || el.hidden; };
  return document.querySelector('#conn').dataset.state === 'ready'
    && hidden('#hint') && hidden('#confirm') && hidden('#replaced') && hidden('#pick-sheet') && hidden('#offline') && hidden('#sleep')
    && (%(picker)s || hidden('#bg-picker'))
    && (%(fx)s || !document.querySelector('.fly, .spark'))
    && document.querySelector('#topics .chooser[data-id]') !== null;
})()"""


class Kid:
    def __init__(self, page: Page, server: Server, out: Output) -> None:
        self.page = page
        self.server = server
        self.out = out
        self.rng = random.Random(7)

    async def open(self) -> None:
        await self.page.navigate(self.server.base + "/")
        await self.page.wait_for("document.querySelector('#conn').dataset.state === 'ready'",
                                 "the kid page to connect", timeout=30)
        await self.page.wait_for("document.querySelector('#topics .chooser[data-id]') !== null",
                                 "topic buttons on the kid page")

    async def clean(self, picker: bool = False, effects: bool = False, timeout: float = 15) -> None:
        script = KID_CLEAN % {"picker": "true" if picker else "false", "fx": "true" if effects else "false"}
        try:
            await self.page.wait_for(script, "a clean kid page (connected, no hint/dialog)", timeout=timeout)
        except CdpError:
            state = await self.page.eval(
                "({conn: document.querySelector('#conn').dataset.state, conntitle: document.querySelector('#conn').title,"
                " hint: document.querySelector('#hint').hidden ? null : document.querySelector('#hint-text').textContent})")
            raise ShowcaseError(f"kid page not in a clean state: {state}") from None

    async def shot(self, stem: str, title: str, caption: str, features: list[str], *,
                   picker: bool = False, effects: bool = False, settle: float = 0.4, **extra: Any) -> None:
        if "kid" not in self.out.groups:
            return
        await self.clean(picker=picker, effects=effects)
        await asyncio.sleep(settle)
        theme = await self.page.eval("document.documentElement.dataset.theme || 'dark'")
        png = await self.page.screenshot()
        device = self.page.device
        self.out.save("kid", stem, png, title=title, caption=caption, device=device.name,
                      orientation=device.orientation, theme=theme, features=features, **extra)

    async def mode(self, mode: str) -> None:
        await self.page.tap_selector(f'#modes [data-mode="{mode}"]')
        await self.page.wait_for(f"document.querySelector('#stage').classList.contains('mode-{mode}')",
                                 f"view mode {mode}")
        await asyncio.sleep(0.35)

    async def pick(self, kind: str, item_id: str) -> None:
        # Picking the current item again still re-renders (used for a fresh live picture).
        current = f'#{kind} .chooser[data-id="{item_id}"]'
        await self.page.tap_selector(f"#{kind} .chooser")
        await self.page.wait_for("!document.querySelector('#pick-sheet').hidden", f"the {kind} sheet")
        tile = f'#pick-grid .pick-tile[data-id="{item_id}"]'
        await self.page.eval(f"document.querySelector('{tile}').scrollIntoView({{block: 'nearest', behavior: 'instant'}})")
        await asyncio.sleep(0.2)
        await self.page.tap_selector(tile, scroll=False)
        await self.page.wait_for(f"document.querySelector('{current}') !== null", f"{kind} {item_id} selected")

    async def color(self, color: str) -> None:
        if await self.page.eval(f"window.dreamer.toolbar.color === '{color}' && !window.dreamer.toolbar.eraser"):
            return
        await self.page.tap_selector(f'#swatches .swatch[data-color="{color}"]', scroll=False)
        await self.page.wait_for(f"window.dreamer.toolbar.color === '{color}'", f"pen colour {color}")

    async def brush(self, width: int) -> None:
        """Set the brush-size range (4..40) by pressing on its track. Chromium's emulated
        touch does not drag a range input, so this uses the mouse (it is a native control;
        the toolbar's own input handler runs either way)."""
        if abs(await self.page.eval("window.dreamer.toolbar.width") - width) <= 1:
            return
        box = await self.page.rect("#brush-size")
        y = box["y"] + box["height"] / 2
        fraction = (width - 4) / 36
        for _attempt in range(4):
            x = box["x"] + min(max(fraction, 0.0), 1.0) * box["width"]
            await self.page.mouse_click(x, y)
            got = await self.page.eval("window.dreamer.toolbar.width")
            if abs(got - width) <= 1:
                return
            fraction += (width - got) / 36  # correct for the thumb's inset
        raise ShowcaseError(f"could not set the brush size to {width} (it is {got})")

    async def new_drawing(self, paper: str, capture_picker: bool = False) -> None:
        await self.page.tap_selector("#new-drawing")
        if await self.page.eval("!document.querySelector('#confirm').hidden"):
            await self.page.tap_selector("#confirm-yes")
        await self.page.wait_for("!document.querySelector('#bg-picker').hidden", "the paper picker to open")
        await asyncio.sleep(0.5)
        if capture_picker:
            await self.shot(f"{self.page.device.name}-paper-picker", "Pick your paper",
                            "Every new drawing starts by picking a paper colour. Black paper gives "
                            "night scenes; the paper can be changed later and the strokes stay.",
                            ["paper picker", "new drawing"], picker=True)
        await self.page.tap_selector(f'#bg-grid [data-bg="{paper}"]', scroll=False)
        await self.page.wait_for(f"document.querySelector('#bg-picker').hidden && window.dreamer.canvas.background === '{paper}'",
                                 f"paper {paper}")

    async def draw(self, strokes: list[Stroke]) -> None:
        width, height = await self.page.eval(
            "[window.dreamer.canvas.drawingWidth, window.dreamer.canvas.drawingHeight]")
        box = await self.page.rect("#sketch")
        for stroke in strokes:
            await self.color(stroke.color)
            await self.brush(stroke.width)
            points = [(box["x"] + x / width * box["width"], box["y"] + y / height * box["height"])
                      for x, y in wobble(stroke.points, self.rng)]
            await self.page.drag(points)
            await asyncio.sleep(self.rng.uniform(0.08, 0.25))

    async def wait_refined(self, quiet: float = 1.5, minimum: float = 4.5, timeout: float = 40) -> None:
        """Wait until the refinement rounds (and the upscale) are done: the final picture is
        back and no new picture arrived for `quiet` seconds."""
        start = time.monotonic()
        last_count, last_change = -1, time.monotonic()
        while True:
            count, awaiting, state = await self.page.eval(
                "[Number(document.querySelector('#stage').dataset.pictures || 0),"
                " window.dreamer.net.awaitingFinal, document.querySelector('#conn').dataset.state]")
            now = time.monotonic()
            if count != last_count:
                last_count, last_change = count, now
            if (not awaiting and state == "ready" and now - last_change >= quiet
                    and now - start >= minimum and count > 0):
                return
            if now - start > timeout:
                raise ShowcaseError(f"picture did not settle within {timeout:.0f} s "
                                    f"(pictures={count}, awaiting={awaiting}, state={state})")
            await asyncio.sleep(0.2)

    async def save(self, art: Artwork, capture: bool = False) -> None:
        """Tap the star, wait for the confirmation, then name the new artwork (as a parent
        would on the admin page) so the gallery shows titles."""
        before = {item["id"] for item in self.server.get("/api/artworks?limit=100")["artworks"]}
        await self.page.tap_selector("#save")
        await self.page.wait_for("document.querySelector('#save').classList.contains('saved')",
                                 "the Save confirmation", timeout=15)
        if capture:
            await asyncio.sleep(0.3)
            await self.shot(f"{self.page.device.name}-{art.name}-saving", "Save: the picture flies into the gallery",
                            "The yellow star keeps the picture and its drawing; the picture flies "
                            "into the Gallery button with a little star burst.",
                            ["save", "gallery"], effects=True, settle=0, artwork=art.name)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            new = [item for item in self.server.get("/api/artworks?limit=100")["artworks"]
                   if item["id"] not in before]
            if new:
                art.artwork_id = new[0]["id"]
                self.server.request("PATCH", f"/api/admin/artworks/{art.artwork_id}", {"title": art.title})
                break
            await asyncio.sleep(0.2)
        else:
            raise ShowcaseError(f"saved artwork {art.name} did not show up in /api/artworks")
        await asyncio.sleep(1.8)  # let the fly-into-gallery animation finish


def feature_caption(art: Artwork) -> str:
    return (f"{art.topic.capitalize()} world, {art.style} style, on {PAPER_NAMES.get(art.paper, art.paper)} paper")


async def make_artworks(kid: Kid, out: Output) -> list[Artwork]:
    page = kid.page
    artworks = [factory() for factory in ARTWORKS]
    for index, art in enumerate(artworks):
        device = IPAD_PORTRAIT if art.orientation == "portrait" else IPAD_LANDSCAPE
        if page.device != device:
            await page.set_device(device)
            await asyncio.sleep(0.8)
        log(f"artwork {index + 1}/{len(artworks)}: {art.name} ({art.topic}/{art.style}, {art.orientation})")
        prefix = f"{device.name}-{art.name}"
        await kid.new_drawing(art.paper, capture_picker=index == 0)
        await page.wait_for(f"window.dreamer.canvas.orientation === '{art.orientation}'",
                            f"a {art.orientation} drawing")
        await kid.pick("topics", art.topic)
        await kid.pick("styles", art.style)
        if art.orientation == "portrait":  # the PC rebuilds its pipeline for the new shape
            await page.wait_for("document.querySelector('#conn').dataset.state === 'ready'",
                                "the engine to switch to portrait", timeout=60)
        if art.live_after:
            await kid.mode("split")
            await kid.draw(art.strokes[:art.live_after])
            await asyncio.sleep(0.6)
            await kid.shot(f"{prefix}-live-split", "The picture follows while drawing",
                           "Side by side: while your child draws, the picture keeps up within a "
                           "fraction of a second.", ["live picture", "split view", "drawing"],
                           artwork=art.name, settle=0)
            await kid.mode("sketch")
            await kid.draw(art.strokes[art.live_after:])
        else:
            await kid.mode("sketch")
            await kid.draw(art.strokes)
        await asyncio.sleep(0.3)
        await kid.shot(f"{prefix}-sketch", f"Drawing: {art.title.lower()}",
                       f"The child's drawing ({feature_caption(art)}), drawn with a finger.",
                       ["drawing", "sketch view"], artwork=art.name, settle=0.2)
        for mode in art.final_modes:
            await kid.mode(mode)
            await kid.wait_refined()
            if mode == "auto":
                await page.wait_for("document.querySelector('#stage').classList.contains('show-picture')",
                                    "the picture to fade in")
            size = await page.eval("[document.querySelector('#picture').width, document.querySelector('#picture').height]")
            log(f"  picture settled at {size[0]}x{size[1]}")
            suffix = "split" if mode == "split" else "picture"
            title = (f"Drawing and picture side by side: {art.title.lower()}" if mode == "split"
                     else f"Dream picture: {art.title.lower()}")
            caption = (f"When the drawing pauses, the picture refines and is upscaled. {feature_caption(art)}."
                       if mode != "split" else
                       f"The 'Both' view: the drawing and its finished picture side by side. {feature_caption(art)}.")
            await kid.shot(f"{prefix}-{suffix}", title, caption,
                           ["refined picture", "split view" if mode == "split" else "auto view"],
                           artwork=art.name, topic=art.topic, style=art.style, paper=art.paper)
        if art.name == "dinosaur":
            await page.tap_selector("#minimise")
            await page.wait_for("document.querySelector('.app').classList.contains('minimised')", "minimised toolbar")
            await asyncio.sleep(0.6)
            await kid.shot(f"{prefix}-minimised", "Minimised toolbar: a bigger drawing",
                           "One tap tucks the tools away; only the colours and the brush size stay, "
                           "and the drawing gets the room.", ["minimised toolbar"], artwork=art.name)
            await page.tap_selector("#expand")
            await page.wait_for("!document.querySelector('.app').classList.contains('minimised')", "expanded toolbar")
            await asyncio.sleep(0.5)
        await kid.save(art, capture=index == 0)
    return artworks


async def kid_extras(kid: Kid, artworks: list[Artwork]) -> None:
    """Gallery, detail, keep drawing, light theme (after all artworks are saved)."""
    page = kid.page
    if page.device != IPAD_LANDSCAPE:
        await page.set_device(IPAD_LANDSCAPE)
        await asyncio.sleep(0.8)
    images_loaded = ("[...document.querySelectorAll('{sel}')].length > 0 && [...document.querySelectorAll('{sel}')]"
                     ".every((img) => img.complete && img.naturalWidth > 0)")
    await page.tap_selector("#open-gallery")
    await page.wait_for(f"document.querySelectorAll('#gallery-grid .thumb').length >= {len(artworks)}",
                        "the gallery grid")
    await page.wait_for(images_loaded.format(sel="#gallery-grid img"), "gallery thumbnails")
    await asyncio.sleep(0.6)
    await kid.shot("ipad-landscape-gallery", "My pictures: the gallery",
                   "Every saved picture in a big grid. Children can look and keep drawing, but not delete.",
                   ["gallery"])
    cat_art = next(art for art in artworks if art.name == "cat")
    await page.tap_selector(f'#gallery-grid .thumb[data-id="{cat_art.artwork_id}"]')
    await page.wait_for("!document.querySelector('#detail').hidden", "the detail view")
    await page.wait_for(images_loaded.format(sel="#detail img"), "detail images")
    await asyncio.sleep(0.5)
    await kid.shot("ipad-landscape-gallery-detail", "A saved picture with its drawing",
                   "Tap a picture to see it next to the drawing it came from, then 'Keep drawing' "
                   "to carry on.", ["gallery", "keep drawing"], artwork="cat")

    # Light theme, switched by the parent (admin API), while the gallery is open.
    kid.server.request("POST", "/api/admin/select", {"kid_theme": "light"})
    await page.wait_for("document.documentElement.dataset.theme === 'light'", "the light kid theme")
    await asyncio.sleep(0.5)
    await kid.shot("ipad-landscape-gallery-detail-light", "Light look: a saved picture",
                   "The drawing page can also have a light look, chosen on the parent page.",
                   ["gallery", "light theme"], artwork="cat")
    await page.tap_selector("#detail-back")
    await page.wait_for(images_loaded.format(sel="#gallery-grid img"), "gallery thumbnails")
    await asyncio.sleep(0.4)
    await kid.shot("ipad-landscape-gallery-light", "Light look: the gallery",
                   "The gallery in the light look.", ["gallery", "light theme"])
    await page.tap_selector(f'#gallery-grid .thumb[data-id="{cat_art.artwork_id}"]')
    await page.wait_for(images_loaded.format(sel="#detail img"), "detail images")
    kid.server.request("POST", "/api/admin/select", {"kid_theme": "dark"})
    await page.wait_for("document.documentElement.dataset.theme === 'dark'", "the dark kid theme")

    # Keep drawing: the cat goes back on the canvas, in the side-by-side view.
    count = await page.eval("Number(document.querySelector('#stage').dataset.pictures || 0)")
    await page.tap_selector("#detail-continue")
    if await page.eval("!document.querySelector('#confirm').hidden"):
        await page.tap_selector("#confirm-yes")
    await page.wait_for("document.querySelector('#gallery').hidden", "the gallery to close", timeout=20)
    await page.wait_for(f"Number(document.querySelector('#stage').dataset.pictures || 0) > {count}",
                        "a picture for the restored drawing", timeout=40)
    await kid.mode("split")
    await kid.draw([Stroke(YELLOW, 12, circle(175, 215, 24, 15)), Stroke(YELLOW, 10, [(151, 215), (126, 198), (128, 232), (151, 215)])])
    await kid.wait_refined()
    await kid.shot("ipad-landscape-cat-keep-drawing", "Keep drawing a saved picture",
                   "'Keep drawing' puts the saved drawing back on the canvas (here a second fish "
                   "was added); saving again updates the same picture.", ["keep drawing", "split view"],
                   artwork="cat")

    # The same drawing, refined, in the light look and in the dark Auto view.
    kid.server.request("POST", "/api/admin/select", {"kid_theme": "light"})
    await page.wait_for("document.documentElement.dataset.theme === 'light'", "the light kid theme")
    await asyncio.sleep(0.6)
    await kid.shot("ipad-landscape-cat-split-light", "Light look: drawing and picture",
                   "The drawing page in the light look, with the 'Both' view.", ["light theme", "split view"],
                   artwork="cat")
    kid.server.request("POST", "/api/admin/select", {"kid_theme": "dark"})
    await page.wait_for("document.documentElement.dataset.theme === 'dark'", "the dark kid theme")
    await kid.mode("auto")


# ============================================================================ admin page

async def admin_shots(cdp: Cdp, server: Server, kid: Kid | None, out: Output, scheme: str) -> None:
    page = await Page.create(cdp, PHONE)
    try:
        await page.set_color_scheme(scheme)
        await page.navigate(server.base + "/admin")
        await page.wait_for("!document.querySelector('#login').hidden", "the PIN pad")
        await asyncio.sleep(0.5)

        async def shot(stem: str, title: str, caption: str, features: list[str], clip: dict | None = None) -> None:
            await page.wait_for("!document.querySelector('#toast').classList.contains('show')",
                                "no toast on the admin page", timeout=10)
            await asyncio.sleep(0.3)
            out.save("admin", f"phone-admin-{stem}-{scheme}", await page.screenshot(clip), title=title,
                     caption=caption, device="phone", orientation="portrait", theme=scheme, features=features)

        await shot("login", "Parent page: PIN pad", "The parent page on a phone is protected by a PIN.",
                   ["admin", "login"])
        for key in [*TEST_PIN, "ok"]:
            await page.tap_selector(f'#pin-pad [data-k="{key}"]', scroll=False)
        await page.wait_for("!document.querySelector('#admin').hidden", "the admin page after login")
        await page.wait_for("document.querySelector('#p-creativity').value !== '' && document.querySelector('#o-gpu').textContent !== ''",
                            "the picture controls")
        if kid is not None:  # a fresh picture on the iPad shows up live on the phone
            await kid.page.send("Page.bringToFront")
            await kid.pick("styles", await kid.page.eval("window.dreamer.toolbar.style"))
            await kid.wait_refined(minimum=4.0)
        await page.send("Page.bringToFront")
        await page.wait_for("!document.querySelector('#live-img').hidden && document.querySelector('#live-img').naturalWidth > 0",
                            "the live picture on the admin page", timeout=20)
        await page.wait_for("document.querySelector('#live-state').dataset.state === 'ready'", "engine state Ready")
        await page.eval("window.scrollTo(0, 0)")
        await asyncio.sleep(1.2)
        await shot("now", "Parent page: Now", "The live picture from the iPad, the engine's speed, and the "
                   "current topic, style, starting view and page look.", ["admin", "live picture", "topics", "styles"])
        await page.scroll_into_view("#h-params", block="start", offset=-64)
        await shot("picture-controls", "Parent page: picture controls",
                   "Creativity, draw reaction, refinement and more; changes apply live. Each ⓘ opens the help page.",
                   ["admin", "picture controls"])
        await page.scroll_into_view("#gpu-meter", block="center")
        await shot("memory-meter", "Parent page: upscale, resolution and memory meter",
                   "AI upscale, resolution and an estimate of the graphics memory; settings that would not fit are disabled.",
                   ["admin", "picture controls", "memory meter", "upscale"])
        await page.wait_for("document.querySelectorAll('#art-grid img').length > 0", "the admin gallery")
        await page.scroll_into_view("#art-grid", block="center", offset=40)
        await asyncio.sleep(0.4)  # the thumbnails load lazily, once they are on screen
        await page.wait_for("[...document.querySelectorAll('#art-grid img')].filter((img) => {"
                            " const r = img.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; })"
                            ".every((img) => img.complete && img.naturalWidth > 0)",
                            "the admin gallery thumbnails")
        await shot("gallery", "Parent page: gallery", "Saved pictures with their names; deleted ones go to the Trash first.",
                   ["admin", "gallery"])
        await page.tap_selector("#art-grid .art-thumb")
        await page.wait_for("!document.querySelector('#art-sheet').hidden && ['#art-picture', '#art-sketch']"
                            ".every((sel) => document.querySelector(sel).complete && document.querySelector(sel).naturalWidth > 0)",
                            "the artwork sheet")
        await asyncio.sleep(0.6)
        await shot("artwork", "Parent page: one picture", "Rename a picture, download the picture or the drawing, "
                   "or move it to the Trash.", ["admin", "gallery", "download"])
        await page.tap_selector("#art-close", scroll=False)
        errors = [error for error in page.errors if "401" not in error]  # the pre-login state check
        if errors:
            log(f"  admin ({scheme}) page errors: {errors}")
    finally:
        await page.close()


# ============================================================================ help page

async def help_shots(cdp: Cdp, server: Server, out: Output) -> None:
    visible_images = ("[...document.images].filter((img) => { const r = img.getBoundingClientRect();"
                      " return r.bottom > 0 && r.top < innerHeight && r.width > 0; })"
                      ".every((img) => img.complete && img.naturalWidth > 0)")
    for device, scheme in ((PHONE, "light"), (DESKTOP, "light")):
        page = await Page.create(cdp, device)
        try:
            await page.set_color_scheme(scheme)
            await page.navigate(server.base + "/static/help.html")
            await page.wait_for("document.querySelectorAll('section.control img').length > 0", "help images")
            await asyncio.sleep(0.5)
            await page.wait_for(visible_images, "visible help images", timeout=20)
            await asyncio.sleep(0.3)
            out.save("help", f"{device.name}-help-top", await page.screenshot(), title="Help: what the picture controls do",
                     caption="Every control on the parent page has a help section with real example pictures made on the PC.",
                     device=device.name, orientation=device.orientation, theme=scheme, features=["help"])
            await page.scroll_into_view("#refine", block="start", offset=-8)
            await page.wait_for(visible_images, "visible help images", timeout=20)
            await asyncio.sleep(0.4)
            out.save("help", f"{device.name}-help-refine", await page.screenshot(), title="Help: refine when they stop",
                     caption="The help for 'Refine when they stop': the same drawing at 0 to 20 refinement steps.",
                     device=device.name, orientation=device.orientation, theme=scheme, features=["help", "refine"])
            if page.errors:
                log(f"  help page errors: {page.errors}")
        finally:
            await page.close()


# ============================================================================ examples

def save_examples(server: Server, artworks: list[Artwork], out: Output) -> None:
    for art in artworks:
        if not art.artwork_id:
            continue
        info = server.get(f"/api/artworks/{art.artwork_id}")
        info = info.get("artwork", info)
        sketch = Image.open(io.BytesIO(server.get(info["sketch"]))).convert("RGBA")
        paper = Image.new("RGBA", sketch.size, info.get("background") or art.paper)
        paper.alpha_composite(sketch)
        buffer = io.BytesIO()
        paper.convert("RGB").save(buffer, "PNG", optimize=True)
        meta = dict(title=f"{art.title}: drawing", device="none", theme="none", artwork=art.name,
                    topic=art.topic, style=art.style, paper=art.paper, orientation=art.orientation)
        out.save_example(EXAMPLES / f"{art.name}-sketch.png", buffer.getvalue(), paper.size,
                         caption=f"The child's drawing over its paper colour ({feature_caption(art)}).",
                         features=["example", "before"], **meta)
        picture = server.get(info["picture"])
        size = Image.open(io.BytesIO(picture)).size
        meta["title"] = f"{art.title}: picture"
        out.save_example(EXAMPLES / f"{art.name}-picture.jpg", picture, size,
                         caption=f"The refined, upscaled picture Dreamer made from it ({feature_caption(art)}).",
                         features=["example", "after"], **meta)


# ============================================================================ main

async def run(groups: set[str], server: Server, out: Output) -> None:
    edge = Edge.launch(extra_args=["--disable-gpu"] if server.mock else [])
    cdp = None
    try:
        cdp = await Cdp.connect(edge.ws_url)
        version = await cdp.send("Browser.getVersion")
        log(f"browser: {version.get('product')}")
        kid = None
        artworks: list[Artwork] = []
        if groups & {"kid", "admin", "examples"}:
            page = await Page.create(cdp, IPAD_LANDSCAPE)
            kid = Kid(page, server, out)
            await kid.open()
            await asyncio.sleep(6.5)  # the Paper button's first-visit wiggle ends
            await kid.shot("ipad-landscape-start", "Ready to draw",
                           "The drawing page on the iPad: colours, brush size, worlds and styles on top; "
                           "Save, Gallery, New and Paper on the side.", ["toolbar", "topics", "styles"])
            artworks = await make_artworks(kid, out)
            if "kid" in groups:
                await kid_extras(kid, artworks)
            if page.errors:
                log(f"  kid page errors: {page.errors}")
        if "admin" in groups:
            for scheme in ("dark", "light"):
                log(f"admin page ({scheme})")
                await admin_shots(cdp, server, kid, out, scheme)
        if "help" in groups:
            log("help page")
            await help_shots(cdp, server, out)
        if "examples" in groups:
            log("example pairs")
            save_examples(server, artworks, out)
    finally:
        if cdp is not None:
            await cdp.close()
        edge.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[],
                        help=f"groups to regenerate ({', '.join(GROUPS)}); comma separated or repeated")
    parser.add_argument("--mock", action="store_true",
                        help="CPU only: mock backend (tinted sketches) and Edge with --disable-gpu; "
                             "writes to a temp folder unless --out is given")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: screenshots/)")
    args = parser.parse_args()
    global OUT, EXAMPLES
    if args.out:
        OUT = args.out.resolve()
    elif args.mock:  # never overwrite the real screenshots with mock pictures
        OUT = Path(tempfile.gettempdir()) / "dreamer-showcase-mock"
    EXAMPLES = OUT / "examples"
    groups = {name.strip() for value in args.only for name in value.split(",") if name.strip()} or set(GROUPS)
    unknown = groups - set(GROUPS)
    if unknown:
        parser.error(f"unknown group(s): {', '.join(sorted(unknown))}; pick from {', '.join(GROUPS)}")
    started = time.monotonic()
    out = Output(groups)
    server = Server(mock=args.mock)
    ok = False
    try:
        server.start()
        server.login()
        params = server.request("PATCH", "/api/admin/params", {"refine_steps": 8, "upscale": "x2"})
        applied = params.get("params", params)
        log(f"output: {OUT}{' (mock backend)' if args.mock else ''}")
        log(f"params: refine_steps={applied.get('refine_steps')} upscale={applied.get('upscale')}")
        asyncio.run(run(groups, server, out))
        out.finish(full_run=groups == set(GROUPS))
        ok = True
    except (ShowcaseError, CdpError, OSError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print("\n--- server log (tail) ---\n" + server.tail(), file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        server.stop()
        server.remove_log()  # its tail was printed above on failure
    log(f"{'done' if ok else 'failed'} in {time.monotonic() - started:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
