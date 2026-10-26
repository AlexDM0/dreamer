"""Render the app icons from static/favicon.svg (the Dreamer mark) and
static/admin-icon.svg (the parent page's variant).

    .venv\\Scripts\\python.exe tools\\make_icons.py

Writes the home-screen icons the iPad uses when the page is saved as a web app
(apple-touch-icon 152/167/180), the web app manifest icons (192/512) and the Windows
shortcut icon (static/dreamer.ico), plus admin-icon-*.png for the parent page. Renders
once at 1024 px with headless Microsoft Edge (already on Windows) and scales down, so every size is crisp. iOS rounds the corners
itself, so the icons are full-bleed squares on the mark's night-sky colours.
"""
from __future__ import annotations

import base64
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC = REPO_ROOT / "static"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
RENDER_SIZE = 1024
PNG_SIZES = {"icon-152.png": 152, "icon-167.png": 167, "icon-180.png": 180,
             "icon-192.png": 192, "icon-512.png": 512}
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
# Each mark and the colours behind its tile (the tile is scaled past the edges).
MARKS = {"": ("favicon.svg", "#7a6ef0,#4134ae"), "admin-": ("admin-icon.svg", "#34302b,#141311")}


def render_master(out_png: Path, svg: str = "favicon.svg", sky: str = MARKS[""][1]) -> None:
    mark = base64.b64encode((STATIC / svg).read_bytes()).decode("ascii")
    # The mark's own tile has rounded corners: fill behind it with the same sky gradient and
    # scale it slightly past the edges (not so far that iOS's rounded corners clip the star).
    html = f"""<!doctype html><html><body style="margin:0;overflow:hidden">
      <div style="width:{RENDER_SIZE}px;height:{RENDER_SIZE}px;overflow:hidden;
                  background:linear-gradient(135deg,{sky})">
        <img src="data:image/svg+xml;base64,{mark}"
             style="width:104%;height:104%;margin:-2%;display:block">
      </div></body></html>"""
    with tempfile.TemporaryDirectory() as temp:
        page = Path(temp) / "icon.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={RENDER_SIZE},{RENDER_SIZE}", "--force-device-scale-factor=1",
                        f"--user-data-dir={Path(temp) / 'profile'}", "--virtual-time-budget=2000",
                        f"--screenshot={out_png}", page.as_uri()],
                       check=True, capture_output=True, timeout=60)


def main() -> int:
    if not EDGE.is_file():
        print(f"Microsoft Edge not found at {EDGE}", file=sys.stderr)
        return 1
    for prefix, (svg, sky) in MARKS.items():
        with tempfile.TemporaryDirectory() as temp:
            master_path = Path(temp) / "master.png"
            render_master(master_path, svg, sky)
            master = Image.open(master_path).convert("RGB").crop((0, 0, RENDER_SIZE, RENDER_SIZE))
        for name, size in PNG_SIZES.items():
            master.resize((size, size), Image.Resampling.LANCZOS).save(STATIC / (prefix + name), optimize=True)
            print("wrote", STATIC / (prefix + name))
        if not prefix:
            master.resize((256, 256), Image.Resampling.LANCZOS).save(
                STATIC / "dreamer.ico", sizes=[(size, size) for size in ICO_SIZES])
            print("wrote", STATIC / "dreamer.ico")
    return 0


if __name__ == "__main__":
    sys.exit(main())
