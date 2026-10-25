"""Render the app icons from static/favicon.svg (the Dreamer mark).

    .venv\\Scripts\\python.exe tools\\make_icons.py

Writes the home-screen icons the iPad uses when the page is saved as a web app
(apple-touch-icon 152/167/180), the web app manifest icons (192/512) and the Windows
shortcut icon (static/dreamer.ico). Renders once at 1024 px with headless Microsoft Edge
(already on Windows) and scales down, so every size is crisp. iOS rounds the corners
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


def render_master(out_png: Path) -> None:
    mark = base64.b64encode((STATIC / "favicon.svg").read_bytes()).decode("ascii")
    # The mark's own tile has rounded corners: fill behind it with the same sky gradient and
    # scale it slightly past the edges (not so far that iOS's rounded corners clip the star).
    html = f"""<!doctype html><html><body style="margin:0;overflow:hidden">
      <div style="width:{RENDER_SIZE}px;height:{RENDER_SIZE}px;overflow:hidden;
                  background:linear-gradient(135deg,#7a6ef0,#4134ae)">
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
    with tempfile.TemporaryDirectory() as temp:
        master_path = Path(temp) / "master.png"
        render_master(master_path)
        master = Image.open(master_path).convert("RGB").crop((0, 0, RENDER_SIZE, RENDER_SIZE))
    for name, size in PNG_SIZES.items():
        master.resize((size, size), Image.Resampling.LANCZOS).save(STATIC / name, optimize=True)
        print("wrote", STATIC / name)
    master.resize((256, 256), Image.Resampling.LANCZOS).save(
        STATIC / "dreamer.ico", sizes=[(size, size) for size in ICO_SIZES])
    print("wrote", STATIC / "dreamer.ico")
    return 0


if __name__ == "__main__":
    sys.exit(main())
