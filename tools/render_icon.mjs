// Render static/icon-180.png (apple-touch-icon) from static/favicon.svg.
// Usage: node tools/render_icon.mjs
// Needs the `playwright` npm package (resolved via NODE_PATH / global install).
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const mark = readFileSync(join(root, "static", "favicon.svg"));
const src = "data:image/svg+xml;base64," + mark.toString("base64");

// Mark on a soft sky -> sun gradient. iOS rounds the corners itself, so fill the square.
const html = `<!doctype html><html><body style="margin:0">
  <div style="width:180px;height:180px;display:grid;place-items:center;
              background:radial-gradient(circle at 30% 20%, #fff4c2 0%, transparent 55%),
                         linear-gradient(160deg, #bfe6ff 0%, #8fd3ff 45%, #ffd2d8 100%)">
    <img src="${src}" style="width:148px;height:148px;margin-top:6px">
  </div></body></html>`;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 180, height: 180 }, deviceScaleFactor: 1 });
await page.setContent(html);
await page.waitForFunction(() => document.images[0].complete);
await page.screenshot({ path: join(root, "static", "icon-180.png"), clip: { x: 0, y: 0, width: 180, height: 180 } });
await browser.close();
console.log("wrote static/icon-180.png");
