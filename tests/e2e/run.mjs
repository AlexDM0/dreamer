// Dreamer end-to-end test: real server (mock backend) + Chromium via Playwright.
//
// Usage (from the repo root):  node tests/e2e/run.mjs
// Needs: .venv with requirements.txt installed, and the `playwright` npm package
// (global install / NODE_PATH is fine). Screenshots go to tests/e2e/screenshots/.
// Exits 0 on success, 1 on the first failed check.
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SHOTS = join(ROOT, "tests", "e2e", "screenshots");
const PIN = "4321";
const PYTHON = [join(ROOT, ".venv", "bin", "python"), join(ROOT, ".venv", "Scripts", "python.exe")]
  .find(existsSync);

// ---------------------------------------------------------------- helpers

let step = 0;
let serverLog = "";
function check(cond, what) {
  step += 1;
  if (!cond) throw new Error(`check ${step} failed: ${what}`);
  console.log(`  ok ${step} - ${what}`);
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function waitForHealth(base, proc, timeoutMs = 20000) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    if (proc.exitCode !== null) throw new Error(`server exited with ${proc.exitCode}`);
    try {
      const res = await fetch(`${base}/healthz`);
      if (res.ok && (await res.json()).state === "ready") return;
    } catch { /* not listening yet */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("server did not become ready");
}

/** Count clearly non-white pixels on a canvas (sampled). */
function inkOn(page, selector) {
  return page.evaluate((sel) => {
    const c = document.querySelector(sel);
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 16) {
      if (255 - Math.min(d[i], d[i + 1], d[i + 2]) > 40) n++;
    }
    return n;
  }, selector);
}

/** Draw a stroke through the given points (fractions of the canvas box) with the mouse. */
async function stroke(page, points) {
  const box = await page.locator("#sketch").boundingBox();
  const at = ([fx, fy]) => [box.x + fx * box.width, box.y + fy * box.height];
  await page.mouse.move(...at(points[0]));
  await page.mouse.down();
  for (const p of points.slice(1)) await page.mouse.move(...at(p), { steps: 8 });
  await page.mouse.up();
}

const pictures = (page) => page.$eval("#stage", (el) => Number(el.dataset.pictures || 0));
const shot = (page, name) => page.screenshot({ path: join(SHOTS, `${name}.png`) });

// ---------------------------------------------------------------- main

async function main() {
  if (!PYTHON) throw new Error("no .venv found; create it and install requirements.txt");
  mkdirSync(SHOTS, { recursive: true });
  const tmp = mkdtempSync(join(tmpdir(), "dreamer-e2e-"));
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  const configPath = join(tmp, "config.yaml");
  writeFileSync(configPath, [
    "host: 127.0.0.1",
    `port: ${port}`,
    "backend: mock",
    `admin_pin: "${PIN}"`,
    `data_dir: ${JSON.stringify(join(tmp, "data"))}`,
    "mock_delay_ms: 20",
    "",
  ].join("\n"));

  const server = spawn(PYTHON, ["-m", "server"], {
    cwd: ROOT,
    env: { ...process.env, SKETCH_CONFIG: configPath, PYTHONUNBUFFERED: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  server.stdout.on("data", (d) => { serverLog += d; });
  server.stderr.on("data", (d) => { serverLog += d; });

  const browser = await chromium.launch();
  const errors = [];
  try {
    await waitForHealth(base, server);
    console.log(`server ready on ${base}`);

    // ----- kid page on an iPad-sized touch screen
    const kidCtx = await browser.newContext({
      viewport: { width: 1180, height: 820 }, hasTouch: true, deviceScaleFactor: 1,
    });
    const kid = await kidCtx.newPage();
    kid.on("pageerror", (e) => errors.push(`kid: ${e.message}`));
    await kid.goto(base + "/");
    await kid.waitForSelector('#conn[data-state="ready"]', { timeout: 10000 });
    await kid.waitForSelector("#topics .pick[data-id]");
    check(true, "kid page connects and lists topics");
    await shot(kid, "01-kid-empty");

    // Draw: a house-ish shape and a sun.
    const before = await pictures(kid);
    await stroke(kid, [[0.2, 0.8], [0.2, 0.45], [0.5, 0.2], [0.8, 0.45], [0.8, 0.8], [0.2, 0.8]]);
    await stroke(kid, [[0.75, 0.15], [0.85, 0.12], [0.88, 0.22], [0.78, 0.25], [0.75, 0.15]]);
    check((await inkOn(kid, "#sketch")) > 50, "strokes appear on the sketch canvas");
    await kid.waitForFunction((n) => Number(document.querySelector("#stage").dataset.pictures) > n,
                              before, { timeout: 10000 });
    await kid.waitForFunction(() => !window.dreamer.net.awaitingFinal, null, { timeout: 10000 });
    await kid.waitForTimeout(300); // let the cross-fade finish
    check((await inkOn(kid, "#picture")) > 200, "a generated picture is drawn on the picture canvas");
    await shot(kid, "02-kid-picture-auto");

    // ----- view modes
    for (const mode of ["sketch", "picture", "split", "auto"]) {
      await kid.click(`#modes [data-mode="${mode}"]`);
      check(await kid.$eval("#stage", (el, m) => el.classList.contains(`mode-${m}`), mode),
            `view mode ${mode} applies class mode-${mode}`);
      check(await kid.$eval(`[data-mode="${mode}"]`, (b) => b.classList.contains("active")),
            `mode button ${mode} is highlighted`);
      if (mode === "split") await shot(kid, "03-kid-split");
    }
    await kid.reload();
    await kid.waitForSelector('#conn[data-state="ready"]');
    check(await kid.$eval("#stage", (el) => el.classList.contains("mode-auto")),
          "view mode survives a reload");

    // Draw again after the reload so there is something to clear.
    await stroke(kid, [[0.3, 0.3], [0.7, 0.7]]);
    await kid.waitForFunction(() => !window.dreamer.net.awaitingFinal
                              && Number(document.querySelector("#stage").dataset.pictures) > 0,
                              null, { timeout: 10000 });

    // ----- clear
    await kid.click("#clear");
    check((await inkOn(kid, "#sketch")) === 0, "clear empties the sketch canvas");
    await kid.waitForFunction(() => !window.dreamer.net.awaitingFinal, null, { timeout: 10000 });
    await kid.waitForTimeout(300);
    check((await inkOn(kid, "#picture")) === 0, "clear results in a blank picture");
    await shot(kid, "04-kid-cleared");

    // Draw something so the admin live view has a picture.
    await stroke(kid, [[0.25, 0.5], [0.5, 0.25], [0.75, 0.5], [0.5, 0.75], [0.25, 0.5]]);

    // ----- admin page on a phone
    const adminCtx = await browser.newContext({
      viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, deviceScaleFactor: 2,
    });
    const admin = await adminCtx.newPage();
    admin.on("pageerror", (e) => errors.push(`admin: ${e.message}`));
    await admin.goto(base + "/admin");
    await admin.waitForSelector("#login:not([hidden])");
    await shot(admin, "05-admin-pin");
    for (const k of [...PIN, "ok"]) await admin.tap(`#pin-pad [data-k="${k}"]`);
    await admin.waitForSelector("#admin:not([hidden])", { timeout: 10000 });
    check(true, "admin logs in with the PIN pad");
    await admin.waitForSelector("#live-img:not([hidden])", { timeout: 10000 });
    check(true, "admin live view shows the kid's picture");

    await admin.tap("#add-style summary");
    await admin.fill('#add-style-form [name="emoji"]', "🌈");
    await admin.fill('#add-style-form [name="label"]', "Rainbow Party");
    await admin.fill('#add-style-form [name="prompt"]', "rainbow colors, shiny stickers");
    await admin.tap('#add-style-form button[type="submit"]');
    await admin.waitForSelector('#styles-list >> text="Rainbow Party"', { timeout: 10000 });
    check(true, "custom style shows in the admin list");
    await shot(admin, "06-admin-styles");

    // ----- the new style reaches the kid page without a reload
    const btn = kid.locator("#styles .pick", { hasText: "Rainbow Party" });
    await btn.waitFor({ timeout: 10000 });
    check(true, "new style button appears on the kid page without reload");
    const n = await pictures(kid);
    await btn.click();
    check(await btn.evaluate((b) => b.classList.contains("active")), "tapping the new style selects it");
    await kid.waitForFunction((k) => Number(document.querySelector("#stage").dataset.pictures) > k,
                              n, { timeout: 10000 });
    check(true, "choosing a style re-renders the picture");
    await kid.waitForTimeout(300);
    await shot(kid, "07-kid-new-style");

    // ----- live picture controls: more creativity changes the picture on the kid page
    const picSig = () => kid.evaluate(() => {
      const c = document.querySelector("#picture");
      const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let h = 0;
      for (let i = 0; i < d.length; i += 64) h = (h * 31 + d[i]) >>> 0;
      return h;
    });
    await admin.waitForFunction(() => document.querySelector("#p-creativity").value !== "", null,
                                { timeout: 10000 });
    const sigBefore = await picSig();
    const patched = admin.waitForResponse((r) => r.url().endsWith("/api/admin/params")
                                          && r.request().method() === "PATCH");
    await admin.$eval("#p-creativity", (el) => {
      el.value = "95";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    check((await patched).ok(), "admin creativity slider PATCHes the params");
    const n2 = await pictures(kid);
    await kid.evaluate(() => window.dreamer.net.sendNow()); // same drawing again
    await kid.waitForFunction((k) => Number(document.querySelector("#stage").dataset.pictures) > k,
                              n2, { timeout: 10000 });
    await kid.waitForTimeout(200);
    check((await picSig()) !== sigBefore, "picture on the kid page changes with creativity");
    await shot(admin, "08-admin-params");

    check(errors.length === 0, `no page errors${errors.length ? ": " + errors.join("; ") : ""}`);
    console.log(`\nPASS (${step} checks)`);
  } finally {
    await browser.close();
    server.kill("SIGTERM");
    await new Promise((r) => (server.exitCode !== null ? r() : server.once("exit", r)));
    rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(`\nFAIL: ${err.message}`);
  console.error("\n--- server log (tail) ---\n" + serverLog.split("\n").slice(-40).join("\n"));
  process.exitCode = 1;
});
