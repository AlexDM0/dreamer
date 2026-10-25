// Dreamer kid page: wires the toolbar, drawing canvas, view and network together.
import { DrawingCanvas } from "./canvas.js";
import { Gallery } from "./gallery.js";
import { Net } from "./net.js";
import { Toolbar } from "./toolbar.js";
import { View } from "./view.js";

const $ = (sel) => document.querySelector(sel);

// ---- Safari hardening: no pinch zoom, no double-tap zoom, no rubber-band scrolling ----
for (const type of ["gesturestart", "gesturechange", "gestureend"]) {
  document.addEventListener(type, (e) => e.preventDefault(), { passive: false });
}
let lastTouchEnd = 0;
document.addEventListener("touchend", (e) => {
  const now = Date.now();
  // Buttons keep fast repeated taps; CSS touch-action: manipulation covers them.
  if (now - lastTouchEnd < 350 && !e.target.closest("button, input, label")) e.preventDefault();
  lastTouchEnd = now;
}, { passive: false });
document.addEventListener("dblclick", (e) => e.preventDefault());
// Allow horizontal scrolling of the pick rows only; everything else stays put.
document.addEventListener("touchmove", (e) => {
  if (!e.target.closest(".pick-row, .gallery-scroll")) e.preventDefault();
}, { passive: false });

// ---- modules ----
const canvas = new DrawingCanvas($("#sketch"));
const toolbar = new Toolbar();
const view = new View($("#stage-wrap"), $("#stage"), $("#picture"));
const net = new Net(canvas);
const gallery = new Gallery();

canvas.setColor(toolbar.color);
canvas.setWidth(toolbar.width);
toolbar.setMode(view.mode);

// Toolbar -> canvas / net / view
toolbar.addEventListener("color", (e) => canvas.setColor(e.detail.color));
toolbar.addEventListener("width", (e) => canvas.setWidth(e.detail.width));
// Clear also drops the autosaved draft, so an unsaved drawing is asked about first (the
// big red button is easy to hit by accident).
toolbar.addEventListener("clear", () => {
  confirmThen(() => {
    canvas.clear();
    view.strokeStart(); // show the (blank) sketch right away in Auto
    view.strokeEnd();
    net.clear();
    unsaved = false;
    drewSomething = false;
    hideHint();
  }, "Clear this drawing?", "Clear it");
});
toolbar.addEventListener("mode", (e) => view.setMode(e.detail.mode));
// A new topic/style re-renders the current drawing (the socket keeps message order, so
// the server applies the selection before it sees this sketch).
toolbar.addEventListener("topic", (e) => { net.selectTopic(e.detail.id); net.sendNow(); });
toolbar.addEventListener("style", (e) => { net.selectStyle(e.detail.id); net.sendNow(); });

// Canvas -> view / net
canvas.addEventListener("strokestart", () => {
  view.strokeStart();
  drewSomething = true;
  hideHint(); // drawing on means "keep it": the Start fresh offer has done its job
});
canvas.addEventListener("strokeend", () => {
  view.strokeEnd();
  net.sendNow();
  unsaved = true;
});

// View -> toolbar
view.addEventListener("mode", (e) => toolbar.setMode(e.detail.mode));
view.addEventListener("layout", () => toolbar.setScale(canvas.displayScale()));
toolbar.setScale(canvas.displayScale());

// Net -> everything
let backendState = null; // last status from the server on this connection
function showConn() {
  if (net.state === "open") toolbar.setConn(backendState || "connecting");
  else toolbar.setConn(net.state);
}

net.addEventListener("state", (e) => {
  $("#replaced").hidden = e.detail.state !== "replaced";
  if (e.detail.state === "open") backendState = null;
  // A reply that never came on the old socket must not swallow later error statuses.
  if (e.detail.state !== "open") { continuing = false; if (saving) saveDone(); }
  showConn();
});
net.addEventListener("status", (e) => {
  if (e.detail.state === "error" && saving) {
    // The answer to our Save (e.g. "Nothing to save yet"); the server re-sends the real
    // backend state right after, so the connection dot is not touched.
    saveDone();
    showHint(e.detail.msg === "Nothing to save yet" ? "Draw something first! ✏️" : e.detail.msg);
    return;
  }
  if (e.detail.state === "error" && continuing) {
    continuing = false;
    showHint(e.detail.msg);
    return;
  }
  backendState = e.detail.state;
  if (net.state === "open") toolbar.setConn(backendState, e.detail.msg);
});
net.addEventListener("ready", (e) => {
  const m = e.detail;
  if (m.t === "ready") {
    // Only on the first connection after loading the page: bring back the drawing the
    // kid left (a reconnect keeps what is on the canvas).
    if (!firstReadyDone && m.draft && !drewSomething) restoreDraft(m.draft);
    firstReadyDone = true;
    if (gallery.isOpen && gallery.failed) gallery.update(); // back online: try again
  }
  // The parent changed the topic/style from the admin page: re-render the drawing.
  // Compare with what the toolbar shows (it includes the kid's own picks, which the
  // server does not echo), not with the server's previous announcement.
  if (m.t === "styles" && (m.topic !== toolbar.topic || m.style !== toolbar.style)) net.sendNow();
  if (Array.isArray(m.topics)) toolbar.setItems("topics", m.topics, m.topic);
  if (Array.isArray(m.styles)) toolbar.setItems("styles", m.styles, m.style);
  if (m.view_mode) view.setDefaultMode(m.view_mode);
  // A ready reply with no status yet: the server is up, assume the backend is too.
  if (m.t === "ready" && !backendState) backendState = "ready";
  showConn();
});
net.addEventListener("picture", (e) => {
  // Recovery from "error" is announced by the server's status message; a picture alone
  // proves nothing (a blank canvas is answered without the backend).
  view.showPicture(e.detail.jpeg, e.detail.final);
});

$("#take-over").addEventListener("click", () => net.takeOver());

// ---- Gallery: save, new drawing, draft restore, keep drawing (SPEC-addendum-gallery) ----

let unsaved = false;        // strokes since the last save / load
let drewSomething = false;  // the kid drew on this page (a draft must not overwrite that)
let firstReadyDone = false;
let saving = false;
let continuing = false;
let hintTimer = 0;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

function showHint(text, button = null, ms = 3200) {
  clearTimeout(hintTimer);
  $("#hint-text").textContent = text;
  const btn = $("#hint-btn");
  btn.hidden = !button;
  if (button) {
    btn.textContent = button.label;
    btn.onclick = () => { hideHint(); button.onClick(); };
  }
  const hint = $("#hint");
  hint.classList.toggle("floating", gallery.isOpen); // stay visible over the gallery
  hint.hidden = false;
  hint.classList.remove("pop");
  void hint.offsetWidth;
  hint.classList.add("pop");
  hintTimer = setTimeout(hideHint, ms);
}

function hideHint() {
  clearTimeout(hintTimer);
  $("#hint").hidden = true;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not load " + url));
    img.src = url;
  });
}

/** Put a saved sketch on the canvas and get a fresh picture for it. `unless()` is checked
 * once the image has loaded: true means the canvas must not be replaced after all.
 * `loaded()` runs once the sketch is on the canvas, before it is sent (it tells the
 * server which drawing the canvas holds now). */
async function loadSketch(url, unless = () => false, loaded = () => {}) {
  const img = await loadImage(url);
  if (unless()) return false;
  canvas.loadImage(img);
  net.dropOlderPictures();
  loaded();
  view.strokeStart(); // show the sketch now; Auto fades to the picture when it arrives
  view.strokeEnd();
  net.sendNow();
  return true;
}

async function restoreDraft(draft) {
  try {
    // The kid may already be drawing while the draft loads: never paint over that.
    if (!await loadSketch(draft.sketch, () => drewSomething,
                          () => net.restored(draft.id, draft.source))) return;
  } catch {
    return; // the draft went away meanwhile: just start with an empty canvas
  }
  unsaved = true; // a draft is not in the gallery yet
  showHint("Welcome back! Here is your drawing.", { label: "Start fresh", onClick: startFresh }, 7000);
}

function startFresh() {
  canvas.clear();
  view.strokeStart();
  view.strokeEnd();
  net.newDrawing();
  unsaved = false;
  drewSomething = false;
}

/** Ask before throwing away an unsaved drawing; runs `fn` when it is fine to go on. */
function confirmThen(fn, title, yes) {
  if (!unsaved || canvas.isBlank()) { fn(); return; }
  $("#confirm-title").textContent = title;
  $("#confirm-yes").textContent = yes;
  const box = $("#confirm");
  box.hidden = false;
  const done = (go) => {
    box.hidden = true;
    $("#confirm-yes").onclick = $("#confirm-no").onclick = null;
    if (go) fn();
  };
  $("#confirm-yes").onclick = () => done(true);
  $("#confirm-no").onclick = () => done(false);
}

$("#new-drawing").addEventListener("click", () => {
  confirmThen(() => { hideHint(); startFresh(); }, "Start a new drawing?", "Start new");
});

function waitForFinalPicture(timeoutMs = 3000) {
  return new Promise((resolve) => {
    const end = Date.now() + timeoutMs;
    const tick = () => {
      if (!net.awaitingFinal || Date.now() > end) resolve();
      else setTimeout(tick, 50);
    };
    tick();
  });
}

function saveDone() {
  saving = false;
  $("#save").classList.remove("busy");
}

$("#save").addEventListener("click", async () => {
  if (saving) return;
  if (canvas.isBlank()) { showHint("Draw something first! ✏️"); return; }
  if (!net.live) { showHint("Not connected yet. Try again in a moment."); return; }
  saving = true;
  $("#save").classList.add("busy");
  await waitForFinalPicture(); // save exactly the picture made from the whole drawing
  if (!net.save()) { saveDone(); showHint("Not connected yet. Try again in a moment."); return; }
  setTimeout(() => { if (saving) saveDone(); }, 8000); // never stay stuck
});

net.addEventListener("saved", () => {
  saveDone();
  unsaved = false;
  celebrateSave();
  if (gallery.isOpen) gallery.update();
});

/** The saved picture flies into the Gallery button, with a little star burst. */
function celebrateSave() {
  const saveBtn = $("#save");
  const label = $("#save-label");
  label.textContent = "Saved!";
  saveBtn.classList.remove("saved");
  void saveBtn.offsetWidth;
  saveBtn.classList.add("saved");
  setTimeout(() => { label.textContent = "Save"; saveBtn.classList.remove("saved"); }, 1600);
  burst(saveBtn);

  const target = $("#open-gallery");
  const bump = () => {
    target.classList.remove("gulp");
    void target.offsetWidth;
    target.classList.add("gulp");
  };
  if (reduceMotion.matches || !Element.prototype.animate) { bump(); return; }

  const src = view.mode === "sketch" ? $("#sketch") : $("#picture");
  const from = src.getBoundingClientRect();
  const to = target.querySelector(".disc").getBoundingClientRect();
  const fly = document.createElement("img");
  fly.className = "fly";
  fly.alt = "";
  fly.src = $("#picture").toDataURL("image/jpeg", 0.8);
  Object.assign(fly.style, {
    left: `${from.left}px`, top: `${from.top}px`, width: `${from.width}px`, height: `${from.height}px`,
  });
  document.body.append(fly);
  const dx = to.left + to.width / 2 - (from.left + from.width / 2);
  const dy = to.top + to.height / 2 - (from.top + from.height / 2);
  const end = Math.max(0.05, to.width / from.width);
  const anim = fly.animate([
    { transform: "translate(0, 0) scale(1) rotate(0deg)", opacity: 1, offset: 0 },
    { transform: `translate(${dx * 0.08}px, ${dy * 0.08 - 30}px) scale(0.86) rotate(-4deg)`, opacity: 1, offset: 0.22 },
    { transform: `translate(${dx * 0.6}px, ${dy * 0.45 - 40}px) scale(${0.25 + end}) rotate(8deg)`, opacity: 1, offset: 0.65 },
    { transform: `translate(${dx}px, ${dy}px) scale(${end}) rotate(14deg)`, opacity: 0.35, offset: 1 },
  ], { duration: 820, easing: "cubic-bezier(.45,.05,.35,1)", fill: "forwards" });
  anim.onfinish = () => { fly.remove(); bump(); burst(target); };
  setTimeout(() => fly.remove(), 2000);
}

function burst(el) {
  if (reduceMotion.matches || !Element.prototype.animate) return;
  const r = el.querySelector(".disc").getBoundingClientRect();
  const cx = r.left + r.width / 2;
  const cy = r.top + r.height / 2;
  const glyphs = ["⭐", "✨", "💛", "⭐", "✨", "🌟", "⭐", "✨"];
  glyphs.forEach((g, i) => {
    const s = document.createElement("span");
    s.className = "spark";
    s.textContent = g;
    s.style.left = `${cx}px`;
    s.style.top = `${cy}px`;
    document.body.append(s);
    const a = (i / glyphs.length) * Math.PI * 2 + Math.random() * 0.4;
    const d = 46 + Math.random() * 26;
    s.animate([
      { transform: "translate(-50%, -50%) scale(0.3)", opacity: 0 },
      { transform: "translate(-50%, -50%) scale(1.1)", opacity: 1, offset: 0.25 },
      { transform: `translate(calc(-50% + ${Math.cos(a) * d}px), calc(-50% + ${Math.sin(a) * d}px)) scale(0.7) rotate(${a}rad)`, opacity: 0 },
    ], { duration: 700 + Math.random() * 200, easing: "cubic-bezier(.2,.7,.3,1)" }).onfinish = () => s.remove();
  });
}

// Gallery view
$("#open-gallery").addEventListener("click", () => { hideHint(); gallery.open(); });
net.addEventListener("gallery", () => { if (gallery.isOpen) gallery.update(); });
gallery.addEventListener("continue", (e) => {
  const art = e.detail.artwork;
  confirmThen(() => {
    continuing = true;
    if (!net.continueArtwork(art.id)) {
      continuing = false;
      showHint("Not connected yet. Try again in a moment.");
    }
  }, "Keep drawing that picture?", "Yes, keep drawing");
});
net.addEventListener("continue", async (e) => {
  const m = e.detail;
  continuing = false;
  if (m.topic) toolbar.select("topics", m.topic);
  if (m.style) toolbar.select("styles", m.style);
  try {
    await loadSketch(m.sketch, undefined, () => net.continuing(m.id));
  } catch {
    // Nothing changed: the canvas and the server still have the previous drawing.
    showHint("Could not open that picture. Try again!");
    return;
  }
  drewSomething = true;
  unsaved = false;
  gallery.close();
});

net.start();

// Handy for debugging and end-to-end tests.
window.dreamer = { canvas, toolbar, view, net, gallery };
