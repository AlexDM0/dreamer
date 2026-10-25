// Dreamer kid page: wires the toolbar, drawing canvas, view and network together.
import { DrawingCanvas } from "./canvas.js";
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
  if (!e.target.closest(".pick-row")) e.preventDefault();
}, { passive: false });

// ---- modules ----
const canvas = new DrawingCanvas($("#sketch"));
const toolbar = new Toolbar();
const view = new View($("#stage-wrap"), $("#stage"), $("#picture"));
const net = new Net(canvas);

canvas.setColor(toolbar.color);
canvas.setWidth(toolbar.width);
toolbar.setMode(view.mode);

// Toolbar -> canvas / net / view
toolbar.addEventListener("color", (e) => canvas.setColor(e.detail.color));
toolbar.addEventListener("width", (e) => canvas.setWidth(e.detail.width));
toolbar.addEventListener("clear", () => {
  canvas.clear();
  view.strokeStart(); // show the (blank) sketch right away in Auto
  view.strokeEnd();
  net.clear();
});
toolbar.addEventListener("mode", (e) => view.setMode(e.detail.mode));
// A new topic/style re-renders the current drawing (the socket keeps message order, so
// the server applies the selection before it sees this sketch).
toolbar.addEventListener("topic", (e) => { net.selectTopic(e.detail.id); net.sendNow(); });
toolbar.addEventListener("style", (e) => { net.selectStyle(e.detail.id); net.sendNow(); });

// Canvas -> view / net
canvas.addEventListener("strokestart", () => view.strokeStart());
canvas.addEventListener("strokeend", () => {
  view.strokeEnd();
  net.sendNow();
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
  showConn();
});
net.addEventListener("status", (e) => {
  backendState = e.detail.state;
  if (net.state === "open") toolbar.setConn(backendState, e.detail.msg);
});
net.addEventListener("ready", (e) => {
  const m = e.detail;
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

net.start();

// Handy for debugging and end-to-end tests.
window.dreamer = { canvas, toolbar, view, net };
