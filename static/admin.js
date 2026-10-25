// Dreamer admin page: PIN login, live view over /ws/admin, and style/topic management.
// Plain ES module, no dependencies. Every API call goes through api(), which sends the
// user back to the PIN pad on a 401 (expired or missing cookie).

const $ = (id) => document.getElementById(id);

const PICTURE = 0x02;          // protocol.py: server->client JPEG frame
const HEADER_BYTES = 5;        // type (1) + seq (uint32 BE)
const PIN_MAX = 12;
const KIND_LABEL = { preset: "Built-in", custom: "Custom", image: "Picture" };

let state = null;              // last GET /api/admin/state
let pin = "";
let ws = null;
let wsRetry = 0;
let wsTimer = 0;
let wantSocket = false;        // false while logged out, so the socket stops reconnecting
let liveUrl = null;            // current object URL shown in the live <img>
let openEditor = null;         // "styles:<id>" of the row whose editor is open

// ----- HTTP -------------------------------------------------------------------------

class AuthError extends Error {}

/** fetch wrapper: JSON in/out, readable error messages, 401 -> PIN pad. */
async function api(method, path, body) {
  const opts = { method, credentials: "same-origin", headers: {} };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Can't reach Dreamer. Is the computer on?");
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty or non-JSON body */ }
  if (res.status === 401 && path !== "/api/admin/login") {
    showLogin("Please log in again.");
    throw new AuthError("Not logged in");
  }
  if (!res.ok) throw new Error(errorText(data, res.status));
  return data;
}

/** FastAPI errors are {"detail": "..."} or {"detail": [{msg}, ...]} for validation. */
function errorText(data, status) {
  const d = data && data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d.length) return d.map((e) => e.msg || String(e)).join(", ");
  return `Something went wrong (${status})`;
}

/** Run an action, reporting errors in a toast. Auth errors are already handled. */
async function attempt(fn, okMsg) {
  try {
    const out = await fn();
    if (okMsg) toast(okMsg);
    return out;
  } catch (err) {
    if (!(err instanceof AuthError)) toast(err.message, true);
    return undefined;
  }
}

let toastTimer = 0;
function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), isError ? 4000 : 2200);
}

// ----- Login ------------------------------------------------------------------------

function showLogin(msg = "") {
  wantSocket = false;
  closeSocket();
  pin = "";
  renderPin();
  $("pin-msg").textContent = msg;
  $("admin").hidden = true;
  $("login").hidden = false;
}

function renderPin() {
  const dots = $("pin-dots");
  const n = Math.max(4, pin.length);
  dots.replaceChildren(...Array.from({ length: n }, (_, i) => {
    const s = document.createElement("span");
    if (i < pin.length) s.className = "on";
    return s;
  }));
  dots.setAttribute("aria-label", `${pin.length} digits entered`);
}

function pressKey(k) {
  if (k === "back") pin = pin.slice(0, -1);
  else if (k === "ok") return submitPin();
  else if (/^\d$/.test(k) && pin.length < PIN_MAX) pin += k;
  $("pin-msg").textContent = "";
  renderPin();
}

let loggingIn = false;
async function submitPin() {
  if (!pin || loggingIn) return;
  loggingIn = true;
  try {
    await api("POST", "/api/admin/login", { pin });
    pin = "";
    await enterAdmin();
  } catch (err) {
    pin = "";
    renderPin();
    $("pin-msg").textContent = err.message;
    const dots = $("pin-dots");
    dots.classList.remove("shake");
    void dots.offsetWidth; // restart the CSS animation
    dots.classList.add("shake");
  } finally {
    loggingIn = false;
  }
}

$("pin-pad").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-k]");
  if (b) pressKey(b.dataset.k);
});

// Physical keyboards (desktop, or a phone with a keyboard attached).
document.addEventListener("keydown", (e) => {
  if ($("login").hidden || e.metaKey || e.ctrlKey || e.altKey) return;
  if (/^\d$/.test(e.key)) pressKey(e.key);
  else if (e.key === "Backspace") pressKey("back");
  else if (e.key === "Enter") pressKey("ok");
  else return;
  e.preventDefault();
});

// ----- Admin view -------------------------------------------------------------------

async function enterAdmin() {
  await loadState();
  attempt(loadParams); // not fatal: the rest of the page works without it
  attempt(() => loadGallery());
  $("login").hidden = true;
  $("admin").hidden = false;
  wantSocket = true;
  openSocket();
}

async function loadState() {
  state = await api("GET", "/api/admin/state");
  render();
}

/** Re-fetch after a mutation so the page always mirrors the server's truth. */
const refresh = () => attempt(loadState);

function render() {
  if (!state) return;
  renderSelects();
  renderList("styles", $("styles-list"));
  renderList("topics", $("topics-list"));
  renderStats(state.stats || {}, state.backend);
}

function renderSelects() {
  fillSelect($("sel-topic"), state.topics, state.topic);
  fillSelect($("sel-style"), state.styles, state.style);
  if (document.activeElement !== $("sel-view")) $("sel-view").value = state.view_mode || "auto";
}

/** Only visible items can be picked: the server resolves hidden ones to a fallback. */
function fillSelect(sel, items, current) {
  if (document.activeElement === sel) return; // don't yank an open picker
  sel.replaceChildren(...items.filter((it) => !it.hidden).map((it) => {
    const o = new Option(`${it.emoji} ${it.label}`, it.id);
    o.selected = it.id === current;
    return o;
  }));
}

for (const [id, key] of [["sel-topic", "topic"], ["sel-style", "style"], ["sel-view", "view_mode"]]) {
  $(id).addEventListener("change", async (e) => {
    const res = await attempt(() => api("POST", "/api/admin/select", { [key]: e.target.value }),
                              "Sent to the iPad");
    if (res) Object.assign(state, { topic: res.topic, style: res.style, view_mode: res.view_mode });
    e.target.blur();
    renderSelects();
  });
}

// ----- Picture controls (live diffusion params) -------------------------------------

const T_MAX = 49;              // highest StreamDiffusion t_index
const PARAMS_DEBOUNCE_MS = 150;
let params = null;             // last params the server reported
let pendingParams = {};        // changes not yet sent
const pendingInputs = {};      // ids edited since the last send (kept as typed)
let paramsTimer = 0;

/** creativity 0..100 + steps 1..8 -> strictly increasing t_index_list.
 * Lower first index = more noise = more creative. Later steps spread towards the end,
 * so the default (creativity 88, 2 steps) gives [6, 26] like config.example.yaml. */
function tIndexList(creativity, steps) {
  const first = Math.min(Math.round(T_MAX * (1 - creativity / 100)), T_MAX - (steps - 1));
  if (steps === 1) return [first];
  const last = Math.min(T_MAX, first + 20 * (steps - 1));
  const gap = Math.max(last - first, steps - 1) / (steps - 1); // >= 1 keeps it increasing
  return Array.from({ length: steps }, (_, i) => Math.round(first + i * gap));
}
const creativityOf = (t) => Math.round((1 - t[0] / T_MAX) * 100);

// [input id, output id, read(params) -> control value, format]
const PARAM_VIEWS = [
  ["p-creativity", "o-creativity", (p) => creativityOf(p.t_index_list), (v) => `${v}%`],
  ["p-steps", "o-steps", (p) => p.t_index_list.length, String],
  ["p-cn-scale", "o-cn-scale", (p) => p.controlnet_scale, (v) => Number(v).toFixed(2)],
  ["p-safety-th", "o-safety-th", (p) => p.safety_threshold, (v) => Number(v).toFixed(2)],
  ["p-line-th", "o-line-th", (p) => p.line_threshold, String],
  ["p-dilate", "o-dilate", (p) => p.line_dilate, (v) => `${v} px`],
  ["p-jpeg", "o-jpeg", (p) => p.jpeg_quality, String],
  ["p-seed", null, (p) => p.seed, String],
  ["p-polarity", null, (p) => p.line_polarity, String],
  ["p-init", null, (p) => p.init_mode, String],
  ["p-init-blur", "o-init-blur", (p) => p.init_blur, (v) => `${v} px`],
];

async function loadParams() {
  applyParams(await api("GET", "/api/admin/params"));
}

/** Show params from the server, without yanking a control the parent is touching. */
function applyParams(res) {
  if (!res || !res.params) return;
  params = res.params;
  for (const [id, out, read, fmt] of PARAM_VIEWS) {
    const input = $(id);
    const value = read(params);
    if (document.activeElement !== input && !(id in pendingInputs)) input.value = value;
    if (out) $(out).textContent = fmt(input.value);
  }
  $("p-cn-on").checked = params.controlnet_enabled;
  $("p-cn-scale").disabled = !params.controlnet_enabled;
  $("p-init-blur").disabled = params.init_mode !== "colors";
  $("p-safety").checked = params.safety_checker;
  $("safety-off").hidden = params.safety_checker;
}

function queueParams(changes, inputId) {
  Object.assign(pendingParams, changes);
  if (inputId) pendingInputs[inputId] = true;
  clearTimeout(paramsTimer);
  paramsTimer = setTimeout(sendParams, PARAMS_DEBOUNCE_MS);
}

async function sendParams() {
  const body = pendingParams;
  pendingParams = {};
  for (const k of Object.keys(pendingInputs)) delete pendingInputs[k];
  if (!Object.keys(body).length) return;
  const res = await attempt(() => api("PATCH", "/api/admin/params", body));
  if (res) applyParams(res);
  else attempt(loadParams); // show what the server really has
}

function onParamInput(id, out, fmt, toChanges) {
  const input = $(id);
  input.addEventListener("input", () => {
    if (out) $(out).textContent = fmt(input.value);
    const changes = toChanges(input.value);
    if (changes) queueParams(changes, id);
  });
}

const tIndexNow = () => tIndexList(Number($("p-creativity").value), Number($("p-steps").value));
onParamInput("p-creativity", "o-creativity", (v) => `${v}%`, () => ({ t_index_list: tIndexNow() }));
onParamInput("p-steps", "o-steps", String, () => ({ t_index_list: tIndexNow() }));
onParamInput("p-cn-scale", "o-cn-scale", (v) => Number(v).toFixed(2), (v) => ({ controlnet_scale: Number(v) }));
onParamInput("p-safety-th", "o-safety-th", (v) => Number(v).toFixed(2), (v) => ({ safety_threshold: Number(v) }));
onParamInput("p-line-th", "o-line-th", String, (v) => ({ line_threshold: Number(v) }));
onParamInput("p-dilate", "o-dilate", (v) => `${v} px`, (v) => ({ line_dilate: Number(v) }));
onParamInput("p-jpeg", "o-jpeg", String, (v) => ({ jpeg_quality: Number(v) }));
onParamInput("p-seed", null, String, (v) => {
  const n = Number(v);
  return v !== "" && Number.isInteger(n) && n >= 0 ? { seed: n } : null;
});
$("p-polarity").addEventListener("change", (e) => queueParams({ line_polarity: e.target.value }));
$("p-init").addEventListener("change", (e) => {
  $("p-init-blur").disabled = e.target.value !== "colors";
  queueParams({ init_mode: e.target.value });
});
onParamInput("p-init-blur", "o-init-blur", (v) => `${v} px`, (v) => ({ init_blur: Number(v) }));
$("p-cn-on").addEventListener("change", (e) => {
  $("p-cn-scale").disabled = !e.target.checked;
  queueParams({ controlnet_enabled: e.target.checked });
});
$("p-safety").addEventListener("change", (e) => {
  $("safety-off").hidden = e.target.checked;
  queueParams({ safety_checker: e.target.checked });
});
$("p-seed-new").addEventListener("click", () => {
  const seed = Math.floor(Math.random() * 2147483647);
  $("p-seed").value = seed;
  queueParams({ seed });
});
$("params-reset").addEventListener("click", async () => {
  clearTimeout(paramsTimer);
  pendingParams = {};
  const res = await attempt(() => api("POST", "/api/admin/params/reset"), "Back to defaults");
  if (res) applyParams(res);
});

// ----- Lists (styles + topics) ------------------------------------------------------

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c != null));
  return node;
}

function iconButton(label, svgPath, onClick, extra = "") {
  const b = el("button", { type: "button", className: `icon ${extra}`.trim(), title: label });
  b.setAttribute("aria-label", label);
  b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${svgPath}"/></svg>`;
  b.addEventListener("click", onClick);
  return b;
}

const ICON = {
  up: "M12 19V5M6 11l6-6 6 6",
  down: "M12 5v14M6 13l6 6 6-6",
  edit: "M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4",
  trash: "M5 7h14M10 4h4M7 7l1 13h8l1-13",
};

function renderList(kind, ul) {
  const items = state[kind];
  ul.replaceChildren(...items.map((item, i) => {
    const li = el("li", { className: `item${item.hidden ? " is-hidden" : ""}` });
    const name = el("div", { className: "name" },
      el("span", { className: "badge", textContent: item.emoji }),
      el("span", { className: "text" },
        el("strong", { textContent: item.label }),
        kind === "styles" ? el("small", { textContent: KIND_LABEL[item.kind] || item.kind }) : null));

    const sw = el("input", { type: "checkbox", checked: !item.hidden, className: "switch" });
    sw.setAttribute("aria-label", `Show ${item.label} on the iPad`);
    sw.addEventListener("change", () => setHidden(kind, item, !sw.checked));

    const up = iconButton(`Move ${item.label} up`, ICON.up, () => move(kind, i, -1));
    const down = iconButton(`Move ${item.label} down`, ICON.down, () => move(kind, i, 1));
    up.disabled = i === 0;
    down.disabled = i === items.length - 1;

    const actions = el("div", { className: "actions" }, up, down);
    if (kind === "styles") {
      const key = `${kind}:${item.id}`;
      const edit = iconButton(`Edit ${item.label}`, ICON.edit, () => {
        openEditor = openEditor === key ? null : key;
        renderList(kind, ul);
      }, openEditor === key ? "active" : "");
      actions.append(edit); // delete lives inside the editor: harder to hit by accident
    }
    actions.append(sw);
    li.append(name, actions);
    if (openEditor === `${kind}:${item.id}`) li.append(editor(kind, item));
    return li;
  }));
}

/** Inline edit form: label + emoji for all styles, prompt only for custom ones. */
function editor(kind, item) {
  const form = el("form", { className: "form editor", autocomplete: "off" });
  const emoji = el("input", { name: "emoji", value: item.emoji, maxLength: 16 });
  const label = el("input", { name: "label", value: item.label, maxLength: 40, required: true });
  form.append(el("div", { className: "row" },
    el("label", { className: "field emoji" }, el("span", { textContent: "Emoji" }), emoji),
    el("label", { className: "field grow" }, el("span", { textContent: "Name" }), label)));
  let prompt = null;
  if (item.kind === "custom") {
    prompt = el("textarea", { name: "prompt", value: item.prompt, maxLength: 120, rows: 2, required: true });
    form.append(el("label", { className: "field" },
      el("span", { textContent: "What it should look like" }), prompt));
  }
  const cancel = el("button", { type: "button", className: "btn ghost", textContent: "Cancel" });
  cancel.addEventListener("click", () => { openEditor = null; render(); });
  form.append(el("div", { className: "row buttons" }, cancel,
    el("button", { type: "submit", className: "btn primary", textContent: "Save" })));
  if (item.kind !== "preset") { // custom and picture styles can be deleted
    const del = el("button", { type: "button", className: "btn danger" });
    del.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${ICON.trash}"/></svg>`;
    del.append(`Delete ${item.label}`);
    del.addEventListener("click", () => removeStyle(item));
    form.append(del);
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { label: label.value, emoji: emoji.value };
    if (prompt) body.prompt = prompt.value;
    const ok = await attempt(() => api("PATCH", itemPath(kind, item.id), body), "Saved");
    if (ok) { openEditor = null; await refresh(); }
  });
  queueMicrotask(() => label.focus({ preventScroll: true }));
  return form;
}

const itemPath = (kind, id) => `/api/admin/items/${kind}/${encodeURIComponent(id)}`;

async function setHidden(kind, item, hidden) {
  await attempt(() => api("PATCH", itemPath(kind, item.id), { hidden }),
                hidden ? `${item.label} is hidden` : `${item.label} is back`);
  await refresh(); // either way: on failure the re-render resets the switch
}

async function move(kind, index, delta) {
  const items = state[kind].slice();
  const j = index + delta;
  if (j < 0 || j >= items.length) return;
  [items[index], items[j]] = [items[j], items[index]];
  state[kind] = items; // optimistic: feels instant on a phone
  render();
  const ok = await attempt(() => api("POST", "/api/admin/order",
                                     { kind, ids: items.map((it) => it.id) }));
  if (!ok) await refresh();
}

async function removeStyle(item) {
  if (!confirm(`Delete the style "${item.label}"?`)) return;
  const ok = await attempt(() => api("DELETE", itemPath("styles", item.id)), "Deleted");
  if (ok) { openEditor = null; await refresh(); }
}

// ----- Add a style ------------------------------------------------------------------

$("add-style-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = { label: f.label.value, emoji: f.emoji.value, prompt: f.prompt.value };
  const res = await attempt(() => api("POST", "/api/admin/styles", body), "Style added");
  if (res) {
    f.reset();
    $("add-style").open = false;
    await refresh();
  }
});

// ----- Style picture upload ---------------------------------------------------------

const picForm = $("pic-form");
let previewUrl = null;

picForm.file.addEventListener("change", () => {
  const file = picForm.file.files[0];
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : null;
  $("file-preview").hidden = !file;
  if (file) $("file-preview").src = previewUrl;
  $("file-label").textContent = file ? file.name : "Choose a picture";
  $("file-box").classList.toggle("has-file", !!file);
});

picForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = picForm.querySelector("button[type=submit]");
  const note = $("pic-note");
  note.hidden = true;
  btn.disabled = true;
  btn.textContent = "Checking the picture…";
  const data = new FormData();
  data.append("file", picForm.file.files[0]);
  data.append("label", picForm.label.value);
  data.append("emoji", picForm.emoji.value);
  const res = await attempt(() => api("POST", "/api/admin/style-image", data), "Picture style added");
  btn.disabled = false;
  btn.textContent = "Upload";
  if (!res) return;
  if (res.note) {
    note.textContent = res.note;
    note.hidden = false;
  }
  picForm.reset();
  picForm.file.dispatchEvent(new Event("change"));
  await refresh();
});

// ----- Gallery (saved artworks) -----------------------------------------------------

const ART_PAGE = 24;
let artTab = "saved";          // "saved" | "trash"
let artItems = [];
let artMore = false;
let artOpen = null;            // artwork shown in the sheet
let artGen = 0;                // newest load wins

async function loadGallery(append = false) {
  const gen = ++artGen;
  const offset = append ? artItems.length : 0;
  const trash = artTab === "trash" ? 1 : 0;
  const res = await api("GET", `/api/admin/artworks?trash=${trash}&limit=${ART_PAGE}&offset=${offset}`);
  if (gen !== artGen) return;
  artItems = append ? artItems.concat(res.artworks.filter((a) => !artItems.some((b) => b.id === a.id)))
                    : res.artworks;
  artMore = res.more;
  $("count-saved").textContent = res.total ? `(${res.total})` : "";
  $("count-trash").textContent = res.trash ? `(${res.trash})` : "";
  renderGallery(res.trash);
  if (artOpen) {
    const fresh = artItems.find((a) => a.id === artOpen.id);
    if (fresh) showArt(fresh); else closeArt();
  }
}

function renderGallery(trashCount) {
  const grid = $("art-grid");
  grid.replaceChildren(...artItems.map((art) => {
    const b = el("button", { type: "button", className: "art-thumb" });
    b.setAttribute("aria-label", art.title || `Picture from ${niceDate(art.created_at)}`);
    b.append(el("img", { src: art.thumb, alt: "", loading: "lazy" }));
    b.append(el("span", { className: "art-cap", textContent: art.title || niceDate(art.created_at) }));
    b.addEventListener("click", () => showArt(art));
    return b;
  }));
  const empty = $("art-empty");
  empty.hidden = artItems.length > 0;
  empty.textContent = artTab === "trash" ? "The trash is empty." : "No saved pictures yet.";
  $("art-more").hidden = !artMore;
  $("art-purge").hidden = artTab !== "trash" || !trashCount;
  for (const [id, tab] of [["tab-saved", "saved"], ["tab-trash", "trash"]]) {
    $(id).classList.toggle("active", artTab === tab);
    $(id).setAttribute("aria-selected", String(artTab === tab));
  }
}

function niceDate(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function itemLabel(kind, id) {
  const item = state && state[kind] && state[kind].find((it) => it.id === id);
  return item ? `${item.emoji} ${item.label}` : id;
}

function withParam(url, key, value) {
  return url + (url.includes("?") ? "&" : "?") + `${key}=${value}`;
}

function showArt(art) {
  artOpen = art;
  const trashed = !!art.deleted_at;
  $("art-title-h").textContent = art.title || "Picture";
  $("art-picture").src = art.picture;
  $("art-sketch").src = art.sketch;
  $("art-meta").textContent = `Saved ${niceDate(art.created_at)}`
    + (art.updated_at !== art.created_at ? ` · changed ${niceDate(art.updated_at)}` : "")
    + (art.topic ? ` · ${itemLabel("topics", art.topic)}` : "")
    + (art.style ? ` / ${itemLabel("styles", art.style)}` : "");
  const input = $("art-title");
  if (document.activeElement !== input) input.value = art.title || "";
  $("art-rename").hidden = trashed;
  $("art-dl-picture").href = withParam(art.picture, "download", 1);
  $("art-dl-sketch").href = withParam(art.sketch, "download", 1);
  $("art-delete").hidden = trashed;
  $("art-restore").hidden = !trashed;
  $("art-sheet").hidden = false;
}

function closeArt() {
  artOpen = null;
  $("art-sheet").hidden = true;
}

const artPath = (id) => `/api/admin/artworks/${encodeURIComponent(id)}`;

$("tab-saved").addEventListener("click", () => { artTab = "saved"; attempt(() => loadGallery()); });
$("tab-trash").addEventListener("click", () => { artTab = "trash"; attempt(() => loadGallery()); });
$("art-more").addEventListener("click", () => attempt(() => loadGallery(true)));
$("art-close").addEventListener("click", closeArt);
$("art-sheet").addEventListener("click", (e) => { if (e.target === e.currentTarget) closeArt(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && artOpen) closeArt(); });

$("art-rename").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!artOpen) return;
  const input = $("art-title");
  const res = await attempt(() => api("PATCH", artPath(artOpen.id), { title: input.value }), "Name saved");
  if (res) { input.blur(); await attempt(() => loadGallery()); }
});

$("art-delete").addEventListener("click", async () => {
  if (!artOpen) return;
  const ok = await attempt(() => api("DELETE", artPath(artOpen.id)), "Moved to the trash");
  if (ok) { closeArt(); await attempt(() => loadGallery()); }
});

$("art-restore").addEventListener("click", async () => {
  if (!artOpen) return;
  const ok = await attempt(() => api("POST", `${artPath(artOpen.id)}/restore`), "Back in the gallery");
  if (ok) { closeArt(); await attempt(() => loadGallery()); }
});

$("art-purge").addEventListener("click", async () => {
  if (!confirm("Delete every picture in the trash for good? This cannot be undone.")) return;
  const res = await attempt(() => api("POST", "/api/admin/artworks/purge"));
  if (res) {
    toast(res.purged === 1 ? "1 picture deleted" : `${res.purged} pictures deleted`);
    await attempt(() => loadGallery());
  }
});

// ----- Live view over /ws/admin -----------------------------------------------------

function openSocket() {
  clearTimeout(wsTimer);
  if (!wantSocket || ws) return;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const sock = new WebSocket(`${proto}//${location.host}/ws/admin`);
  sock.binaryType = "arraybuffer";
  let opened = false;
  ws = sock;
  setLive("connecting");

  sock.onopen = () => { opened = true; wsRetry = 0; };
  sock.onmessage = (e) => {
    if (typeof e.data === "string") onJson(e.data);
    else onBinary(e.data);
  };
  sock.onclose = () => {
    if (ws !== sock) return;
    ws = null;
    if (!wantSocket) return;
    setLive("offline");
    // A socket refused before opening is usually an expired cookie (the server closes
    // with 1008 before accept). Ask the API, which shows the PIN pad on a 401.
    if (!opened) attempt(loadState);
    const delay = Math.min(5000, 500 * 2 ** wsRetry++);
    wsTimer = setTimeout(openSocket, delay);
  };
}

function closeSocket() {
  clearTimeout(wsTimer);
  if (ws) {
    const s = ws;
    ws = null;
    s.close();
  }
}

function onBinary(buf) {
  if (buf.byteLength <= HEADER_BYTES) return;
  if (new DataView(buf).getUint8(0) !== PICTURE) return;
  const url = URL.createObjectURL(new Blob([buf.slice(HEADER_BYTES)], { type: "image/jpeg" }));
  const img = $("live-img");
  const prev = liveUrl;
  liveUrl = url;
  img.src = url;
  // The previous URL is no longer the src (a pending load of it is aborted), so free it
  // now; waiting for onload leaks whenever frames arrive faster than they load.
  if (prev) URL.revokeObjectURL(prev);
  img.hidden = false;
  $("live-empty").hidden = true;
}

function onJson(text) {
  let msg;
  try { msg = JSON.parse(text); } catch { return; }
  if (msg.t === "params") {
    applyParams(msg); // changed on another phone, or reverted after an apply error
  } else if (msg.t === "gallery") {
    attempt(() => loadGallery()); // the kid saved, or another phone changed something
  } else if (msg.t === "status") {
    setLive(msg.state, msg.msg);
  } else if (msg.t === "stats") {
    renderStats(msg, msg.backend);
    if (msg.state) setLive(msg.state, msg.state === "error" ? undefined : "");
    // Stats carry the current selection, so changes made elsewhere show up here too.
    if (state && (msg.topic !== state.topic || msg.style !== state.style
                  || msg.view_mode !== state.view_mode)) {
      Object.assign(state, { topic: msg.topic, style: msg.style, view_mode: msg.view_mode });
      renderSelects();
    }
  }
}

const LIVE_TEXT = { ready: "Ready", busy: "Warming up", error: "Problem", connecting: "Connecting", offline: "Offline" };

/** msg undefined = leave the current message alone; "" = clear it. */
function setLive(s, msg) {
  const pill = $("live-state");
  pill.dataset.state = s;
  pill.querySelector("b").textContent = LIVE_TEXT[s] || s;
  if (msg === undefined) return;
  const m = $("st-msg");
  m.textContent = msg;
  m.hidden = !msg || s !== "error";
}

function renderStats(stats, backend) {
  $("st-backend").textContent = backend || stats.backend || "-";
  $("st-fps").textContent = stats.fps ? `${Number(stats.fps).toFixed(1)} fps` : "-";
  $("st-ms").textContent = stats.last_ms ? `${Math.round(stats.last_ms)} ms` : "-";
}

// ----- Log out ----------------------------------------------------------------------

$("logout").addEventListener("click", async () => {
  await attempt(() => api("POST", "/api/admin/logout"));
  showLogin();
});

// ----- Boot -------------------------------------------------------------------------

(async () => {
  try {
    await enterAdmin();
  } catch (err) {
    if (!(err instanceof AuthError)) showLogin(err.message);
  }
})();
