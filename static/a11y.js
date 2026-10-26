// Keyboard and screen-reader help for the kid page: dialogs take and give back focus and
// keep Tab inside, and arrow keys move between the buttons of a radio group.

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])';

function visible(el) {
  return !el.closest("[hidden]") && el.getClientRects().length > 0;
}

function focusables(root) {
  return [...root.querySelectorAll(FOCUSABLE)].filter(visible);
}

function focus(el) {
  try { el.focus({ preventScroll: true }); } catch { /* old Safari */ }
}

/** Every `.overlay` dialog: on open, focus moves into it (unless its own code already put
 * it there); Tab cycles inside it; on close, focus goes back to where it was. */
export function manageDialogs(overlays) {
  const openers = new Map();
  const observer = new MutationObserver((records) => {
    for (const { target: overlay } of records) {
      if (!overlay.hidden && !openers.has(overlay)) {
        openers.set(overlay, document.activeElement);
        // After the opener's own code ran (it may focus the selected item itself).
        setTimeout(() => {
          if (overlay.hidden || overlay.contains(document.activeElement)) return;
          const first = focusables(overlay)[0];
          if (first) focus(first);
        }, 0);
      } else if (overlay.hidden && openers.has(overlay)) {
        const opener = openers.get(overlay);
        openers.delete(overlay);
        const lost = !document.activeElement || document.activeElement === document.body
          || overlay.contains(document.activeElement);
        if (lost && opener && opener.isConnected && visible(opener)) focus(opener);
      }
    }
  });
  for (const overlay of overlays) observer.observe(overlay, { attributes: true, attributeFilter: ["hidden"] });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    const open = [...overlays].filter((o) => !o.hidden);
    const top = open[open.length - 1];
    if (!top) return;
    const items = focusables(top);
    if (!items.length) { e.preventDefault(); return; }
    const at = items.indexOf(document.activeElement);
    const next = e.shiftKey ? (at <= 0 ? items.length - 1 : at - 1)
                            : (at < 0 || at === items.length - 1 ? 0 : at + 1);
    e.preventDefault();
    focus(items[next]);
  });
}

/** Arrow keys (and Home/End) move focus between the radios of a `[role=radiogroup]`;
 * Space/Enter picks, as on any button. */
export function radioArrows() {
  const STEP = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
  document.addEventListener("keydown", (e) => {
    const radio = e.target instanceof Element && e.target.closest('[role="radio"]');
    const group = radio && radio.closest('[role="radiogroup"]');
    if (!group || e.metaKey || e.ctrlKey || e.altKey) return;
    const radios = [...group.querySelectorAll('[role="radio"]')].filter(visible);
    const at = radios.indexOf(radio);
    let next;
    if (e.key in STEP) next = (at + STEP[e.key] + radios.length) % radios.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = radios.length - 1;
    else return;
    e.preventDefault();
    focus(radios[next]);
  });
}
