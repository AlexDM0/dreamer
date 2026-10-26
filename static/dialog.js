// Modal overlays (role dialog / alertdialog with aria-modal): opening one moves the focus
// into it, closing it gives the focus back to where it was (usually the button that
// opened it), so keyboard and VoiceOver users are not left behind the overlay.

const FOCUSABLE = "button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex='-1'])";
const returnFocus = new WeakMap(); // overlay -> element focused before it opened

/** Show `overlay` and focus `focusTarget` (default: its first focusable element). */
export function showDialog(overlay, focusTarget = null) {
  if (!overlay.hidden) return;
  returnFocus.set(overlay, document.activeElement);
  overlay.hidden = false;
  const target = focusTarget || overlay.querySelector(FOCUSABLE);
  if (target) target.focus({ preventScroll: true });
}

/** Hide `overlay` and put the focus back. */
export function hideDialog(overlay) {
  if (overlay.hidden) return;
  overlay.hidden = true;
  const previous = returnFocus.get(overlay);
  returnFocus.delete(overlay);
  if (previous && previous !== document.body && previous.isConnected
      && typeof previous.focus === "function") {
    previous.focus({ preventScroll: true });
  }
}
