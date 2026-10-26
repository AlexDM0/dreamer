// PIN pad helpers shared by the parent gate on the kid page and the admin login.

export const PIN_MAX = 12;

/** One dot per digit typed (at least four, so the pad never looks empty). */
export function renderPinDots(dots, length) {
  const n = Math.max(4, length);
  dots.replaceChildren(...Array.from({ length: n }, (_, i) => {
    const s = document.createElement("span");
    if (i < length) s.className = "on";
    return s;
  }));
  dots.setAttribute("aria-label", `${length} digits entered`);
}

/** Shake the dots after a wrong PIN. */
export function shakePinDots(dots) {
  dots.classList.remove("shake");
  void dots.offsetWidth; // restart the CSS animation
  dots.classList.add("shake");
}
