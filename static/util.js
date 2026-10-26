// Small helpers shared by the kid page's modules.

/** Load an image from a URL or a Blob. A Blob's object URL is revoked on failure; on
 * success the caller owns it (`URL.revokeObjectURL(img.src)` when done). */
export function loadImage(source) {
  const blob = source instanceof Blob;
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => {
      if (blob) URL.revokeObjectURL(img.src);
      reject(new Error(blob ? "bad image" : `could not load ${source}`));
    };
    img.src = blob ? URL.createObjectURL(source) : source;
  });
}

/** Rough perceived brightness check, to outline very light colours. */
export function isLight(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return false;
  const n = parseInt(m[1], 16);
  const r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return 0.299 * r + 0.587 * g + 0.114 * b > 225;
}
