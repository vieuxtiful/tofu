// 🍢 ToFU — projective corner maths
//
// The Translate preview reproduces what scribe.render() will draw. For the
// affine transform it does that with CSS skewX/skewY/scale, which is exact.
// A quad is not affine — skew keeps opposite edges parallel while a
// photographed plane makes them converge — so the preview needs the one CSS
// primitive that can express a homography: matrix3d().
//
// Pure by design: no React, no DOM, no canvas. Same reason doppelganger.ts
// and fontCatalog.ts are — the maths is the part worth testing, and testing
// it should not require a browser.

import { BBox } from "./api";

/** Corner order everywhere in ToFU: top-left, top-right, bottom-right, bottom-left. */
export type Quad = [number, number][];

/** Corners at the bounding box itself: no perspective. */
export const IDENTITY_QUAD: Quad = [[0, 0], [1, 0], [1, 1], [0, 1]];

const EPS = 1e-4;

/** Four [x, y] pairs, or null when absent or malformed.
 *
 * A manifest is user-editable and round-trips through JSON, so a bad quad
 * has to mean "no perspective" rather than throwing inside a render. */
export function parseQuad(value: unknown): Quad | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const corners: [number, number][] = [];
  for (const point of value) {
    if (!Array.isArray(point) || point.length < 2) return null;
    const x = Number(point[0]);
    const y = Number(point[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    corners.push([x, y]);
  }
  return corners as Quad;
}

export function isIdentityQuad(quad: Quad | null | undefined): boolean {
  if (!quad) return true;
  return quad.every(
    ([x, y], i) => Math.abs(x - IDENTITY_QUAD[i][0]) < EPS
      && Math.abs(y - IDENTITY_QUAD[i][1]) < EPS,
  );
}

/** Bbox-relative corners → pixel space.
 *
 * Normalised storage is what lets a region move or resize without the
 * perspective detaching from it; this is the only place that has to know. */
export function denormaliseQuad(quad: Quad, bbox: BBox): Quad {
  return quad.map(([nx, ny]) => [
    bbox.x + nx * bbox.width,
    bbox.y + ny * bbox.height,
  ]) as Quad;
}

/** Convex, correctly wound, non-degenerate.
 *
 * A self-intersecting or collinear quad has no invertible homography. The
 * sign of the cross product at each vertex catches both: a convex polygon
 * turns the same way at every corner, a collinear triple turns not at all.
 * Mirrors `_quad_is_usable` in scribe.py so preview and render agree on
 * which quads are renderable. */
export function isUsableQuad(corners: Quad): boolean {
  const signs: boolean[] = [];
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = corners[i];
    const [bx, by] = corners[(i + 1) % 4];
    const [cx, cy] = corners[(i + 2) % 4];
    const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if (Math.abs(cross) < 1e-9) return false;
    signs.push(cross > 0);
  }
  return signs.every(Boolean) || signs.every((s) => !s);
}

/**
 * Seed a quad from a region's detected outline.
 *
 * Cicerone already fits a plane to each region it reads — `segmentation_
 * mask.polygon` is that outline — so the sensible starting point for a
 * perspective is the plane the detector already found, not a rectangle the
 * user then has to rebuild by hand.
 *
 * Corners are picked by the standard document-rectification extremes: the
 * sums x+y are smallest at the top-left and largest at the bottom-right,
 * and the differences x−y are largest at the top-right and smallest at the
 * bottom-left. That yields the four corners already in TL/TR/BR/BL order,
 * which is the order everything else here expects.
 *
 * Returns null when the polygon is too small, degenerate, or produces
 * duplicate corners — the caller keeps the identity rectangle.
 */
export function quadFromPolygon(
  polygon: number[][] | null | undefined, bbox: BBox,
): Quad | null {
  if (!Array.isArray(polygon) || polygon.length < 4) return null;
  if (bbox.width <= 0 || bbox.height <= 0) return null;
  const points: [number, number][] = [];
  for (const point of polygon) {
    if (!Array.isArray(point) || point.length < 2) continue;
    const x = Number(point[0]);
    const y = Number(point[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
  }
  if (points.length < 4) return null;

  const pick = (score: (p: [number, number]) => number) =>
    points.reduce((best, p) => (score(p) < score(best) ? p : best));
  const corners: Quad = [
    pick(([x, y]) => x + y),      // top-left
    pick(([x, y]) => y - x),      // top-right
    pick(([x, y]) => -(x + y)),   // bottom-right
    pick(([x, y]) => x - y),      // bottom-left
  ];

  // Four DISTINCT corners, or the outline was a line or a point.
  const seen = new Set(corners.map(([x, y]) => `${x},${y}`));
  if (seen.size < 4 || !isUsableQuad(corners)) return null;

  return corners.map(([x, y]) => [
    (x - bbox.x) / bbox.width,
    (y - bbox.y) / bbox.height,
  ]) as Quad;
}

/** Solve the 8 unknowns of the homography taking `from` to `to`.
 *
 * Gauss-Jordan on the standard 8x8 DLT system. Eight points, eight
 * equations, no external dependency — and small enough that partial
 * pivoting is the only numerical care required. */
function solveHomography(from: Quad, to: Quad): number[] | null {
  const a: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [sx, sy] = from[i];
    const [dx, dy] = to[i];
    a.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
    b.push(dx);
    a.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
    b.push(dy);
  }
  for (let col = 0; col < 8; col++) {
    let pivot = col;
    for (let row = col + 1; row < 8; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) return null;   // singular
    [a[col], a[pivot]] = [a[pivot], a[col]];
    [b[col], b[pivot]] = [b[pivot], b[col]];
    const divisor = a[col][col];
    for (let k = col; k < 8; k++) a[col][k] /= divisor;
    b[col] /= divisor;
    for (let row = 0; row < 8; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      if (!factor) continue;
      for (let k = col; k < 8; k++) a[row][k] -= factor * a[col][k];
      b[row] -= factor * b[col];
    }
  }
  return b.every((v) => Number.isFinite(v)) ? b : null;
}

/**
 * A CSS `matrix3d(...)` string placing the region's box onto `quad`.
 *
 * Applied with `transform-origin: 0 0` and the element positioned at the
 * bbox, because the solve maps the box's own local space (0,0)-(w,h) rather
 * than image space — that keeps it independent of where the region sits on
 * the canvas, exactly as scribe's transform is deliberately region-local.
 *
 * Returns null for an identity or unusable quad so the caller can fall back
 * to the existing skew/scale chain.
 */
export function quadToMatrix3d(quad: Quad | null | undefined, bbox: BBox): string | null {
  if (!quad || isIdentityQuad(quad)) return null;
  if (bbox.width <= 0 || bbox.height <= 0) return null;
  const source: Quad = [
    [0, 0], [bbox.width, 0], [bbox.width, bbox.height], [0, bbox.height],
  ];
  const destination = quad.map(([nx, ny]) => [
    nx * bbox.width, ny * bbox.height,
  ]) as Quad;
  if (!isUsableQuad(destination)) return null;
  const h = solveHomography(source, destination);
  if (!h) return null;
  const [a, b, c, d, e, f, g, i] = h;
  // CSS matrix3d is COLUMN-major and 4x4; a 2-D homography occupies the x,
  // y and w rows, with the z axis left as identity.
  const m = [
    a, d, 0, g,
    b, e, 0, i,
    0, 0, 1, 0,
    c, f, 0, 1,
  ];
  return `matrix3d(${m.map(cssNumber).join(", ")})`;
}

/** Fixed-point with enough digits for the perspective terms to survive.
 *
 * Two constraints meet here. CSS does not accept exponential notation, and
 * the perspective terms g and i are on the order of 1/width — round them to
 * six places and the error, multiplied back by a coordinate of a few
 * hundred pixels, moves a corner by a visible fraction of a pixel. So:
 * fixed notation, twelve places, trailing zeros trimmed. */
function cssNumber(value: number): string {
  const fixed = value.toFixed(12);
  const trimmed = fixed.replace(/0+$/, "").replace(/\.$/, "");
  return trimmed === "-0" ? "0" : trimmed;
}
