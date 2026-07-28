// 🍢 ToFU — projective corner maths tests
//
// The whole point of perspective is the property affine cannot produce:
// converging edges. These assert that directly rather than checking that a
// matrix has some expected shape.

import { describe, it, expect } from "vitest";
import { BBox } from "./api";
import {
  IDENTITY_QUAD, Quad, denormaliseQuad, isIdentityQuad, isUsableQuad,
  parseQuad, quadFromPolygon, quadToMatrix3d,
} from "./perspective";

const BOX: BBox = { x: 80, y: 60, width: 240, height: 90 };

/** Read the 4x4 back out of a matrix3d string, column-major as CSS emits it. */
function coeffs(css: string): number[] {
  const inner = css.slice("matrix3d(".length, -1);
  return inner.split(",").map((v) => Number(v.trim()));
}

/** Apply the recovered homography to a point in the box's local space. */
function project(css: string, x: number, y: number): [number, number] {
  const m = coeffs(css);
  // column-major: a=m[0] d=m[1] g=m[3] | b=m[4] e=m[5] i=m[7] | c=m[12] f=m[13]
  const [a, d, , g, b, e, , i] = m;
  const c = m[12];
  const f = m[13];
  const w = g * x + i * y + 1;
  return [(a * x + b * y + c) / w, (d * x + e * y + f) / w];
}

describe("parseQuad", () => {
  it("accepts four pairs", () => {
    expect(parseQuad([[0, 0], [1, 0], [1, 1], [0, 1]])).toEqual(IDENTITY_QUAD);
  });

  it("rejects malformed values rather than throwing in a render", () => {
    for (const bad of [null, undefined, [], [[0, 0]], [[0, 0], [1, 0], [1, 1]],
                       "quad", [[0, 0], [1, 0], [1, 1], [0]],
                       [[0, 0], [1, 0], [1, 1], [0, NaN]]]) {
      expect(parseQuad(bad)).toBeNull();
    }
  });
});

describe("isIdentityQuad", () => {
  it("treats absent and identity alike", () => {
    expect(isIdentityQuad(null)).toBe(true);
    expect(isIdentityQuad(IDENTITY_QUAD)).toBe(true);
  });

  it("tolerates float noise from a JSON round-trip", () => {
    expect(isIdentityQuad([[0, 0], [1.00001, 0], [1, 1], [0, 1]])).toBe(true);
  });

  it("sees a real corner pull", () => {
    expect(isIdentityQuad([[0.25, 0], [1, 0], [1, 1], [0, 1]])).toBe(false);
  });
});

describe("denormaliseQuad", () => {
  it("maps the identity onto the box itself", () => {
    expect(denormaliseQuad(IDENTITY_QUAD, BOX)).toEqual([
      [80, 60], [320, 60], [320, 150], [80, 150],
    ]);
  });

  it("lets a corner leave the box", () => {
    const [tl] = denormaliseQuad([[-0.5, 0], [1, 0], [1, 1], [0, 1]], BOX);
    expect(tl).toEqual([-40, 60]);
  });
});

describe("isUsableQuad", () => {
  it("accepts a convex trapezoid", () => {
    expect(isUsableQuad([[60, 0], [180, 0], [240, 90], [0, 90]])).toBe(true);
  });

  it("rejects collinear corners", () => {
    expect(isUsableQuad([[0, 0], [120, 0], [240, 0], [0, 90]])).toBe(false);
  });

  it("rejects a self-intersecting bowtie", () => {
    expect(isUsableQuad([[0, 0], [240, 0], [0, 90], [240, 90]])).toBe(false);
  });
});

describe("quadFromPolygon", () => {
  it("recovers a rectangle outline as the identity", () => {
    const outline = [[80, 60], [320, 60], [320, 150], [80, 150]];
    const quad = quadFromPolygon(outline, BOX)!;
    expect(isIdentityQuad(quad)).toBe(true);
  });

  it("orders corners TL, TR, BR, BL regardless of the outline's winding", () => {
    // same trapezoid, listed anticlockwise from the bottom-right
    const outline = [[320, 150], [80, 150], [140, 60], [260, 60]];
    const quad = quadFromPolygon(outline, BOX)!;
    expect(quad[0][1]).toBeCloseTo(0);   // TL on the top edge
    expect(quad[1][1]).toBeCloseTo(0);   // TR on the top edge
    expect(quad[0][0]).toBeLessThan(quad[1][0]);
    expect(quad[3][1]).toBeCloseTo(1);   // BL on the bottom edge
  });

  it("normalises against the box so the seed survives a move", () => {
    const outline = [[140, 60], [260, 60], [320, 150], [80, 150]];
    const quad = quadFromPolygon(outline, BOX)!;
    expect(quad[0][0]).toBeCloseTo((140 - 80) / 240, 6);
    expect(quad[2]).toEqual([1, 1]);
  });

  it("keeps a denser outline's extreme corners", () => {
    // a detector polygon with midpoints along each edge
    const outline = [[140, 60], [200, 60], [260, 60], [290, 105],
                     [320, 150], [200, 150], [80, 150], [110, 105]];
    const quad = quadFromPolygon(outline, BOX)!;
    expect(quad).toHaveLength(4);
    expect(isUsableQuad(denormaliseQuad(quad, BOX))).toBe(true);
  });

  it("declines rather than guessing when the outline cannot be a plane", () => {
    expect(quadFromPolygon(null, BOX)).toBeNull();
    expect(quadFromPolygon([[0, 0], [1, 1]], BOX)).toBeNull();
    expect(quadFromPolygon([[80, 60], [80, 60], [80, 60], [80, 60]], BOX)).toBeNull();
    // collinear outline: no plane to fit
    expect(quadFromPolygon([[80, 60], [160, 60], [240, 60], [320, 60]], BOX)).toBeNull();
    expect(quadFromPolygon([[80, 60], [320, 60], [320, 150], [80, 150]],
                           { x: 0, y: 0, width: 0, height: 90 })).toBeNull();
  });

  it("round-trips through quadToMatrix3d", () => {
    const outline = [[140, 60], [260, 60], [320, 150], [80, 150]];
    const quad = quadFromPolygon(outline, BOX)!;
    expect(quadToMatrix3d(quad, BOX)).toMatch(/^matrix3d\(/);
  });
});

describe("quadToMatrix3d", () => {
  it("returns null for identity, so the caller keeps the skew chain", () => {
    expect(quadToMatrix3d(IDENTITY_QUAD, BOX)).toBeNull();
    expect(quadToMatrix3d(null, BOX)).toBeNull();
  });

  it("returns null for a degenerate quad instead of an unusable matrix", () => {
    expect(quadToMatrix3d([[0, 0], [0.5, 0], [1, 0], [0, 1]], BOX)).toBeNull();
    expect(quadToMatrix3d([[0, 0], [1, 0], [0, 1], [1, 1]], BOX)).toBeNull();
  });

  it("returns null for a zero-area box", () => {
    expect(quadToMatrix3d([[0.25, 0], [1, 0], [1, 1], [0, 1]],
                          { x: 0, y: 0, width: 0, height: 90 })).toBeNull();
  });

  it("sends each box corner to its quad corner", () => {
    const quad: Quad = [[0.25, 0], [0.75, 0], [1, 1], [0, 1]];
    const css = quadToMatrix3d(quad, BOX)!;
    const local: Array<[number, number]> = [
      [0, 0], [BOX.width, 0], [BOX.width, BOX.height], [0, BOX.height],
    ];
    quad.forEach(([nx, ny], i) => {
      const [px, py] = project(css, local[i][0], local[i][1]);
      expect(px).toBeCloseTo(nx * BOX.width, 3);
      expect(py).toBeCloseTo(ny * BOX.height, 3);
    });
  });

  it("produces CONVERGING edges — the thing skew cannot do", () => {
    // top corners pulled inward: the top edge must come out shorter
    const css = quadToMatrix3d([[0.25, 0], [0.75, 0], [1, 1], [0, 1]], BOX)!;
    const topLeft = project(css, 0, 0);
    const topRight = project(css, BOX.width, 0);
    const bottomLeft = project(css, 0, BOX.height);
    const bottomRight = project(css, BOX.width, BOX.height);
    const topWidth = topRight[0] - topLeft[0];
    const bottomWidth = bottomRight[0] - bottomLeft[0];
    expect(topWidth).toBeLessThan(bottomWidth * 0.6);
  });

  it("has a genuinely projective row for a trapezoid", () => {
    // g and i are the perspective terms; an affine map leaves them zero,
    // which is precisely the two degrees of freedom skew does not have
    const css = quadToMatrix3d([[0.25, 0], [0.75, 0], [1, 1], [0, 1]], BOX)!;
    const m = coeffs(css);
    expect(Math.abs(m[3]) + Math.abs(m[7])).toBeGreaterThan(1e-6);
  });

  it("stays affine when the quad only shears", () => {
    // a parallelogram IS expressible affinely, so the perspective terms
    // must stay zero rather than picking up numerical noise
    const css = quadToMatrix3d([[0.25, 0], [1.25, 0], [1, 1], [0, 1]], BOX)!;
    const m = coeffs(css);
    expect(Math.abs(m[3])).toBeLessThan(1e-9);
    expect(Math.abs(m[7])).toBeLessThan(1e-9);
  });

  it("emits 16 finite column-major values", () => {
    const m = coeffs(quadToMatrix3d([[0.25, 0], [0.75, 0], [1, 1], [0, 1]], BOX)!);
    expect(m).toHaveLength(16);
    expect(m.every(Number.isFinite)).toBe(true);
    expect(m[10]).toBe(1);   // z axis untouched
    expect(m[15]).toBe(1);
  });
});
