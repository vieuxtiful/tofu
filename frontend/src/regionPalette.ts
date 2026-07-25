// 🍢 per-region colour assignment for Basil's plating halo.
//
// Until now the app had exactly one bbox colour, a global user setting
// (tofu.bboxColor) delivered to the canvas as --bbox-color. Making a
// semantic unit's members individually identifiable needs a *per-region*
// colour instead, so this wraps the existing eight-colour palette rather
// than introducing a second, differently-tuned one.

import { BBOX_COLORS } from "./BBoxColorDropdown";

/** Colour for the nth member of a semantic unit, cycling the palette. */
export function regionColor(index: number): string {
  const colors = BBOX_COLORS;
  return colors[((index % colors.length) + colors.length) % colors.length].value;
}

/** Human-readable name of the same colour, for tooltips and aria labels. */
export function regionColorName(index: number): string {
  const colors = BBOX_COLORS;
  return colors[((index % colors.length) + colors.length) % colors.length].label;
}

/** Map a unit's region ids to colours, so the strip, the typed tokens and
 * the plan rows all agree on which colour means which region. */
export function regionColorMap(regionIds: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  regionIds.forEach((regionId, index) => {
    map[regionId] = regionColor(index);
  });
  return map;
}

/** Translucent form of a palette colour, using the RRGGBBAA hex-suffix
 * idiom already used throughout BBoxCanvas. */
export function regionTint(color: string, alpha: number): string {
  const byte = Math.max(0, Math.min(255, Math.round(alpha * 255)));
  return `${color}${byte.toString(16).padStart(2, "0")}`;
}
