// Human-readable labels and explanations for OCR quality reason codes.
//
// The backend (src/tofu/layers/savor.py, assess_ocr_quality) is the source of
// truth for the codes themselves; this map only translates each code into a
// short chip label and a one-sentence explanation shown in a hover overlay.
// An unknown code falls back to the legacy space-replaced rendering so a new
// backend reason never renders as an empty chip.

export interface OcrReasonInfo {
  /** Short label shown on the chip in the Capture-tab review list. */
  label: string;
  /** One-sentence explanation shown in the MUI Tooltip on hover/focus. */
  description: string;
}

export const OCR_REASON_INFO: Record<string, OcrReasonInfo> = {
  asset_unreadable: {
    label: "Image unreadable",
    description:
      "The source image could not be decoded, so this region can't be analyzed.",
  },
  region_too_small: {
    label: "Region too small",
    description:
      "The detected box is under 3px — too small to measure glyph geometry.",
  },
  geometry_unavailable: {
    label: "Geometry unavailable",
    description:
      "Connected-component analysis couldn't isolate glyphs in this crop, so glyph-level checks were skipped.",
  },
  below_mark_resolution: {
    label: "Glyphs too small",
    description:
      "Estimated glyph height is under 3px; diacritics and fine marks can't be resolved reliably.",
  },
  low_effective_glyph_height: {
    label: "Small glyphs",
    description:
      "Glyphs are 3–8px tall — readable, but fine strokes may have been misread. Worth a visual check.",
  },
  component_count_mismatch: {
    label: "Component mismatch",
    description:
      "The number of ink blobs doesn't match the character count. Often harmless (serifs, Q-tails), but flagged for provenance.",
  },
  ambiguous_segmentation: {
    label: "Ambiguous segmentation",
    description:
      "The crop's ink can't be cleanly split into the recognized characters, and the read confidence is low. Re-reading may help.",
  },
  engine_unavailable: {
    label: "No re-read engine",
    description:
      "No OCR engine is available to re-read this region, so a suspected misread can't be auto-corrected.",
  },
  no_recognized_glyphs: {
    label: "No text recognized",
    description: "The recognizer returned no characters for this region.",
  },
};

/**
 * Resolve a reason code to its chip info, falling back to the legacy
 * space-replaced rendering for any code not yet in the map. The description
 * is empty for unknown codes so the tooltip is suppressed naturally.
 */
export function ocrReasonInfo(code: string): OcrReasonInfo {
  return (
    OCR_REASON_INFO[code] ?? {
      label: code.replace(/_/g, " "),
      description: "",
    }
  );
}
