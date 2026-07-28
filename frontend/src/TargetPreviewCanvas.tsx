import { useCallback, useEffect, useMemo, useRef, useState, CSSProperties } from "react";
import { FaPlus, FaMinus } from "react-icons/fa";
import { InstText, FontFamily, SemanticTextUnit } from "./api";
import { loadFontPreview } from "./FontCombobox";
import { parseQuad, quadToMatrix3d } from "./perspective";
import { effectiveStrokeWidth, fitWrappedText, getMeasureContext, FitResult, MIN_FONT_PX } from "./textFit";
import {
  dupeSpec, resolveFontPath, measureFontSpec, adviseShrink,
  SizeAdvisory, ITALIC_SHEAR_DEG, VERTICAL_ROW_FACTOR, SUB_SUPER_RATIO,
} from "./doppelganger";
import "./bbox.css";

interface TargetPreviewCanvasProps {
  className?: string;
  imageUrl: string | null;
  manifest: InstText[];
  semanticUnits?: SemanticTextUnit[];
  imgNaturalSize: { width: number; height: number } | null;
  familiesByLang?: Record<string, FontFamily[]>;
  defaultTargLang: string;
  label: string;
  // linked mode
  linked: boolean;
  showZoom?: boolean;
  controlledZoom?: number;
  onZoomChange?: (zoom: number) => void;
  controlledScroll?: { x: number; y: number };
  onScrollChange?: (scroll: { x: number; y: number }) => void;
  controlledHeight?: number | null;
  onHeightChange?: (height: number) => void;
  onDoubleClickExpand?: () => void;
}

type PanState = { startClientX: number; startClientY: number; startScrollLeft: number; startScrollTop: number } | null;

/** Below this, cicerone's own detection polygon is not trusted enough to
 *  shape the suppression plate, and the full bbox is used instead — an
 *  over-tight plate would leave source ink visible around the target. */
const PLATE_MASK_MIN_CONF = 0.5;

export default function TargetPreviewCanvas({
  className, imageUrl, manifest, semanticUnits = [], imgNaturalSize, familiesByLang, defaultTargLang, label,
  linked, showZoom = true, controlledZoom, onZoomChange, controlledScroll, onScrollChange,
  controlledHeight, onHeightChange, onDoubleClickExpand,
}: TargetPreviewCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [internalZoom, setInternalZoom] = useState(1);
  const [fitWidth, setFitWidth] = useState<number | null>(null);
  const [pan, setPan] = useState<PanState>(null);
  const lastDragHandleDown = useRef(0);

  const zoom = linked && controlledZoom !== undefined ? controlledZoom : internalZoom;
  // Basil's persisted plan is authoritative after plating.  Keep the
  // underlying boxes/styles from their cube instances, but project the
  // target-language block text onto those cubes before measurement/drawing.
  const displayManifest = useMemo(() => {
    const plated = new Map<string, string>();
    for (const unit of semanticUnits) {
      const substitution = unit.substitution;
      if (!substitution?.applied) continue;
      const assignments = substitution.assignments ?? [];
      const cubes = substitution.spatial_anchor_order ?? substitution.source_region_order ?? unit.region_ids;
      const orderedBlocks = [...assignments].sort((left, right) =>
        Math.min(...left.target_positions) - Math.min(...right.target_positions)
      );
      for (const assignment of assignments) {
        // Legacy plans predate anchor_id.  Recover their deterministic
        // block→cube relation in the client immediately, even before the
        // server's persisted migration is fetched on the next reload.
        const legacyIndex = orderedBlocks.indexOf(assignment);
        const anchorId = assignment.anchor_id ?? cubes[legacyIndex] ?? assignment.region_id;
        if (assignment.text.trim()) plated.set(anchorId, assignment.text);
      }
    }
    return manifest.map((inst) => plated.has(inst.id) ? { ...inst, target_text: plated.get(inst.id)! } : inst);
  }, [manifest, semanticUnits]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setFitWidth(el.clientWidth));
    ro.observe(el);
    setFitWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // sync scroll from external source when linked
  useEffect(() => {
    if (!linked || !controlledScroll) return;
    const el = containerRef.current;
    if (!el) return;
    if (el.scrollLeft !== controlledScroll.x || el.scrollTop !== controlledScroll.y) {
      el.scrollLeft = controlledScroll.x;
      el.scrollTop = controlledScroll.y;
    }
  }, [linked, controlledScroll]);

  const natural = imgNaturalSize;
  const baseWidth = natural
    ? Math.min(natural.width, fitWidth ?? natural.width)
    : null;
  const renderedW = natural && baseWidth ? baseWidth * zoom : null;
  const scale = natural && renderedW ? renderedW / natural.width : 1;
  const px = (v: number) => v * scale;

  // check if image fits entirely within the viewport at zoom=1
  const containerHeight = 300;
  const fitsEntirely = natural && baseWidth
    ? baseWidth <= (fitWidth ?? 0) && natural.height * (baseWidth / natural.width) <= containerHeight
    : true;

  const setZoomBoth = useCallback((z: number) => {
    const clamped = Math.max(0.25, Math.min(4, z));
    setInternalZoom(clamped);
    onZoomChange?.(clamped);
  }, [onZoomChange]);

  const onPanStart = useCallback((e: React.MouseEvent) => {
    const el = containerRef.current;
    if (!el) return;
    setPan({
      startClientX: e.clientX, startClientY: e.clientY,
      startScrollLeft: el.scrollLeft, startScrollTop: el.scrollTop,
    });
  }, []);

  const onPanMove = useCallback((e: React.MouseEvent) => {
    if (!pan) return;
    const el = containerRef.current;
    if (el) {
      el.scrollLeft = pan.startScrollLeft - (e.clientX - pan.startClientX);
      el.scrollTop = pan.startScrollTop - (e.clientY - pan.startClientY);
    }
  }, [pan]);

  const onPanEnd = useCallback(() => {
    if (pan) {
      const el = containerRef.current;
      if (el) onScrollChange?.({ x: el.scrollLeft, y: el.scrollTop });
    }
    setPan(null);
  }, [pan, onScrollChange]);

  // report scroll changes when linked
  useEffect(() => {
    if (!linked || !onScrollChange) return;
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => onScrollChange({ x: el.scrollLeft, y: el.scrollTop });
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [linked, onScrollChange]);

  // drag handle to expand/collapse canvas height
  const MIN_CANVAS_H = 200;
  const MAX_CANVAS_H = 800;
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(300);
  const onExpandDragStart = useCallback((e: React.MouseEvent) => {
    // double-click detection via timestamp
    const now = Date.now();
    if (now - lastDragHandleDown.current < 300) {
      lastDragHandleDown.current = 0;
      onDoubleClickExpand?.();
      return;
    }
    lastDragHandleDown.current = now;

    dragStartY.current = e.clientY;
    dragStartHeight.current = containerRef.current?.parentElement?.offsetHeight ?? 300;
    const dynamicMax = MAX_CANVAS_H;
    let dragging = false;
    const onMove = (ev: MouseEvent) => {
      if (!dragging) {
        dragging = true;
        ev.preventDefault();
      }
      const delta = ev.clientY - dragStartY.current;
      const newHeight = Math.max(MIN_CANVAS_H, Math.min(dynamicMax, dragStartHeight.current + delta));
      if (onHeightChange) {
        onHeightChange(newHeight);
      }
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [onHeightChange, imgNaturalSize, fitWidth, zoom, onDoubleClickExpand]);

  // one region's typographic "dupe" — the ticket describing exactly what
  // scribe.render() will do to it. Shared by both the offscreen canvas
  // measurement (textFit) and the actual React style, so the two are
  // guaranteed to agree on what's being measured vs drawn.
  const dupeCtx = useMemo(
    () => ({ familiesByLang, defaultTargLang }),
    [familiesByLang, defaultTargLang],
  );
  const specFor = useCallback(
    (inst: InstText) => dupeSpec(inst, dupeCtx),
    [dupeCtx],
  );

  // load fonts for target text rendering, THEN compute the wrap/fit for
  // each region against the font that's actually going to be used —
  // measuring before the @font-face finishes loading silently measures
  // the browser's fallback system font instead, producing a fit that
  // doesn't match what then gets painted
  const [fitCache, setFitCache] = useState<Record<string, FitResult>>({});
  const [advisories, setAdvisories] = useState<Record<string, SizeAdvisory>>({});
  useEffect(() => {
    let cancelled = false;
    // Any resolved path gets an @font-face directly. The /api/fonts family
    // list is NOT consulted: it is truncated to 24 families and is only
    // populated for languages another panel already fetched, so gating on
    // it degraded both the measurement and the paint to a system font for
    // every face outside that cut.
    const paths = new Set<string>();
    displayManifest.forEach((inst) => {
      const { path } = resolveFontPath(inst, dupeCtx);
      if (path) paths.add(path);
    });
    Promise.all([...paths].map((p) => loadFontPreview(p))).then(() => {
      if (cancelled) return;
      const ctx = getMeasureContext();
      // fit against the REGION'S OWN natural-pixel bbox, UNPADDED —
      // scribe._fit_wrapped() fits against bbox.width/height directly
      // with zero inset (render() draws flush to the bbox edges; see
      // its line-position math), so subtracting any margin here before
      // the fit search would systematically pick a smaller font than
      // the server does. scaled to display size afterward — keeps
      // MIN_FONT_PX etc. meaningful at any zoom level
      const fitOnce = (inst: InstText, forcedSizePx: number | null): FitResult => {
        const spec = specFor(inst);
        return fitWrappedText(
          ctx, inst.target_text ?? "",
          Math.max(1, inst.bounding_box.width),
          Math.max(1, inst.bounding_box.height),
          measureFontSpec(spec),
          forcedSizePx ?? spec.explicitSizePx,
          spec.leadingPx, spec.wrapText, spec.strokeWidthPx,
        );
      };
      const next: Record<string, FitResult> = {};
      for (const inst of displayManifest) {
        if (!inst.target_text || inst.dnt) continue;
        const spec = specFor(inst);
        let fit = fitOnce(inst, null);
        // scribe re-fits super/subscript at round(size * 0.65) — and
        // re-wraps, since a smaller font breaks lines differently —
        // but only in auto-fit mode; an explicit size is taken verbatim.
        if ((spec.superscript || spec.subscript) && !spec.explicitSizePx) {
          fit = fitOnce(inst, Math.max(MIN_FONT_PX, Math.round(fit.fontSizePx * SUB_SUPER_RATIO)));
        }
        next[inst.id] = fit;
      }
      setFitCache(next);
      // The advisory reads the SAME context the fit just used, at the
      // fitted size, so the ink it reports is the ink about to be painted.
      // Tallest line, not the whole block: the question is how big the
      // GLYPHS came out next to the source's, which a wrapped block's
      // total height would hide.
      setAdvisories(adviseShrink(next, displayManifest, semanticUnits, (inst, fit) => {
        const spec = specFor(inst);
        ctx.font = measureFontSpec(spec).replace("{size}", String(fit.fontSizePx));
        let tallest = 0;
        for (const line of fit.lines) {
          const m = ctx.measureText(line || " ");
          const asc = m.actualBoundingBoxAscent, desc = m.actualBoundingBoxDescent;
          if (asc != null && desc != null) tallest = Math.max(tallest, asc + desc);
        }
        return tallest;
      }));
    });
    return () => { cancelled = true; };
  }, [displayManifest, semanticUnits, dupeCtx, specFor]);

  return (
    <div
      className={`bezier-card soft-shadow relative flex min-h-[200px] flex-col rounded-lg bg-zinc-100 dark:bg-zinc-900 ${className ?? ""}`}
      style={{ height: controlledHeight !== undefined ? (controlledHeight ?? 300) : 300, transition: "height 0.3s ease-in-out" }}
    >
      <div
        ref={containerRef}
        className="bbox-canvas-scroll relative flex-1 mr-6"
        style={{
          cursor: pan ? "grabbing" : "grab",
          overflow: "auto",
        }}
        onMouseDown={onPanStart}
        onMouseMove={onPanMove}
        onMouseUp={onPanEnd}
        onMouseLeave={() => { onPanEnd(); }}
      >
        {imageUrl ? (
          <div style={{ position: "relative", display: "inline-block" }}>
            <img
              ref={imgRef}
              src={imageUrl}
              alt="target preview"
              style={{
                display: "block",
                width: renderedW ?? "auto",
                height: "auto",
                maxWidth: "none",
                opacity: 0.3,
              }}
              draggable={false}
            />
            {natural && renderedW && (
              <div
                style={{
                  position: "absolute",
                  top: 0, left: 0,
                  width: renderedW,
                  height: natural.height * scale,
                  pointerEvents: "none",
                }}
              >
                {/* Keep source-text suppression beneath every target glyph.
                    A per-region background attached to the text container
                    creates an accidental occlusion layer when semantic cubes
                    overlap after plating. */}
                {displayManifest.map((inst) => {
                  if (!inst.target_text || inst.dnt) return null;
                  const bg = inst.background_profile?.dominant_color;
                  if (!bg || bg === "transparent") return null;
                  const b = inst.bounding_box;
                  // Cleanse erases the STROKE MASK, not the box. Clipping
                  // the plate to cicerone's own detection polygon follows
                  // the same outline, so a rotated or irregular run stops
                  // punching a rectangular hole through artwork the render
                  // will actually leave intact.
                  const mask = inst.segmentation_mask;
                  const poly = mask && mask.confidence >= PLATE_MASK_MIN_CONF && mask.polygon?.length >= 3
                    ? `polygon(${mask.polygon
                        .map(([mx, my]) => `${px(mx - b.x)}px ${px(my - b.y)}px`)
                        .join(", ")})`
                    : undefined;
                  return <div key={`background-${inst.id}`} style={{
                    position: "absolute",
                    left: px(b.x), top: px(b.y),
                    width: px(b.width), height: px(b.height),
                    background: bg, zIndex: 0, clipPath: poly,
                  }} />;
                })}
                {displayManifest.map((inst) => {
                  if (!inst.target_text || inst.dnt) return null;
                  const spec = specFor(inst);
                  const fit = fitCache[inst.id];
                  // wrap/fit not computed yet (fonts still loading) —
                  // fall back to the region's own detected size as a
                  // single line so something reasonable shows immediately
                  const lines = fit?.lines ?? [inst.target_text];
                  const fontSizePx = (fit?.fontSizePx ?? inst.characteristics?.size ?? inst.bounding_box.height * 0.7) * scale;
                  const lineAdvancePx = (fit?.lineAdvancePx ?? fontSizePx * 1.2 / scale) * scale;
                  const effectiveStrokeWidthPx = effectiveStrokeWidth(
                    spec.strokeWidthPx, fontSizePx / scale,
                  ) * scale;
                  const alignH = spec.alignH;  // scribe defaults to "center", not left
                  const defaultAlignItems = alignH === "right" ? "flex-end" : alignH === "left" ? "flex-start" : "center";
                  // "last_left"/"last_right"/"justify_center" describe the
                  // LAST line's alignment when the rest are justified —
                  // real inter-word stretch-justify isn't implemented
                  // (see textFit.ts docs: this preview approximates, it
                  // doesn't pixel-match), but the last-line override maps
                  // cleanly onto flexbox align-self per line
                  const lastLineAlignSelf = spec.justification === "last_left" ? "flex-start"
                    : spec.justification === "last_right" ? "flex-end"
                    : spec.justification === "justify_center" ? "center"
                    : undefined;
                  const isVertical = spec.isVertical;
                  // vertical mode: split target text into words, each word
                  // becomes a column of characters stacked top-to-bottom,
                  // words juxtaposed horizontally left-to-right (or right-to-left)
                  const verticalWords = isVertical
                    ? (spec.isRtl
                      ? inst.target_text.split(/\s+/).filter(Boolean).reverse()
                      : inst.target_text.split(/\s+/).filter(Boolean))
                    : [];
                  // scribe steps a stacked column by font.size * 1.15,
                  // compressed by tsume * font.size * 0.1 per row — not by
                  // the horizontal line advance.
                  const verticalRowPx = fontSizePx * VERTICAL_ROW_FACTOR - spec.tsume * fontSizePx * 0.1;
                  // scribe shifts the whole BLOCK for super/subscript
                  // (block_h * -0.3 / +0.2) rather than relying on an
                  // inline baseline shift, which a flex item would ignore.
                  const blockPx = lines.length * lineAdvancePx;
                  const scriptShiftPx = spec.superscript ? -blockPx * 0.3
                    : spec.subscript ? blockPx * 0.2 : 0;
                  const spatial = inst.style_profile?.transform ?? {};
                  // A quad REPLACES the affine stage, matching scribe: the
                  // corner positions already encode any shear, so applying
                  // skew as well would apply it twice. matrix3d maps the
                  // box's own local space, so the origin moves to its
                  // top-left corner and the anchor stops applying — an
                  // anchor names the point a shear holds still, and a quad
                  // states outright where all four corners land.
                  const quadMatrix = quadToMatrix3d(
                    parseQuad(spatial.quad), inst.bounding_box,
                  );
                  const anchor = quadMatrix
                    ? "0 0"
                    : (spatial.skew_anchor ?? "center").replace("middle", "center").replace("_", " ");
                  const contentTransform = [
                    spatial.offset_x ? `translateX(${Number(spatial.offset_x) * scale}px)` : null,
                    spatial.offset_y ? `translateY(${Number(spatial.offset_y) * scale}px)` : null,
                    spec.baselineShiftPx ? `translateY(${-spec.baselineShiftPx * scale}px)` : null,
                    scriptShiftPx ? `translateY(${scriptShiftPx}px)` : null,
                    spec.rotationDeg ? `rotate(${spec.rotationDeg}deg)` : null,
                    quadMatrix,
                    quadMatrix ? null : (spatial.skew_x ? `skewX(${Number(spatial.skew_x)}deg)` : null),
                    quadMatrix ? null : (spatial.skew_y ? `skewY(${Number(spatial.skew_y)}deg)` : null),
                    quadMatrix || !(spatial.scale_x || spatial.scale_y) ? null
                      : `scale(${Number(spatial.scale_x ?? 1)}, ${Number(spatial.scale_y ?? 1)})`,
                  ].filter(Boolean).join(" ") || undefined;
                  // CSS text-shadow takes the same (dx, dy, blur, color)
                  // scribe's _shadow_layer composites beneath each line,
                  // and like scribe it is drawn only when the region
                  // actually carries a shadow.
                  const textShadow = spec.shadow
                    ? `${spec.shadow.offset_x * scale}px ${spec.shadow.offset_y * scale}px ${spec.shadow.blur * scale}px ${spec.shadow.color}`
                    : undefined;
                  // A synthetic italic is a mechanical SHEAR of the upright
                  // face (scribe's ITALIC_SHEAR, applied per line AFTER the
                  // fit), not a real italic face — so it is a transform
                  // here too, never `font-style: italic`, which would make
                  // the browser pick the file verbatim and slant nothing.
                  const glyphSkew = spec.syntheticItalic ? `skewX(-${ITALIC_SHEAR_DEG}deg)` : undefined;
                  // Shared per-line/per-glyph typography.
                  const inkStyle: CSSProperties = {
                    fontFamily: spec.cssFontFamily,
                    color: spec.color,
                    fontSize: `${fontSizePx}px`,
                    lineHeight: `${lineAdvancePx}px`,
                    letterSpacing: spec.letterSpacingPx ? `${spec.letterSpacingPx * scale}px` : undefined,
                    textDecoration: spec.underline ? "underline" : undefined,
                    textUnderlineOffset: spec.underline && spec.underlineOffsetPx != null
                      ? `${spec.underlineOffsetPx * scale}px` : undefined,
                    textDecorationThickness: spec.underline && spec.underlineWidthPx != null
                      ? `${spec.underlineWidthPx * scale}px` : undefined,
                    textShadow,
                    transform: glyphSkew,
                    WebkitTextStroke: spec.strokeColor && effectiveStrokeWidthPx
                      ? `${spec.strokePosition === "outer" ? effectiveStrokeWidthPx * 2 : effectiveStrokeWidthPx}px ${spec.strokeColor}` : undefined,
                    paintOrder: spec.strokePosition === "inner" ? "fill stroke" : "stroke fill",
                    whiteSpace: "pre",
                  };
                  // Say it out loud whenever this region is NOT a faithful
                  // prediction of the render. Silence here is what made the
                  // old sans-serif fallback so misleading: the preview
                  // looked authoritative while measuring a font the server
                  // would never load.
                  // A region can be BOTH font-substituted and shrunk, and
                  // they are independent facts about it — collect every
                  // note rather than letting the first one mask the rest.
                  // Colour follows the most severe.
                  const advisory = advisories[inst.id];
                  const notes: string[] = [];
                  let markerColor: string | null = null;
                  if (spec.provenance === "unresolved") {
                    notes.push("no font resolved — this region's size and wrapping are not a prediction");
                    markerColor = "#dc2626";
                  } else if (spec.provenance === "nearest_neighbor") {
                    const fm = inst.font_match;
                    const cohort = fm?.cohort;
                    notes.push(
                      cohort
                        // A cohort answer is a stronger claim than a lone
                        // region's: it had to survive every region on the
                        // sign, so say which regions vouched for it.
                        ? `font: one face agreed across ${cohort.region_ids.join(", ")} on this sign `
                          + `(worst-fit member ${(cohort.agreement * 100).toFixed(0)}% of its own best) — `
                          + `the render keeps its own auto font until you accept this in the region table`
                        : `font: nearest installed substitute by glyph shape (${fm?.status}, score `
                          + `${fm?.recommended_substitute?.score?.toFixed(2)}) — the render keeps its own auto `
                          + `font until you accept this in the region table`,
                    );
                    const overruled = cohort?.dissent?.find((d) => d.region_id === inst.id);
                    if (overruled) {
                      notes.push(
                        `  ↳ on its own this region preferred ${overruled.preferred} `
                        + `(${overruled.preferred_score?.toFixed(2)} vs ${overruled.cohort_score.toFixed(2)}); `
                        + `the sign's shared face won`,
                      );
                    }
                    markerColor = "#f59e0b";
                  }
                  if (advisory) {
                    notes.push(
                      `size: the translation renders ${Math.round((1 - advisory.ratio) * 100)}% shorter than the `
                      + `source lettering (${advisory.targetInkPx}px vs ${advisory.sourceInkPx}px tall) — it cannot `
                      + `fit this box at the source's size`
                      + (advisory.unshrunkPeers?.length
                        ? `, while ${advisory.unshrunkPeers.join(", ")} on the same sign kept full size`
                        : ""),
                    );
                    markerColor = markerColor ?? "#0ea5e9";
                  }
                  const marker = notes.length
                    ? { color: markerColor!, title: notes.join("\n") }
                    : null;
                  return (
                    <div
                      key={inst.id}
                      style={{
                        position: "absolute",
                        left: px(inst.bounding_box.x),
                        top: px(inst.bounding_box.y),
                        width: px(inst.bounding_box.width),
                        height: px(inst.bounding_box.height),
                        // Boxes are spatial anchors, not crop masks.  Let a
                        // transformed glyph occupy its actual visual extent.
                        overflow: "visible",
                        zIndex: 1,
                        pointerEvents: "none",
                      }}
                    >
                      <div style={{
                        width: "100%", height: "100%", display: "flex",
                        flexDirection: isVertical ? "row" : "column",
                        justifyContent: spec.alignV === "top" ? "flex-start" : spec.alignV === "bottom" ? "flex-end" : "center",
                        alignItems: isVertical ? (alignH === "right" ? "flex-end" : alignH === "left" ? "flex-start" : "center") : defaultAlignItems,
                        // scribe reorders RTL runs itself because PIL has no
                        // bidi; the browser already does that natively, so
                        // the equivalent here is to declare the base
                        // direction rather than to reverse anything.
                        direction: !isVertical && spec.isRtl ? "rtl" : undefined,
                        transform: contentTransform, transformOrigin: anchor,
                      }}>
                      {isVertical ? (
                        // vertical: each word is a column of characters
                        verticalWords.map((word, wi) => (
                          <div
                            key={wi}
                            style={{
                              display: "flex",
                              flexDirection: "column",
                              alignItems: "center",
                              marginRight: wi < verticalWords.length - 1 ? `${fontSizePx * 0.3}px` : undefined,
                            }}
                          >
                            {word.split("").map((ch, ci) => (
                              <span
                                key={ci}
                                style={{ ...inkStyle, lineHeight: `${verticalRowPx}px` }}
                              >
                                {ch}
                              </span>
                            ))}
                          </div>
                        ))
                      ) : (
                      lines.map((line, i) => (
                        <span
                          key={i}
                          style={{
                            ...inkStyle,
                            // scribe applies `indent` to the FIRST line only
                            marginLeft: i === 0 && spec.indentPx ? `${spec.indentPx * scale}px` : undefined,
                            alignSelf: i === lines.length - 1 ? lastLineAlignSelf : undefined,
                          }}
                        >
                          {line}
                        </span>
                      ))
                      )}
                      </div>
                      {marker && (
                        <div
                          title={marker.title}
                          style={{
                            position: "absolute", top: -3, right: -3,
                            width: 6, height: 6, borderRadius: "50%",
                            background: marker.color, zIndex: 2,
                          }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="flex h-48 items-center justify-center text-zinc-600 text-sm">
            no image loaded
          </div>
        )}
      </div>
      {/* zoom controls — only when showZoom and image doesn't fit entirely */}
      {showZoom && !fitsEntirely && (
        <div className="absolute bottom-8 left-2 flex gap-1 z-20 w-fit">
          <button
            onClick={() => {
              const el = containerRef.current;
              if (!el || !renderedW || !natural) { setZoomBoth(zoom - 0.25); return; }
              const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
              const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
              setZoomBoth(zoom - 0.25);
              requestAnimationFrame(() => {
                const ne = containerRef.current;
                if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; }
              });
            }}
            className="flex items-center rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
          >
            <FaMinus size={10} />
          </button>
          <span className="rounded-sm bg-white px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">{(zoom * 100).toFixed(0)}%</span>
          <button
            onClick={() => {
              const el = containerRef.current;
              if (!el || !renderedW || !natural) { setZoomBoth(zoom + 0.25); return; }
              const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
              const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
              setZoomBoth(zoom + 0.25);
              requestAnimationFrame(() => {
                const ne = containerRef.current;
                if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; }
              });
            }}
            className="flex items-center rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
          >
            <FaPlus size={10} />
          </button>
          <button
            onClick={() => setZoomBoth(1)}
            className="rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
          >
            fit
          </button>
        </div>
      )}
      {/* label + grabber */}
      <div className="title-drag-handle shrink-0 flex items-center justify-center" onMouseDown={onExpandDragStart} style={{ position: "relative" }} title="drag to resize · double-click to toggle height">
        <svg width="42" height="14" viewBox="0 0 42 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="0.5" y="1" width="41" height="4.5" rx="2.25" fill="currentColor" />
          <rect x="11" y="8" width="20" height="4.5" rx="2.25" fill="currentColor" />
        </svg>
        <span className="subtext text-xs text-zinc-500" style={{ position: "absolute", right: "0.75rem", top: "50%", transform: "translateY(-50%)" }}>{label}</span>
      </div>
    </div>
  );
}
