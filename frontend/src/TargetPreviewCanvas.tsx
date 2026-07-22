import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FaPlus, FaMinus } from "react-icons/fa";
import { InstText, FontFamily, SemanticTextUnit } from "./api";
import { loadFontPreview, fontNameForPath } from "./FontCombobox";
import { fitWrappedText, getMeasureContext, FitResult } from "./textFit";
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

  // resolved, non-size CSS bits for a region's target text — shared by
  // both the canvas measurement (textFit) and the actual React style, so
  // the two are guaranteed to agree on what's being measured vs drawn.
  // explicit style_profile.font_family wins; otherwise falls back to
  // resolved_font_family (what "auto" currently resolves to, computed
  // server-side by scribe.resolve_auto_font — the same resolution
  // render() itself performs) instead of the browser's generic default.
  const resolveTextSpec = useCallback((inst: InstText) => {
    const fontPath = inst.style_profile?.font_family ?? inst.resolved_font_family ?? undefined;
    const fontStyleHint = (inst.characteristics?.font_style ?? "").toLowerCase();
    let fontWeight = 400;
    let fontStyle: "italic" | "normal" = "normal";
    let cssFontFamily = "sans-serif";
    if (fontPath && familiesByLang) {
      const lang = inst.target_language ?? defaultTargLang;
      const families = familiesByLang[lang] ?? [];
      const fam = families.find((f) => f.weights.some((w) => w.path === fontPath) || f.best_path === fontPath);
      const weight = fam?.weights.find((w) => w.path === fontPath);
      if (weight) {
        const wc = weight.weight_class ?? 400;
        fontWeight = wc <= 300 ? 300 : wc <= 500 ? 400 : wc <= 700 ? 700 : 800;
        if ((weight.subfamily || "").toLowerCase().includes("italic")) fontStyle = "italic";
        cssFontFamily = `"${fontNameForPath(fontPath)}"`;
      } else {
        if (fontStyleHint.includes("bold")) fontWeight = 700;
        if (fontStyleHint.includes("italic")) fontStyle = "italic";
      }
    } else {
      if (fontStyleHint.includes("bold")) fontWeight = 700;
      if (fontStyleHint.includes("italic")) fontStyle = "italic";
    }
    const color = inst.style_profile?.color ?? inst.characteristics?.color ?? undefined;
    // render() passes style_profile.font_size ALONE as _fit_wrapped's
    // explicit_size -- None means true binary-search auto-fit. falling
    // back to characteristics.size (the DETECTED source-text size) here
    // would skip that auto-fit entirely and draw at the source's own
    // size, which is tuned for the SOURCE string's length, not the
    // (usually different-length) translation -- a real mismatch for any
    // region where the translation runs longer or shorter than source.
    const explicitSizePx = inst.style_profile?.font_size ?? null;
    const sp = inst.style_profile;
    return {
      fontPath, fontWeight, fontStyle, cssFontFamily, color, explicitSizePx,
      alignV: sp?.align_v ?? null,
      justification: sp?.justification ?? null,
      indentPx: sp?.indent ?? 0,
      trackingPx: sp?.tracking ?? 0,
      leadingPx: sp?.leading ?? null,
      baselineShiftPx: sp?.baseline_shift ?? 0,
      strokeColor: sp?.stroke_color ?? null,
      strokeWidthPx: sp?.stroke_width ?? 0,
      subscript: sp?.subscript ?? false,
      superscript: sp?.superscript ?? false,
      rotationDeg: inst.characteristics?.positioning?.rotation_deg ?? 0,
    };
  }, [familiesByLang, defaultTargLang]);

  // load fonts for target text rendering, THEN compute the wrap/fit for
  // each region against the font that's actually going to be used —
  // measuring before the @font-face finishes loading silently measures
  // the browser's fallback system font instead, producing a fit that
  // doesn't match what then gets painted
  const [fitCache, setFitCache] = useState<Record<string, FitResult>>({});
  useEffect(() => {
    let cancelled = false;
    const loads: Promise<void>[] = [];
    displayManifest.forEach((inst) => {
      const fontPath = inst.style_profile?.font_family ?? inst.resolved_font_family;
      if (!fontPath || !familiesByLang) return;
      const lang = inst.target_language ?? defaultTargLang;
      const families = familiesByLang[lang] ?? [];
      const fam = families.find((f) => f.weights.some((w) => w.path === fontPath) || f.best_path === fontPath);
      if (fam) loads.push(loadFontPreview(fontPath, fam.family));
    });
    Promise.all(loads).then(() => {
      if (cancelled) return;
      const ctx = getMeasureContext();
      const next: Record<string, FitResult> = {};
      for (const inst of displayManifest) {
        if (!inst.target_text || inst.dnt) continue;
        const spec = resolveTextSpec(inst);
        const fontSpecTemplate = `${spec.fontStyle} ${spec.fontWeight} {size}px ${spec.cssFontFamily}`;
        // fit against the REGION'S OWN natural-pixel bbox, UNPADDED —
        // scribe._fit_wrapped() fits against bbox.width/height directly
        // with zero inset (render() draws flush to the bbox edges; see
        // its line-position math), so subtracting any margin here before
        // the fit search would systematically pick a smaller font than
        // the server does. scaled to display size afterward — keeps
        // MIN_FONT_PX etc. meaningful at any zoom level
        next[inst.id] = fitWrappedText(
          ctx, inst.target_text,
          Math.max(1, inst.bounding_box.width),
          Math.max(1, inst.bounding_box.height),
          fontSpecTemplate, spec.explicitSizePx, spec.leadingPx,
          Boolean(inst.style_profile?.transform?.wrap_text),
        );
      }
      setFitCache(next);
    });
    return () => { cancelled = true; };
  }, [displayManifest, familiesByLang, defaultTargLang, resolveTextSpec]);

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
                {displayManifest.map((inst) => {
                  if (!inst.target_text || inst.dnt) return null;
                  const bg = inst.background_profile?.dominant_color ?? "transparent";
                  const spec = resolveTextSpec(inst);
                  const fit = fitCache[inst.id];
                  // wrap/fit not computed yet (fonts still loading) —
                  // fall back to the region's own detected size as a
                  // single line so something reasonable shows immediately
                  const lines = fit?.lines ?? [inst.target_text];
                  const fontSizePx = (fit?.fontSizePx ?? inst.characteristics?.size ?? inst.bounding_box.height * 0.7) * scale;
                  const lineAdvancePx = (fit?.lineAdvancePx ?? fontSizePx * 1.2 / scale) * scale;
                  const alignH = inst.style_profile?.align_h;
                  const defaultAlignItems = alignH === "right" ? "flex-end" : alignH === "center" ? "center" : "flex-start";
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
                  const isVertical = inst.style_profile?.target_orientation === "vertical";
                  // vertical mode: split target text into words, each word
                  // becomes a column of characters stacked top-to-bottom,
                  // words juxtaposed horizontally left-to-right (or right-to-left)
                  const isRtl = inst.style_profile?.word_order === "rtl";
                  const verticalWords = isVertical
                    ? (isRtl
                      ? inst.target_text.split(/\s+/).filter(Boolean).reverse()
                      : inst.target_text.split(/\s+/).filter(Boolean))
                    : [];
                  return (
                    <div
                      key={inst.id}
                      style={{
                        position: "absolute",
                        left: px(inst.bounding_box.x),
                        top: px(inst.bounding_box.y),
                        width: px(inst.bounding_box.width),
                        height: px(inst.bounding_box.height),
                        background: bg !== "transparent" ? bg : undefined,
                        display: "flex",
                        flexDirection: isVertical ? "row" : "column",
                        justifyContent: isVertical
                          ? (spec.alignV === "top" ? "flex-start" : spec.alignV === "bottom" ? "flex-end" : "center")
                          : (spec.alignV === "top" ? "flex-start" : spec.alignV === "bottom" ? "flex-end" : "center"),
                        alignItems: isVertical
                          ? (alignH === "right" ? "flex-end" : alignH === "center" ? "center" : "flex-start")
                          : defaultAlignItems,
                        overflow: "hidden",
                        pointerEvents: "none",
                        transform: [
                          spec.baselineShiftPx ? `translateY(${-spec.baselineShiftPx * scale}px)` : null,
                          spec.rotationDeg ? `rotate(${spec.rotationDeg}deg)` : null,
                        ].filter(Boolean).join(" ") || undefined,
                      }}
                    >
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
                                style={{
                                  fontFamily: spec.cssFontFamily,
                                  fontWeight: spec.fontWeight,
                                  fontStyle: spec.fontStyle,
                                  color: spec.color,
                                  fontSize: spec.subscript || spec.superscript ? `${fontSizePx * 0.7}px` : `${fontSizePx}px`,
                                  verticalAlign: spec.subscript ? "sub" : spec.superscript ? "super" : undefined,
                                  lineHeight: `${lineAdvancePx}px`,
                                  letterSpacing: spec.trackingPx ? `${spec.trackingPx * scale}px` : undefined,
                                  textDecoration: inst.style_profile?.underline ? "underline" : undefined,
                                  WebkitTextStroke: spec.strokeColor && spec.strokeWidthPx
                                    ? `${spec.strokeWidthPx * scale}px ${spec.strokeColor}` : undefined,
                                  whiteSpace: "pre",
                                }}
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
                            fontFamily: spec.cssFontFamily,
                            fontWeight: spec.fontWeight,
                            fontStyle: spec.fontStyle,
                            color: spec.color,
                            fontSize: spec.subscript || spec.superscript ? `${fontSizePx * 0.7}px` : `${fontSizePx}px`,
                            verticalAlign: spec.subscript ? "sub" : spec.superscript ? "super" : undefined,
                            lineHeight: `${lineAdvancePx}px`,
                            letterSpacing: spec.trackingPx ? `${spec.trackingPx * scale}px` : undefined,
                            marginLeft: i === 0 && spec.indentPx ? `${spec.indentPx * scale}px` : undefined,
                            alignSelf: i === lines.length - 1 ? lastLineAlignSelf : undefined,
                            textDecoration: inst.style_profile?.underline ? "underline" : undefined,
                            WebkitTextStroke: spec.strokeColor && spec.strokeWidthPx
                              ? `${spec.strokeWidthPx * scale}px ${spec.strokeColor}` : undefined,
                            whiteSpace: "pre",
                          }}
                        >
                          {line}
                        </span>
                      ))
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
            className="flex items-center rounded bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
          >
            <FaMinus size={10} />
          </button>
          <span className="rounded bg-white px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">{(zoom * 100).toFixed(0)}%</span>
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
            className="flex items-center rounded bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
          >
            <FaPlus size={10} />
          </button>
          <button
            onClick={() => setZoomBoth(1)}
            className="rounded bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
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
