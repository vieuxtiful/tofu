import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { TbZoomInFilled, TbCircleDashedPlus, TbCircleDashedMinus } from "react-icons/tb";
import { FaPlus, FaMinus } from "react-icons/fa";
import { InstText, BBox, SceneRegion } from "./api";
import "./bbox.css";

// magnifier loupe: moderate halo that magnifies the area under the cursor
const LOUPE_RADIUS = 70;        // px — 140px diameter, deliberately not overbearing
const LOUPE_MAGNIFICATION = 2.5; // relative to the current display scale

interface BBoxCanvasProps {
  imageUrl: string | null;
  manifest: InstText[];
  sceneRegions?: SceneRegion[];
  selectedId: string | null;
  hoveredId: string | null;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
  onAddRegion: (bbox: BBox) => void;
  onUpdateRegion: (id: string, bbox: BBox) => void;
  drawMode: boolean;
  imgNaturalSize: { width: number; height: number } | null;
  onImgLoad: (size: { width: number; height: number }) => void;
  onExpandToggle?: (expanded: boolean) => void;
  preview?: boolean;
  canvasLabel?: string;
  showPreviewControls?: boolean;
  controlledZoom?: number;
  onZoomChange?: (zoom: number) => void;
  controlledScroll?: { x: number; y: number };
  onScrollChange?: (scroll: { x: number; y: number }) => void;
  controlledHeight?: number | null;
  onHeightChange?: (height: number) => void;
  onDoubleClickExpand?: () => void;
  bboxColor?: string;
  bboxBlink?: boolean;
  /** ids of regions that just arrived from the server. Each gets a one-off
   * two-pass shimmer. The set is owned (and cleared) by the caller so the
   * animation cannot replay on an ordinary rerender. */
  newRegionIds?: Set<string> | null;
  onDragStart?: () => void;
  onDragEnd?: () => void;
}

type DragState =
  | { type: "draw"; startX: number; startY: number }
  | { type: "move"; id: string; startMouseX: number; startMouseY: number; origBBox: BBox }
  | { type: "resize"; id: string; handle: string; startMouseX: number; startMouseY: number; origBBox: BBox }
  | { type: "pan"; startClientX: number; startClientY: number; startScrollLeft: number; startScrollTop: number }
  | null;

function bboxClass(inst: InstText): string {
  if (inst.dnt) return "bbox dnt";
  if (inst.confidence !== null && inst.confidence < 0.6) return "bbox bbox-low-conf";
  if (inst.text === null && inst.confidence === null) return "bbox bbox-manual";
  return "bbox bbox-text";
}

function confColor(conf: number | null): string {
  if (conf === null) return "#a1a1aa";
  if (conf >= 0.8) return "#22c55e";
  if (conf >= 0.6) return "#f59e0b";
  return "#ef4444";
}

export default function BBoxCanvas({
  imageUrl, manifest, sceneRegions, selectedId, hoveredId, onSelect, onHover,
  onAddRegion, onUpdateRegion, drawMode, imgNaturalSize, onImgLoad, onExpandToggle,
  preview = false, canvasLabel, showPreviewControls = false,
  controlledZoom, onZoomChange, controlledScroll, onScrollChange,
  controlledHeight, onHeightChange, onDoubleClickExpand,
  bboxColor = "#22d3ee", bboxBlink = false, newRegionIds, onDragStart, onDragEnd,
}: BBoxCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const syncBarRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const hoveredBboxRef = useRef<HTMLDivElement | null>(null);
  const [portalTooltip, setPortalTooltip] = useState<{ x: number; y: number; width: number; height: number; inst: InstText } | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const [tooltipMetrics, setTooltipMetrics] = useState<{ w: number; h: number } | null>(null);
  // preview-mode tooltip flip: when the hovered bbox is too close to the
  // card's top edge, the .bbox-preview-tooltip (positioned above the bbox)
  // would overflow the scroll container and get clipped.  This flag flips
  // it below the bbox instead, keeping it card-bound.
  const [previewTooltipFlip, setPreviewTooltipFlip] = useState(false);
  const [zoom, setZoom] = useState(1);
  const effectiveZoom = controlledZoom !== undefined ? controlledZoom : zoom;

  const setZoomBoth = useCallback((z: number) => {
    const clamped = Math.max(0.25, Math.min(4, z));
    setZoom(clamped);
    onZoomChange?.(clamped);
  }, [onZoomChange]);

  const [fitWidth, setFitWidth] = useState<number | null>(null);
  const [showSurfaces, setShowSurfaces] = useState(true);
  const [drag, setDrag] = useState<DragState>(null);
  const [drawRect, setDrawRect] = useState<BBox | null>(null);
  const [loupeOn, setLoupeOn] = useState(false);
  const [loupe, setLoupe] = useState<{ clientX: number; clientY: number; imgX: number; imgY: number } | null>(null);
  const [canvasHeight, setCanvasHeight] = useState<number | null>(null);
  const [canvasExpanded, setCanvasExpanded] = useState(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);
  const expandClickTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const DEFAULT_CANVAS_H = 400;
  const MIN_CANVAS_H = 180;
  const MAX_CANVAS_H = 800;
  const preExpandHeight = useRef<number>(MIN_CANVAS_H);
  const lastDragHandleDown = useRef(0);
  const lastExpandedH = useRef<number | null>(null);

  // display scale: rendered px per natural px. zoom=1 fits the container
  // width (small images stay natural size); zoom scales from there.
  const natural = imgNaturalSize;
  const baseWidth = natural
    ? Math.min(natural.width, fitWidth ?? natural.width)
    : null;

  // track the container's content width so zoom=1 means fit-to-width
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
    if (controlledScroll === undefined) return;
    const el = containerRef.current;
    if (!el) return;
    if (el.scrollLeft !== controlledScroll.x || el.scrollTop !== controlledScroll.y) {
      el.scrollLeft = controlledScroll.x;
      el.scrollTop = controlledScroll.y;
    }
  }, [controlledScroll]);

  // report scroll changes when linked
  useEffect(() => {
    if (!onScrollChange) return;
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => onScrollChange({ x: el.scrollLeft, y: el.scrollTop });
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [onScrollChange]);

  // sync horizontal scroll between container and sticky bottom scrollbar
  useEffect(() => {
    const el = containerRef.current;
    const bar = syncBarRef.current;
    if (!el || !bar) return;

    const sync = () => {
      if (bar.scrollLeft !== el.scrollLeft) bar.scrollLeft = el.scrollLeft;
    };
    const onBarScroll = () => {
      if (el.scrollLeft !== bar.scrollLeft) el.scrollLeft = bar.scrollLeft;
    };
    el.addEventListener("scroll", sync);
    bar.addEventListener("scroll", onBarScroll);
    return () => {
      el.removeEventListener("scroll", sync);
      bar.removeEventListener("scroll", onBarScroll);
    };
  }, []);

  // drag handle to expand/collapse canvas height (like title screen project list)
  // double-click is detected via mousedown timestamps (more reliable than onClick
  // which may not fire when mousedown adds document-level listeners)
  const onExpandDragStart = useCallback((e: React.MouseEvent) => {
    // double-click detection via timestamp
    const now = Date.now();
    if (now - lastDragHandleDown.current < 300) {
      lastDragHandleDown.current = 0;
      if (onDoubleClickExpand) {
        onDoubleClickExpand();
        return;
      }
      setCanvasHeight((h) => {
        if (h !== null) {
          lastExpandedH.current = h; // remember height before collapsing
          return null; // collapse to standard
        }
        // restore last expanded height, or use default
        return lastExpandedH.current ?? DEFAULT_CANVAS_H;
      });
      return;
    }
    lastDragHandleDown.current = now;

    dragStartY.current = e.clientY;
    dragStartHeight.current = containerRef.current?.parentElement?.offsetHeight ?? DEFAULT_CANVAS_H;
    const dynamicMax = MAX_CANVAS_H;
    let dragging = false;
    const onMove = (ev: MouseEvent) => {
      if (!dragging) {
        dragging = true;
        ev.preventDefault();
      }
      const delta = ev.clientY - dragStartY.current;
      const newHeight = Math.max(MIN_CANVAS_H, Math.min(dynamicMax, dragStartHeight.current + delta));
      if (controlledHeight !== undefined && onHeightChange) {
        onHeightChange(newHeight);
      } else {
        setCanvasHeight(newHeight);
      }
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [controlledHeight, onHeightChange, natural, baseWidth, effectiveZoom, preview, onDoubleClickExpand]);

  const onExpandClick = useCallback(() => {
    if (expandClickTimer.current) {
      clearTimeout(expandClickTimer.current);
      expandClickTimer.current = null;
      setCanvasExpanded((v) => { const nv = !v; onExpandToggle?.(nv); return nv; });
      return;
    }
    // single click: start a timer that expires without toggling.
    // a second click before expiry cancels it and toggles (double-click).
    expandClickTimer.current = setTimeout(() => {
      expandClickTimer.current = null;
    }, 250);
  }, [onExpandToggle]);

  const renderedW = natural && baseWidth ? baseWidth * effectiveZoom : null;
  const scale = natural && renderedW ? renderedW / natural.width : 1;
  const px = (v: number) => v * scale;

  // in preview mode, zoom controls only show when the image doesn't fit entirely
  const previewFitsEntirely = natural && baseWidth && fitWidth
    ? baseWidth <= fitWidth && natural.height * (baseWidth / natural.width) <= (preview ? 300 : 400)
    : true;

  // client coords → natural-image coords. returns FLOATS — callers round
  // only when committing a bbox, so drags don't accumulate rounding jitter.
  const toImgCoords = useCallback(
    (clientX: number, clientY: number, clamp = true): { x: number; y: number } | null => {
      const img = imgRef.current;
      if (!img || !natural) return null;
      const rect = img.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      let x = ((clientX - rect.left) / rect.width) * natural.width;
      let y = ((clientY - rect.top) / rect.height) * natural.height;
      if (clamp) {
        x = Math.min(Math.max(x, 0), natural.width);
        y = Math.min(Math.max(y, 0), natural.height);
      }
      return { x, y };
    },
    [natural]
  );

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    // Every branch below starts a drag, and a mouse-down-and-move is also how
    // the browser starts a text selection -- which painted a blue highlight
    // across the asset while a box was being resized. The select-none class
    // on the container stops that highlight being drawn; this stops the
    // selection existing at all, which is what keeps a stray copy from
    // picking up half the page.
    e.preventDefault();
    if (preview) {
      // in preview mode, drag pans the view (same as empty-canvas drag in edit mode)
      const el = containerRef.current;
      if (el) {
        setDrag({
          type: "pan",
          startClientX: e.clientX, startClientY: e.clientY,
          startScrollLeft: el.scrollLeft, startScrollTop: el.scrollTop,
        });
      }
      return;
    }
    if (drawMode) {
      // raw (unclamped) hit-test: a draw must start ON the image, not in
      // the container gutter around it
      const raw = toImgCoords(e.clientX, e.clientY, false);
      if (!raw || !natural) return;
      if (raw.x < 0 || raw.y < 0 || raw.x > natural.width || raw.y > natural.height) return;
      setDrag({ type: "draw", startX: raw.x, startY: raw.y });
    } else {
      // empty-canvas drag pans the view (bbox hits stopPropagation first);
      // spec: drag right → image slides left (view scrolls with the drag)
      onSelect(null);
      const el = containerRef.current;
      if (el) {
        setDrag({
          type: "pan",
          startClientX: e.clientX, startClientY: e.clientY,
          startScrollLeft: el.scrollLeft, startScrollTop: el.scrollTop,
        });
      }
    }
  }, [preview, drawMode, toImgCoords, natural, onSelect]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (loupeOn) {
      const raw = toImgCoords(e.clientX, e.clientY, false);
      if (
        raw && natural
        && raw.x >= 0 && raw.y >= 0
        && raw.x <= natural.width && raw.y <= natural.height
      ) {
        setLoupe({ clientX: e.clientX, clientY: e.clientY, imgX: raw.x, imgY: raw.y });
      } else {
        setLoupe(null);
      }
    }
    if (!drag) return;
    if (drag.type === "pan") {
      const el = containerRef.current;
      if (el) {
        el.scrollLeft = drag.startScrollLeft - (e.clientX - drag.startClientX);
        el.scrollTop = drag.startScrollTop - (e.clientY - drag.startClientY);
      }
      return;
    }
    const pt = toImgCoords(e.clientX, e.clientY);
    if (!pt) return;

    if (drag.type === "draw") {
      setDrawRect({
        x: Math.round(Math.min(drag.startX, pt.x)),
        y: Math.round(Math.min(drag.startY, pt.y)),
        width: Math.round(Math.abs(pt.x - drag.startX)),
        height: Math.round(Math.abs(pt.y - drag.startY)),
      });
    } else if (drag.type === "move") {
      const dx = pt.x - drag.startMouseX;
      const dy = pt.y - drag.startMouseY;
      onUpdateRegion(drag.id, {
        ...drag.origBBox,
        x: Math.round(drag.origBBox.x + dx),
        y: Math.round(drag.origBBox.y + dy),
      });
    } else if (drag.type === "resize") {
      const dx = pt.x - drag.startMouseX;
      const dy = pt.y - drag.startMouseY;
      const ob = drag.origBBox;
      const h = drag.handle;
      let { x, y, width, height } = ob;
      if (h.includes("n")) { y = ob.y + dy; height = ob.height - dy; }
      if (h.includes("s")) { height = ob.height + dy; }
      if (h.includes("w")) { x = ob.x + dx; width = ob.width - dx; }
      if (h.includes("e")) { width = ob.width + dx; }
      const nb: BBox = {
        x: Math.round(x), y: Math.round(y),
        width: Math.round(width), height: Math.round(height),
      };
      if (nb.width < 5) nb.width = 5;
      if (nb.height < 5) nb.height = 5;
      onUpdateRegion(drag.id, nb);
    }
  }, [drag, toImgCoords, onUpdateRegion, loupeOn, natural]);

  const onMouseUp = useCallback(() => {
    if (drag?.type === "draw" && drawRect && drawRect.width > 5 && drawRect.height > 5) {
      onAddRegion(drawRect);
    }
    if (drag && (drag.type === "move" || drag.type === "resize")) {
      onDragEnd?.();
    }
    setDrag(null);
    setDrawRect(null);
  }, [drag, drawRect, onAddRegion, onDragEnd]);

  useEffect(() => {
    if (!drag) return;
    const up = () => { onMouseUp(); };
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, [drag, onMouseUp]);

  const handleBboxMouseEnter = useCallback((id: string, el: HTMLDivElement, inst: InstText) => {
    onHover(id);
    hoveredBboxRef.current = el;
    const rect = el.getBoundingClientRect();
    setPortalTooltip({ x: rect.left, y: rect.top, width: rect.width, height: rect.height, inst });
  }, [onHover]);

  const handleBboxMouseLeave = useCallback(() => {
    onHover(null);
    hoveredBboxRef.current = null;
    setPortalTooltip(null);
  }, [onHover]);

  // keep the portal tooltip glued to the bbox while the canvas scrolls
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      const bbox = hoveredBboxRef.current;
      if (!bbox) return;
      const rect = bbox.getBoundingClientRect();
      setPortalTooltip((prev) => prev ? { ...prev, x: rect.left, y: rect.top, width: rect.width, height: rect.height } : prev);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // measure the actual rendered tooltip dimensions for viewport collision.
  // useLayoutEffect fires synchronously after render but before paint, so
  // there is no visible flash of the pre-measurement position.
  useLayoutEffect(() => {
    const el = tooltipRef.current;
    if (!el || !portalTooltip) { setTooltipMetrics(null); return; }
    const rect = el.getBoundingClientRect();
    setTooltipMetrics({ w: rect.width, h: rect.height });
  }, [portalTooltip]);

  // recompute on resize — bbox coordinates are viewport-relative (position: fixed)
  useEffect(() => {
    if (!portalTooltip) return;
    const onResize = () => {
      const bbox = hoveredBboxRef.current;
      if (!bbox) return;
      const rect = bbox.getBoundingClientRect();
      setPortalTooltip((prev) => prev ? { ...prev, x: rect.left, y: rect.top, width: rect.width, height: rect.height } : prev);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [portalTooltip]);

  // card-bound protection for the preview tooltip: when a region is hovered
  // (either by mouse-enter on the bbox or by hovering a row in the
  // RegionTable, which sets hoveredId externally), measure the bbox's
  // position relative to the scroll container.  If there isn't enough room
  // above the bbox for the tooltip (~30px: 6px gap + ~24px tooltip height),
  // flip it below so it stays inside the card instead of being clipped.
  useEffect(() => {
    if (!preview || !hoveredId) { setPreviewTooltipFlip(false); return; }
    const container = containerRef.current;
    if (!container) { setPreviewTooltipFlip(false); return; }
    const bboxEl = container.querySelector<HTMLElement>(`[data-region-id="${CSS.escape(hoveredId)}"]`);
    if (!bboxEl) { setPreviewTooltipFlip(false); return; }
    // offsetTop is relative to the offsetParent (the bbox overlay div),
    // which is positioned at 0,0 inside the scroll container — so this is
    // effectively the bbox's distance from the container's content top.
    // Add the container's scrollTop to account for scrolled-away bboxes.
    const bboxTopInContainer = bboxEl.offsetTop + container.scrollTop;
    const TOOLTIP_GAP = 6;
    const TOOLTIP_EST_HEIGHT = 24;
    setPreviewTooltipFlip(bboxTopInContainer < TOOLTIP_GAP + TOOLTIP_EST_HEIGHT);
  }, [preview, hoveredId]);

  const startDrag = (e: React.MouseEvent, state: Exclude<DragState, null>) => {
    e.stopPropagation();
    setDrag(state);
  };

  const resizeHandle = (inst: InstText, handle: string) => (
    <div
      className={`bbox-handle ${handle}`}
      onMouseDown={(e) => {
        const pt = toImgCoords(e.clientX, e.clientY);
        if (pt) {
          onDragStart?.();
          startDrag(e, {
          type: "resize", id: inst.id, handle,
          startMouseX: pt.x, startMouseY: pt.y, origBBox: inst.bounding_box,
        });
        }
      }}
    />
  );

  return (
    <div
      className="bezier-card soft-shadow relative flex min-h-[180px] flex-col rounded-lg bg-zinc-100 dark:bg-zinc-900"
      style={{
        ...(controlledHeight !== undefined
          ? { height: controlledHeight ?? (preview ? 300 : "100%"), transition: "height 0.3s ease-in-out" }
          : canvasHeight !== null
            ? { height: canvasHeight, transition: "height 0.3s ease-in-out" }
            : { height: preview ? 300 : "100%", transition: "height 0.3s ease-in-out" }),
        "--bbox-color": bboxColor,
      } as React.CSSProperties}
    >
    <div
      ref={containerRef}
      className="bbox-canvas-scroll relative flex-1 mr-6 select-none"
      style={{
        cursor: preview ? (drag?.type === "pan" ? "grabbing" : "grab") : drawMode ? "crosshair" : drag?.type === "pan" ? "grabbing" : "grab",
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseLeave={() => setLoupe(null)}
    >
      {imageUrl ? (
        <div style={{ position: "relative", display: "inline-block" }}>
          <img
            ref={imgRef}
            src={imageUrl}
            alt="asset"
            style={{
              display: "block",
              width: renderedW ?? "auto",
              height: "auto",
              // tailwind preflight sets img { max-width: 100% }, which
              // would clamp the image at container width while the
              // overlay keeps scaling — they must never diverge
              maxWidth: "none",
            }}
            onLoad={(e) => {
              const t = e.currentTarget;
              onImgLoad({ width: t.naturalWidth, height: t.naturalHeight });
            }}
            draggable={false}
          />
          {/* bbox overlay — positioned directly over the rendered image */}
          {natural && renderedW && (
            <div
              // blink is a canvas-level class, not per-region state: every
              // existing and newly added box then participates without a
              // single per-region timer or rerender
              className={bboxBlink ? "bbox-blink" : undefined}
              style={{
                position: "absolute",
                top: 0, left: 0,
                width: renderedW,
                height: natural.height * scale,
                pointerEvents: "none",
              }}
            >
              {/* scene-surface underlay (candidate text-bearing surfaces) */}
              {showSurfaces && sceneRegions?.map((r, i) => {
                // ``text_cluster`` is an implementation grouping, not a
                // material users can act on. Prefer Scene's conservative
                // material descriptor and only fall back to the routing type.
                const label = r.material ?? (r.texture ? `${r.texture.replace(/_/g, " ")} surface` : r.semantic_label.replace(/_/g, " "));
                if (r.polygon && r.polygon.length >= 3) {
                  const points = r.polygon.map(([x, y]) => `${px(x)},${px(y)}`).join(" ");
                  return (
                    <svg key={`sr-${i}`} className="absolute inset-0 overflow-visible" width={renderedW} height={natural.height * scale}>
                      <polygon points={points} fill={`${bboxColor}0D`} stroke={`${bboxColor}80`} strokeWidth="1" strokeDasharray="4 3" />
                      <text x={px(r.bbox.x + 3)} y={px(r.bbox.y + 14)} className="bbox-surface-label">{label}</text>
                    </svg>
                  );
                }
                return (
                  <div key={`sr-${i}`} style={{ position: "absolute", left: px(r.bbox.x), top: px(r.bbox.y), width: px(r.bbox.width), height: px(r.bbox.height), border: `1px dashed ${bboxColor}59`, borderRadius: 4, pointerEvents: "none" }}>
                    <span className="bbox-surface-label">{label}</span>
                  </div>
                );
              })}
              {manifest.map((inst, visualOrder) => {
                const isSel = inst.id === selectedId;
                const isHovered = inst.id === hoveredId;
                return (
                  <div
                    key={inst.id}
                    data-region-id={inst.id}
                    className={`${bboxClass(inst)} ${isSel ? "selected" : ""} ${isHovered && !isSel ? "hovered" : ""}`}
                    style={{
                      left: px(inst.bounding_box.x),
                      top: px(inst.bounding_box.y),
                      width: px(inst.bounding_box.width),
                      height: px(inst.bounding_box.height),
                      pointerEvents: drawMode ? "none" : "auto",
                    }}
                    onMouseDown={(e) => {
                      if (preview) {
                        e.stopPropagation();
                        onSelect(inst.id);
                        return;
                      }
                      if (!drawMode) {
                        e.stopPropagation();
                        onSelect(inst.id);
                        const pt = toImgCoords(e.clientX, e.clientY);
                        if (pt) {
                          onDragStart?.();
                          setDrag({
                          type: "move", id: inst.id,
                          startMouseX: pt.x, startMouseY: pt.y,
                          origBBox: inst.bounding_box,
                        });
                        }
                      }
                    }}
                    onMouseEnter={(e) => handleBboxMouseEnter(inst.id, e.currentTarget, inst)}
                    onMouseLeave={handleBboxMouseLeave}
                  >
                    <div className="bbox-corners" />
                    {/* Its own element on purpose: .bbox::before (scanlines)
                        and .bbox::after (scan sweep) are both already taken.
                        Boxes are keyed on inst.id, so this mounts exactly
                        once per arrival and rerenders never restart it. */}
                    {newRegionIds?.has(inst.id) && <div className="bbox-shimmer" aria-hidden="true" />}
                    {inst.reading_order !== null && (
                      <div
                        className="bbox-badge"
                        style={{ background: inst.confidence !== null && inst.confidence < 0.6 ? "#ef4444" : bboxColor }}
                        title={`Region ${inst.id} · reading order ${visualOrder + 1}`}
                      >
                        {/* Immutable rN identity and visual reading order are
                            different things.  Do not make VIEUX (r3) look
                            like r2 merely because it is second in its line. */}
                        {inst.id}
                      </div>
                    )}
                    {inst.confidence !== null && (
                      <div className="bbox-conf" style={{ color: confColor(inst.confidence) }}>
                        {(inst.confidence * 100).toFixed(0)}%
                      </div>
                    )}
                    {inst.dnt && <div className="bbox-dnt-badge">DNT</div>}
                    {preview && isHovered && (
                      <div className={`bbox-preview-tooltip subtext${previewTooltipFlip ? " flip-below" : ""}`}>
                        {inst.target_text || "No translation yet"}
                      </div>
                    )}
                    {/* capture-mode metadata tooltip is rendered via portal
                        at the end of this component so it escapes the
                        scroll container's overflow clipping.  See
                        portalTooltip state + createPortal below. */}
                    {isSel && !drawMode && !preview && (
                      <>
                        {resizeHandle(inst, "nw")}
                        {resizeHandle(inst, "ne")}
                        {resizeHandle(inst, "sw")}
                        {resizeHandle(inst, "se")}
                        {resizeHandle(inst, "n")}
                        {resizeHandle(inst, "s")}
                        {resizeHandle(inst, "w")}
                        {resizeHandle(inst, "e")}
                      </>
                    )}
                  </div>
                );
              })}
              {/* drawing preview */}
              {drawRect && (
                <div
                  className="bbox bbox-manual"
                  style={{
                    left: px(drawRect.x),
                    top: px(drawRect.y),
                    width: px(drawRect.width),
                    height: px(drawRect.height),
                    pointerEvents: "none",
                  }}
                >
                  <div className="bbox-corners" />
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="flex h-64 items-center justify-center text-zinc-600 text-sm">
          no image loaded
        </div>
      )}
      {/* magnifier loupe: circular halo following the cursor, magnifying
          the image area beneath it via a scaled CSS background. rendered
          through a portal to <body>: transformed ancestors (e.g. the
          step-fade tab animation) would otherwise re-base position:fixed
          and offset the lens from the cursor */}
      {loupeOn && loupe && imageUrl && natural && createPortal(
        <div
          style={{
            position: "fixed",
            left: loupe.clientX - LOUPE_RADIUS,
            top: loupe.clientY - LOUPE_RADIUS,
            width: LOUPE_RADIUS * 2,
            height: LOUPE_RADIUS * 2,
            borderRadius: "50%",
            border: `2px solid ${bboxColor}cc`,
            boxShadow: "0 0 12px rgba(0,0,0,0.6), inset 0 0 6px rgba(0,0,0,0.4)",
            backgroundImage: `url(${imageUrl})`,
            backgroundRepeat: "no-repeat",
            backgroundSize: `${natural.width * scale * LOUPE_MAGNIFICATION}px ${natural.height * scale * LOUPE_MAGNIFICATION}px`,
            backgroundPosition: `${LOUPE_RADIUS - loupe.imgX * scale * LOUPE_MAGNIFICATION}px ${LOUPE_RADIUS - loupe.imgY * scale * LOUPE_MAGNIFICATION}px`,
            backgroundColor: "#09090b",
            pointerEvents: "none",
            zIndex: 60,
          }}
        />,
        document.body
      )}
      </div>
      {/* synced horizontal scrollbar — pinned to the bottom of the canvas card */}
      {!preview && (
        <div
          ref={syncBarRef}
          style={{ overflowX: "auto", overflowY: "hidden", flexShrink: 0, marginRight: "1.5rem" }}
        >
          <div style={{ width: renderedW ?? 0, height: 1 }} />
        </div>
      )}
      {/* zoom controls — hidden in preview mode unless showPreviewControls and image doesn't fit */}
      {(!preview || (showPreviewControls && !previewFitsEntirely)) && (
      <div className="absolute bottom-12 left-2 flex gap-1 z-20 w-fit">
        <button onClick={() => {
          const el = containerRef.current;
          if (!el || !renderedW || !natural) { setZoomBoth(effectiveZoom - 0.25); return; }
          const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
          const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
          setZoomBoth(effectiveZoom - 0.25);
          requestAnimationFrame(() => {
            const ne = containerRef.current;
            if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; }
          });
        }} className="flex items-center rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
          {showPreviewControls ? <FaMinus size={10} /> : <TbCircleDashedMinus size={12} />}
        </button>
        <span className="rounded-sm bg-white px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">{(effectiveZoom * 100).toFixed(0)}%</span>
        <button onClick={() => {
          const el = containerRef.current;
          if (!el || !renderedW || !natural) { setZoomBoth(effectiveZoom + 0.25); return; }
          const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
          const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
          setZoomBoth(effectiveZoom + 0.25);
          requestAnimationFrame(() => {
            const ne = containerRef.current;
            if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; }
          });
        }} className="flex items-center rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
          {showPreviewControls ? <FaPlus size={10} /> : <TbCircleDashedPlus size={12} />}
        </button>
        <button onClick={() => setZoomBoth(1)} className="rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">fit</button>
        {!showPreviewControls && (
          <button
            onClick={() => { setLoupeOn((v) => !v); setLoupe(null); }}
            className={`flex items-center rounded-sm px-2 py-1 text-xs ${loupeOn ? "bg-cyan-900/70 text-cyan-300" : "bg-white text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
            title="Magnifier: hover the image to inspect detail under the cursor"
          >
            <TbZoomInFilled size={12} />
          </button>
        )}
        {!showPreviewControls && (sceneRegions?.length ?? 0) > 0 && (
          <button
            onClick={() => setShowSurfaces((s) => !s)}
            className={`rounded-sm px-2 py-1 text-xs ${showSurfaces ? "bg-cyan-900/70 text-cyan-300" : "bg-white text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-500 dark:hover:bg-zinc-700"}`}
            title="Toggle detected surface outlines"
          >
            surfaces
          </button>
        )}
      </div>
      )}
      {/* expansion drag handle — always visible so users can collapse too */}
      <div className="title-drag-handle shrink-0 flex items-center justify-center" onMouseDown={onExpandDragStart} title="drag to resize · double-click to toggle height" style={{ position: "relative" }}>
        <svg width="42" height="14" viewBox="0 0 42 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="0.5" y="1" width="41" height="4.5" rx="2.25" fill="currentColor" />
          <rect x="11" y="8" width="20" height="4.5" rx="2.25" fill="currentColor" />
        </svg>
        {canvasLabel && <span className="subtext text-xs text-zinc-500" style={{ position: "absolute", right: "0.75rem", top: "50%", transform: "translateY(-50%)" }}>{canvasLabel}</span>}
      </div>
      {/* horizontal expand handle — on the right side, hidden in preview mode */}
      {!preview && (
      <div
        className="title-drag-handle absolute right-0 top-1/2 -translate-y-1/2 shrink-0 z-20"
        style={{ cursor: "pointer", padding: "2px 4px" }}
        onClick={onExpandClick}
        title={canvasExpanded ? "double-click to return to standard" : "double-click to expand to full width"}
      >
        <svg width="14" height="42" viewBox="0 0 14 42" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="1" y="0.5" width="4.5" height="41" rx="2.25" fill="currentColor" />
          <rect x="8" y="11" width="4.5" height="20" rx="2.25" fill="currentColor" />
        </svg>
      </div>
      )}
      {/* capture-mode metadata tooltip — portaled to document.body so it
          escapes the scroll container's overflow clipping.  Positioned
          above the hovered bbox via fixed coordinates from
          getBoundingClientRect(), updated on scroll. */}
      {!preview && portalTooltip && (() => {
        const inst = portalTooltip.inst;
        const sp = inst.style_profile;
        const bp = inst.background_profile;
        const ch = inst.characteristics;
        const styleBits = [ch?.font_style].filter(Boolean).join(" · ");
        const bgBits = [
          bp?.material,
          bp?.material ? null : bp?.semantic_label?.replace(/_/g, " "),
          bp?.texture,
          ...(bp?.gradients ?? []),
        ].filter(Boolean).join(" · ");
        if (!sp?.color && !styleBits && !bgBits && !inst.detected_language && ch?.size == null) return null;
        // Use measured dimensions (from useLayoutEffect) for accurate
        // collision detection; fall back to CSS max-width / estimated
        // height on first render before measurement completes.
        const TOOLTIP_W = tooltipMetrics?.w ?? 260;
        const TOOLTIP_H = tooltipMetrics?.h ?? 80;
        const GAP = 6;
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const bboxCx = portalTooltip.x + portalTooltip.width / 2;
        const bboxCy = portalTooltip.y + portalTooltip.height / 2;
        // default: above, centered horizontally
        let placement: "above" | "below" | "left" | "right" = "above";
        const fitsAbove = portalTooltip.y - GAP - TOOLTIP_H >= 0;
        const fitsBelow = portalTooltip.y + portalTooltip.height + GAP + TOOLTIP_H <= vh;
        const fitsLeft = portalTooltip.x - GAP - TOOLTIP_W >= 0;
        const fitsRight = portalTooltip.x + portalTooltip.width + GAP + TOOLTIP_W <= vw;
        if (!fitsAbove && fitsBelow) {
          placement = "below";
        } else if (!fitsAbove && !fitsBelow) {
          // neither vertical fits — pick the horizontal direction with more room
          if (fitsRight) placement = "right";
          else if (fitsLeft) placement = "left";
          // if none fit, keep "above" (default) so it at least anchors to the bbox
        }
        let left: number, top: number, transform: string;
        if (placement === "above") {
          left = bboxCx;
          top = portalTooltip.y - GAP;
          transform = "translate(-50%, -100%)";
        } else if (placement === "below") {
          left = bboxCx;
          top = portalTooltip.y + portalTooltip.height + GAP;
          transform = "translateX(-50%)";
        } else if (placement === "left") {
          left = portalTooltip.x - GAP;
          top = bboxCy;
          transform = "translate(-100%, -50%)";
        } else { // right
          left = portalTooltip.x + portalTooltip.width + GAP;
          top = bboxCy;
          transform = "translateY(-50%)";
        }
        // edge clamping — keep the tooltip fully visible even when no
        // direction fits cleanly.  For vertical placements, clamp
        // horizontally; for horizontal placements, clamp vertically.
        const halfW = TOOLTIP_W / 2;
        const halfH = TOOLTIP_H / 2;
        if (placement === "above" || placement === "below") {
          if (left - halfW < 4) left = halfW + 4;
          if (left + halfW > vw - 4) left = vw - halfW - 4;
        } else {
          if (top - halfH < 4) top = halfH + 4;
          if (top + halfH > vh - 4) top = vh - halfH - 4;
        }
        return createPortal(
          <div
            ref={tooltipRef}
            className="bbox-meta-tooltip-portal subtext"
            style={{
              position: "fixed",
              left,
              top,
              transform,
              "--bbox-color": bboxColor,
            } as React.CSSProperties}
          >
            {inst.detected_language && (
              <div className="meta-row">
                <span className="meta-key">lang</span>
                <span>{inst.detected_language}</span>
                {inst.confidence !== null && (
                  <span className={`text-[10px] ${
                    inst.confidence >= 0.8 ? "text-emerald-400"
                    : inst.confidence >= 0.6 ? "text-amber-400"
                    : "text-red-400"
                  }`}>
                    {(inst.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            )}
            {(sp?.color || styleBits || ch?.size != null) && (
              <div className="meta-row">
                <span className="meta-key">style</span>
                {sp?.color && <span className="meta-swatch" style={{ background: sp.color }} />}
                {(styleBits || sp?.color) && <span>{styleBits || sp?.color}</span>}
                {ch?.size != null && (
                  <span className="text-[10px] text-[#2d8cf0]">~{ch.size}px</span>
                )}
              </div>
            )}
            {(bp?.dominant_color || bgBits) && (
              <div className="meta-row">
                <span className="meta-key">bg</span>
                {bp?.dominant_color && <span className="meta-swatch" style={{ background: bp.dominant_color }} />}
                <span>{bgBits || bp?.dominant_color}</span>
              </div>
            )}
          </div>,
          document.body
        );
      })()}
    </div>
  );
}
