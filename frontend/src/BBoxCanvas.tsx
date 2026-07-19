import { useCallback, useEffect, useRef, useState } from "react";
import { TbZoomInFilled, TbCircleDashedPlus, TbCircleDashedMinus } from "react-icons/tb";
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
  preview = false,
}: BBoxCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const syncBarRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [zoom, setZoom] = useState(1);
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
  const MIN_CANVAS_H = 200;
  const MAX_CANVAS_H = 800;
  const preExpandHeight = useRef<number>(MIN_CANVAS_H);

  // track the container's content width so zoom=1 means fit-to-width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setFitWidth(el.clientWidth));
    ro.observe(el);
    setFitWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

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
  const onExpandDragStart = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    dragStartY.current = e.clientY;
    dragStartHeight.current = containerRef.current?.parentElement?.offsetHeight ?? DEFAULT_CANVAS_H;
    const onMove = (ev: MouseEvent) => {
      const delta = ev.clientY - dragStartY.current;
      const newHeight = Math.max(MIN_CANVAS_H, Math.min(MAX_CANVAS_H, dragStartHeight.current + delta));
      setCanvasHeight(newHeight);
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);

  const onExpandClick = useCallback(() => {
    if (expandClickTimer.current) {
      clearTimeout(expandClickTimer.current);
      expandClickTimer.current = null;
      setCanvasExpanded(false);
      onExpandToggle?.(false);
      return;
    }
    expandClickTimer.current = setTimeout(() => {
      expandClickTimer.current = null;
      setCanvasExpanded((v) => { const nv = !v; onExpandToggle?.(nv); return nv; });
    }, 250);
  }, [onExpandToggle]);

  // display scale: rendered px per natural px. zoom=1 fits the container
  // width (small images stay natural size); zoom scales from there.
  const natural = imgNaturalSize;
  const baseWidth = natural
    ? Math.min(natural.width, fitWidth ?? natural.width)
    : null;
  const renderedW = natural && baseWidth ? baseWidth * zoom : null;
  const scale = natural && renderedW ? renderedW / natural.width : 1;
  const px = (v: number) => v * scale;

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
    if (preview) return;
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
  }, [drawMode, toImgCoords, natural, onSelect]);

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
    setDrag(null);
    setDrawRect(null);
  }, [drag, drawRect, onAddRegion]);

  useEffect(() => {
    if (!drag) return;
    const up = () => { onMouseUp(); };
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, [drag, onMouseUp]);

  const handleBboxMouseEnter = useCallback((id: string) => {
    onHover(id);
  }, [onHover]);

  const handleBboxMouseLeave = useCallback(() => {
    onHover(null);
  }, [onHover]);

  const startDrag = (e: React.MouseEvent, state: Exclude<DragState, null>) => {
    e.stopPropagation();
    setDrag(state);
  };

  const resizeHandle = (inst: InstText, handle: string) => (
    <div
      className={`bbox-handle ${handle}`}
      onMouseDown={(e) => {
        const pt = toImgCoords(e.clientX, e.clientY);
        if (pt) startDrag(e, {
          type: "resize", id: inst.id, handle,
          startMouseX: pt.x, startMouseY: pt.y, origBBox: inst.bounding_box,
        });
      }}
    />
  );

  return (
    <div
      className="bezier-card soft-shadow relative flex min-h-[200px] flex-col rounded-lg bg-zinc-100 dark:bg-zinc-900"
      style={canvasHeight !== null ? { height: canvasHeight, transition: "height 0.3s ease-in-out" } : { height: preview ? 300 : "100%", transition: "height 0.3s ease-in-out" }}
    >
    <div
      ref={containerRef}
      className="bbox-canvas-scroll relative flex-1 mr-6"
      style={{
        cursor: preview ? "default" : drawMode ? "crosshair" : drag?.type === "pan" ? "grabbing" : "grab",
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
              style={{
                position: "absolute",
                top: 0, left: 0,
                width: renderedW,
                height: natural.height * scale,
                pointerEvents: "none",
              }}
            >
              {/* scene-surface underlay (candidate text-bearing surfaces) */}
              {showSurfaces && sceneRegions?.map((r, i) => (
                <div
                  key={`sr-${i}`}
                  style={{
                    position: "absolute",
                    left: px(r.bbox.x),
                    top: px(r.bbox.y),
                    width: px(r.bbox.width),
                    height: px(r.bbox.height),
                    border: "1px dashed rgba(34, 211, 238, 0.35)",
                    borderRadius: 4,
                    pointerEvents: "none",
                  }}
                >
                  <span className="bbox-surface-label">
                    {r.semantic_label}
                  </span>
                </div>
              ))}
              {manifest.map((inst) => {
                const isSel = inst.id === selectedId;
                const isHovered = inst.id === hoveredId;
                return (
                  <div
                    key={inst.id}
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
                        if (pt) setDrag({
                          type: "move", id: inst.id,
                          startMouseX: pt.x, startMouseY: pt.y,
                          origBBox: inst.bounding_box,
                        });
                      }
                    }}
                    onMouseEnter={() => handleBboxMouseEnter(inst.id)}
                    onMouseLeave={handleBboxMouseLeave}
                  >
                    <div className="bbox-corners" />
                    {inst.reading_order !== null && (
                      <div
                        className="bbox-badge"
                        style={{ background: inst.confidence !== null && inst.confidence < 0.6 ? "#ef4444" : "#22d3ee" }}
                      >
                        {(inst.reading_order ?? 0) + 1}
                      </div>
                    )}
                    {inst.confidence !== null && (
                      <div className="bbox-conf" style={{ color: confColor(inst.confidence) }}>
                        {(inst.confidence * 100).toFixed(0)}%
                      </div>
                    )}
                    {inst.dnt && <div className="bbox-dnt-badge">DNT</div>}
                    {preview && isHovered && (
                      <div className="bbox-preview-tooltip subtext">
                        {inst.target_text || "No translation yet"}
                      </div>
                    )}
                    {/* capture-mode metadata tooltip: scene/typography
                        enrichment (style + background) detected at capture */}
                    {!preview && isHovered && !drawMode && !drag && (() => {
                      const sp = inst.style_profile;
                      const bp = inst.background_profile;
                      const ch = inst.characteristics;
                      const styleBits = [
                        ch?.font_style,
                        ch?.size != null ? `~${ch.size}px` : null,
                      ].filter(Boolean).join(" · ");
                      const bgBits = [
                        bp?.semantic_label,
                        bp?.texture,
                        ...(bp?.gradients ?? []),
                      ].filter(Boolean).join(" · ");
                      if (!sp?.color && !styleBits && !bgBits && !inst.detected_language) return null;
                      return (
                        <div className="bbox-meta-tooltip subtext">
                          {inst.detected_language && (
                            <div className="meta-row">
                              <span className="meta-key">lang</span>
                              <span>{inst.detected_language}
                                {inst.confidence !== null && ` (${(inst.confidence * 100).toFixed(0)}%)`}
                              </span>
                            </div>
                          )}
                          {(sp?.color || styleBits) && (
                            <div className="meta-row">
                              <span className="meta-key">style</span>
                              {sp?.color && <span className="meta-swatch" style={{ background: sp.color }} />}
                              <span>{styleBits || sp?.color}</span>
                            </div>
                          )}
                          {(bp?.dominant_color || bgBits) && (
                            <div className="meta-row">
                              <span className="meta-key">bg</span>
                              {bp?.dominant_color && <span className="meta-swatch" style={{ background: bp.dominant_color }} />}
                              <span>{bgBits || bp?.dominant_color}</span>
                            </div>
                          )}
                        </div>
                      );
                    })()}
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
          the image area beneath it via a scaled CSS background */}
      {loupeOn && loupe && imageUrl && natural && (
        <div
          style={{
            position: "fixed",
            left: loupe.clientX - LOUPE_RADIUS,
            top: loupe.clientY - LOUPE_RADIUS,
            width: LOUPE_RADIUS * 2,
            height: LOUPE_RADIUS * 2,
            borderRadius: "50%",
            border: "2px solid rgba(34, 211, 238, 0.8)",
            boxShadow: "0 0 12px rgba(0,0,0,0.6), inset 0 0 6px rgba(0,0,0,0.4)",
            backgroundImage: `url(${imageUrl})`,
            backgroundRepeat: "no-repeat",
            backgroundSize: `${natural.width * scale * LOUPE_MAGNIFICATION}px ${natural.height * scale * LOUPE_MAGNIFICATION}px`,
            backgroundPosition: `${LOUPE_RADIUS - loupe.imgX * scale * LOUPE_MAGNIFICATION}px ${LOUPE_RADIUS - loupe.imgY * scale * LOUPE_MAGNIFICATION}px`,
            backgroundColor: "#09090b",
            pointerEvents: "none",
            zIndex: 60,
          }}
        />
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
      {/* zoom controls — hidden in preview mode */}
      {!preview && (
      <div className="absolute bottom-8 left-2 flex gap-1 z-20 w-fit">
        <button onClick={() => {
          const el = containerRef.current;
          if (!el || !renderedW || !natural) { setZoom((z) => Math.max(0.25, z - 0.25)); return; }
          const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
          const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
          setZoom((z) => Math.max(0.25, z - 0.25));
          requestAnimationFrame(() => {
            const ne = containerRef.current;
            if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; }
          });
        }} className="flex items-center rounded bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"><TbCircleDashedMinus size={12} /></button>
        <span className="rounded bg-white px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">{(zoom * 100).toFixed(0)}%</span>
        <button onClick={() => {
          const el = containerRef.current;
          if (!el || !renderedW || !natural) { setZoom((z) => Math.min(4, z + 0.25)); return; }
          const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
          const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
          setZoom((z) => Math.min(4, z + 0.25));
          requestAnimationFrame(() => {
            const ne = containerRef.current;
            if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; }
          });
        }} className="flex items-center rounded bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"><TbCircleDashedPlus size={12} /></button>
        <button onClick={() => setZoom(1)} className="rounded bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">fit</button>
        <button
          onClick={() => { setLoupeOn((v) => !v); setLoupe(null); }}
          className={`flex items-center rounded px-2 py-1 text-xs ${loupeOn ? "bg-cyan-900/70 text-cyan-300" : "bg-white text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
          title="Magnifier: hover the image to inspect detail under the cursor"
        >
          <TbZoomInFilled size={12} />
        </button>
        {(sceneRegions?.length ?? 0) > 0 && (
          <button
            onClick={() => setShowSurfaces((s) => !s)}
            className={`rounded px-2 py-1 text-xs ${showSurfaces ? "bg-cyan-900/70 text-cyan-300" : "bg-white text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-500 dark:hover:bg-zinc-700"}`}
            title="Toggle detected surface outlines"
          >
            surfaces
          </button>
        )}
      </div>
      )}
      {/* expansion drag handle — always visible so users can collapse too */}
      <div className="title-drag-handle shrink-0" onMouseDown={onExpandDragStart} onDoubleClick={() => setCanvasHeight((h) => {
        if (h !== null && h < DEFAULT_CANVAS_H) {
          preExpandHeight.current = h;
          return DEFAULT_CANVAS_H;
        }
        return preExpandHeight.current;
      })} title="drag to resize · double-click to toggle height">
        <svg width="42" height="14" viewBox="0 0 42 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="0.5" y="1" width="41" height="4.5" rx="2.25" fill="currentColor" />
          <rect x="11" y="8" width="20" height="4.5" rx="2.25" fill="currentColor" />
        </svg>
      </div>
      {/* horizontal expand handle — on the right side, vertical version of the drag handle */}
      <div
        className="title-drag-handle absolute right-0 top-1/2 -translate-y-1/2 shrink-0 z-20"
        style={{ cursor: "pointer", padding: "2px 4px" }}
        onClick={onExpandClick}
        title={canvasExpanded ? "collapse to grid · double-click to snap back" : "expand to full width · double-click to snap back"}
      >
        <svg width="14" height="42" viewBox="0 0 14 42" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="1" y="0.5" width="4.5" height="41" rx="2.25" fill="currentColor" />
          <rect x="8" y="11" width="4.5" height="20" rx="2.25" fill="currentColor" />
        </svg>
      </div>
    </div>
  );
}
