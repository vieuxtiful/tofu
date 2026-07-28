import { KeyboardEvent, PointerEvent, useEffect, useRef } from "react";
import { BBox } from "./api";
import {
  IDENTITY_QUAD, Quad, denormaliseQuad, isUsableQuad, moveCorner, normaliseQuad,
} from "./perspective";

type CornerIndex = 0 | 1 | 2 | 3;

interface PerspectiveHandlesProps {
  bbox: BBox;
  quad: Quad | null;
  displayScale: number;
  imageRect: () => DOMRect | null;
  onGestureStart: (quad: Quad) => void;
  onGestureChange: (quad: Quad) => void;
  onGestureEnd: (quad: Quad, cancelled?: boolean) => void;
}

type Drag = {
  pointerId: number;
  index: CornerIndex;
  start: Quad;
  startPointer: [number, number];
  latest: Quad;
};

const LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"] as const;

export default function PerspectiveHandles({
  bbox, quad, displayScale, imageRect, onGestureStart, onGestureChange, onGestureEnd,
}: PerspectiveHandlesProps) {
  const active = quad ?? IDENTITY_QUAD;
  const corners = denormaliseQuad(active, bbox);
  const valid = isUsableQuad(corners);
  const dragRef = useRef<Drag | null>(null);

  const imagePoint = (clientX: number, clientY: number): [number, number] | null => {
    const rect = imageRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    const svg = document.querySelector<SVGSVGElement>("[data-perspective-overlay]");
    const viewBox = svg?.viewBox.baseVal;
    if (!viewBox || viewBox.width <= 0 || viewBox.height <= 0) return null;
    return [
      (clientX - rect.left) * viewBox.width / rect.width,
      (clientY - rect.top) * viewBox.height / rect.height,
    ];
  };

  const updateDrag = (clientX: number, clientY: number, shift: boolean, alt: boolean) => {
    const drag = dragRef.current;
    const point = imagePoint(clientX, clientY);
    if (!drag || !point) return;
    const dx = point[0] - drag.startPointer[0];
    const dy = point[1] - drag.startPointer[1];
    const axisLock = shift ? (Math.abs(dx) >= Math.abs(dy) ? "x" : "y") : undefined;
    const startPixels = denormaliseQuad(drag.start, bbox);
    drag.latest = normaliseQuad(
      moveCorner(startPixels, drag.index, point, { mirror: alt, axisLock }),
      bbox,
    );
    onGestureChange(drag.latest);
  };

  const finish = (cancelled = false) => {
    const drag = dragRef.current;
    if (!drag) return;
    dragRef.current = null;
    onGestureEnd(cancelled ? drag.start : drag.latest, cancelled);
  };

  useEffect(() => {
    const up = () => finish(false);
    const cancel = () => finish(true);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", cancel);
    return () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
    };
  });

  const begin = (event: PointerEvent<SVGCircleElement>, index: CornerIndex) => {
    event.preventDefault();
    event.stopPropagation();
    const point = imagePoint(event.clientX, event.clientY);
    if (!point) return;
    const start = active.map(([x, y]) => [x, y] as [number, number]) as Quad;
    dragRef.current = {
      pointerId: event.pointerId, index, start, startPointer: point, latest: start,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    onGestureStart(start);
  };

  const keyDown = (event: KeyboardEvent<SVGCircleElement>, index: CornerIndex) => {
    const delta = event.shiftKey ? 10 : 1;
    const offsets: Partial<Record<string, [number, number]>> = {
      ArrowLeft: [-delta, 0], ArrowRight: [delta, 0],
      ArrowUp: [0, -delta], ArrowDown: [0, delta],
    };
    if (event.key === "Escape" && dragRef.current) {
      event.preventDefault();
      finish(true);
      return;
    }
    const offset = offsets[event.key];
    if (!offset) return;
    event.preventDefault();
    event.stopPropagation();
    const pixels = denormaliseQuad(active, bbox);
    const [x, y] = pixels[index];
    const next = normaliseQuad(
      moveCorner(pixels, index, [x + offset[0], y + offset[1]], { mirror: event.altKey }),
      bbox,
    );
    onGestureStart(active);
    onGestureChange(next);
    onGestureEnd(next);
  };

  const scale = Math.max(displayScale, 0.01);
  const handleRadius = 8 / scale;
  const hitRadius = 14 / scale;
  const strokeWidth = valid ? 1.5 : 2;
  const stroke = valid ? "#06b6d4" : "#ef4444";
  const fill = valid ? "rgba(6,182,212,.08)" : "rgba(239,68,68,.10)";

  return (
    <g aria-label="Perspective corners">
      <polygon
        points={corners.map((point) => point.join(",")).join(" ")}
        fill={fill}
        stroke={stroke}
        strokeWidth={strokeWidth}
        vectorEffect="non-scaling-stroke"
        pointerEvents="none"
      />
      {corners.map(([cx, cy], index) => (
        <g key={LABELS[index]}>
          <circle
            cx={cx} cy={cy} r={hitRadius}
            fill="transparent"
            className="cursor-move"
            onPointerDown={(event) => begin(event, index as CornerIndex)}
            onPointerMove={(event) => {
              if (dragRef.current?.pointerId !== event.pointerId) return;
              event.preventDefault();
              updateDrag(event.clientX, event.clientY, event.shiftKey, event.altKey);
            }}
          />
          <circle
            cx={cx} cy={cy} r={handleRadius}
            fill="white" stroke={stroke} strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            className="cursor-move focus:outline-none"
            tabIndex={0}
            role="slider"
            aria-label={`${LABELS[index]} perspective corner`}
            onPointerDown={(event) => begin(event, index as CornerIndex)}
            onPointerMove={(event) => {
              if (dragRef.current?.pointerId !== event.pointerId) return;
              event.preventDefault();
              updateDrag(event.clientX, event.clientY, event.shiftKey, event.altKey);
            }}
            onKeyDown={(event) => keyDown(event, index as CornerIndex)}
          />
        </g>
      ))}
    </g>
  );
}
