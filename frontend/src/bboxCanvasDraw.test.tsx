import { describe, it, expect, vi, afterEach, beforeAll } from "vitest";
import { render, cleanup, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import BBoxCanvas from "./BBoxCanvas";
import type { InstText } from "./api";

afterEach(() => cleanup());

/** jsdom lays nothing out, so the canvas can map client coords to image
 *  coords only if the image reports a size. */
beforeAll(() => {
  // The overlay layer that carries the draw preview and the guided
  // confirm/reject controls is gated on `renderedW`, which derives from the
  // scroll container's clientWidth. jsdom reports 0 for every element, so
  // without this the layer never renders and DOM assertions test nothing.
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get() { return 400; },
  });
  Object.defineProperty(HTMLImageElement.prototype, "getBoundingClientRect", {
    configurable: true,
    value: () => ({
      x: 0, y: 0, left: 0, top: 0, right: 400, bottom: 200,
      width: 400, height: 200, toJSON: () => ({}),
    }),
  });
});

function canvas(over: Partial<React.ComponentProps<typeof BBoxCanvas>> = {}) {
  const props = {
    imageUrl: "/a.png", manifest: [], selectedId: null, hoveredId: null,
    onSelect: () => {}, onHover: () => {},
    onAddRegion: vi.fn(), onUpdateRegion: () => {},
    drawMode: true,
    imgNaturalSize: { width: 400, height: 200 },
    onImgLoad: () => {},
    ...over,
  } as React.ComponentProps<typeof BBoxCanvas>;
  const { container } = render(<BBoxCanvas {...props} />);
  return { container, props };
}

function drag(container: HTMLElement, from: [number, number], to: [number, number]) {
  const surface = container.querySelector(".bbox-canvas-scroll")!;
  fireEvent.mouseDown(surface, { clientX: from[0], clientY: from[1] });
  fireEvent.mouseMove(surface, { clientX: to[0], clientY: to[1] });
  fireEvent.mouseUp(surface);
}

describe("BBoxCanvas draw gestures", () => {
  it("adds a region for an ordinary drag", () => {
    const onAddRegion = vi.fn();
    const { container } = canvas({ onAddRegion });
    drag(container, [10, 10], [120, 60]);
    expect(onAddRegion).toHaveBeenCalledTimes(1);
  });

  it("reports a too-small box instead of dropping it silently", () => {
    // Before this the gesture simply vanished: no region, no message, and
    // in Guided the same Block still sitting there asking for a box.
    const onAddRegion = vi.fn();
    const onDrawTooSmall = vi.fn();
    const { container } = canvas({ onAddRegion, onDrawTooSmall });
    drag(container, [10, 10], [13, 12]);
    expect(onAddRegion).not.toHaveBeenCalled();
    expect(onDrawTooSmall).toHaveBeenCalledTimes(1);
  });

  it("stays silent on a stray click", () => {
    // A 0x0 rect is a click that missed a box, not a failed draw. Reporting
    // it would fire a toast on every deselect.
    const onDrawTooSmall = vi.fn();
    const { container } = canvas({ onDrawTooSmall });
    drag(container, [10, 10], [10, 10]);
    expect(onDrawTooSmall).not.toHaveBeenCalled();
  });

  it("refuses to start a draw while a previous box is saving", () => {
    // The old code let the draw start and then dropped it in the handler,
    // so a second box during the (multi-second) OCR round trip disappeared
    // with no explanation.
    const onAddRegion = vi.fn();
    const { container } = canvas({ onAddRegion, drawLocked: true });
    drag(container, [10, 10], [120, 60]);
    expect(onAddRegion).not.toHaveBeenCalled();
  });

  it("shows the tool as busy rather than disarmed while locked", () => {
    // Not `drawMode: false`: that hands the gesture to the panner, so the
    // user drags the image instead of being told to wait.
    const { container } = canvas({ drawLocked: true });
    const surface = container.querySelector(".bbox-canvas-scroll") as HTMLElement;
    expect(surface.style.cursor).toBe("progress");
  });

  it("accepts draws again once the save finishes", () => {
    const onAddRegion = vi.fn();
    const { container } = canvas({ onAddRegion, drawLocked: false });
    drag(container, [10, 10], [120, 60]);
    expect(onAddRegion).toHaveBeenCalledTimes(1);
  });
});

describe("Guided confirm/reject gate", () => {
  it("proposes rather than applies a drawn box", () => {
    // Guided persists the Block's DECLARED text onto whatever rectangle it
    // is given, so an accidental box becomes confidently wrong data with
    // nothing to contradict it. The verdict is the gate.
    const onAddRegion = vi.fn();
    const { container } = canvas({ onAddRegion, requireConfirm: true });
    drag(container, [10, 10], [120, 60]);
    expect(onAddRegion).not.toHaveBeenCalled();
    expect(container.querySelector("[data-testid='pending-region']")).toBeTruthy();
  });

  it("applies the box once confirmed", async () => {
    const onAddRegion = vi.fn();
    const { container } = canvas({ onAddRegion, requireConfirm: true });
    drag(container, [10, 10], [120, 60]);
    fireEvent.click(screen.getByLabelText("Confirm this region"));
    // The hand-off waits on the accept animation, so acceptance is seen
    // rather than inferred from a row appearing elsewhere.
    await waitFor(() => expect(onAddRegion).toHaveBeenCalledTimes(1));
  });

  it("discards the box on reject and frees the tool to redraw", () => {
    const onAddRegion = vi.fn();
    const { container } = canvas({ onAddRegion, requireConfirm: true });
    drag(container, [10, 10], [120, 60]);
    fireEvent.click(screen.getByLabelText("Reject and redraw this region"));
    expect(onAddRegion).not.toHaveBeenCalled();
    expect(container.querySelector("[data-testid='pending-region']")).toBeNull();
    // Nothing was created, so nothing is undone -- the tool is simply free.
    drag(container, [20, 20], [140, 80]);
    expect(container.querySelector("[data-testid='pending-region']")).toBeTruthy();
  });

  it("refuses a second draw while a verdict is outstanding", () => {
    // Otherwise the confirmation would silently apply to whichever rectangle
    // happened to be drawn last.
    const { container } = canvas({ requireConfirm: true });
    drag(container, [10, 10], [120, 60]);
    const first = container.querySelector("[data-testid='pending-region']")!.getAttribute("style");
    drag(container, [200, 100], [300, 160]);
    expect(container.querySelector("[data-testid='pending-region']")!.getAttribute("style")).toBe(first);
  });

  it("flips the controls inside the canvas near the right edge", () => {
    // The image bounds the canvas horizontally, so controls on a box drawn
    // hard against the right edge would be cut off.
    const { container } = canvas({ requireConfirm: true });
    drag(container, [340, 10], [399, 60]);
    const controls = container.querySelector(".bbox-confirm");
    expect(controls?.getAttribute("data-side")).toBe("left");
  });

  it("keeps the controls on the right when there is room", () => {
    const { container } = canvas({ requireConfirm: true });
    drag(container, [10, 10], [120, 60]);
    expect(container.querySelector(".bbox-confirm")?.getAttribute("data-side")).toBe("right");
  });

  it("holds the proposal on screen until the real region arrives", () => {
    // The proposal used to be cleared on a timer, which emptied the canvas
    // for the length of the round trip: the box appeared to flicker out and
    // back, or to have not been confirmed at all.
    const onAddRegion = vi.fn();
    const props = {
      imageUrl: "/a.png", manifest: [] as InstText[], selectedId: null, hoveredId: null,
      onSelect: () => {}, onHover: () => {}, onAddRegion, onUpdateRegion: () => {},
      drawMode: true, imgNaturalSize: { width: 400, height: 200 },
      onImgLoad: () => {}, requireConfirm: true,
    } as React.ComponentProps<typeof BBoxCanvas>;
    const { container, rerender } = render(<BBoxCanvas {...props} />);
    drag(container, [10, 10], [120, 60]);
    fireEvent.click(screen.getByLabelText("Confirm this region"));
    expect(onAddRegion).toHaveBeenCalledTimes(1);
    // Still drawn: nothing has replaced it yet.
    expect(container.querySelector("[data-testid='pending-region']")).toBeTruthy();

    rerender(<BBoxCanvas {...props} manifest={[{
      id: "r1", bounding_box: { x: 10, y: 10, width: 110, height: 50 },
      text: "SORTIE", target_text: null, confidence: null, reading_order: 0,
    } as unknown as InstText]} />);
    expect(container.querySelector("[data-testid='pending-region']")).toBeNull();
  });

  it("shimmers a confirmed region once, not twice", () => {
    // Two passes announce a region that arrived WITHOUT being asked for.
    const props = {
      imageUrl: "/a.png", selectedId: null, hoveredId: null,
      onSelect: () => {}, onHover: () => {}, onAddRegion: () => {}, onUpdateRegion: () => {},
      drawMode: true, imgNaturalSize: { width: 400, height: 200 },
      onImgLoad: () => {}, requireConfirm: true,
      newRegionIds: new Set(["r1"]),
      manifest: [{
        id: "r1", bounding_box: { x: 10, y: 10, width: 110, height: 50 },
        text: "SORTIE", target_text: null, confidence: null, reading_order: 0,
      }] as unknown as InstText[],
    } as React.ComponentProps<typeof BBoxCanvas>;
    const { container } = render(<BBoxCanvas {...props} />);
    expect(container.querySelector(".bbox-shimmer.once")).toBeTruthy();
    expect(container.querySelector(".bbox-arriving")).toBeTruthy();
  });

  it("keeps the two-pass shimmer outside Guided", () => {
    const props = {
      imageUrl: "/a.png", selectedId: null, hoveredId: null,
      onSelect: () => {}, onHover: () => {}, onAddRegion: () => {}, onUpdateRegion: () => {},
      drawMode: true, imgNaturalSize: { width: 400, height: 200 },
      onImgLoad: () => {}, newRegionIds: new Set(["r1"]),
      manifest: [{
        id: "r1", bounding_box: { x: 10, y: 10, width: 110, height: 50 },
        text: "SORTIE", target_text: null, confidence: null, reading_order: 0,
      }] as unknown as InstText[],
    } as React.ComponentProps<typeof BBoxCanvas>;
    const { container } = render(<BBoxCanvas {...props} />);
    expect(container.querySelector(".bbox-shimmer")).toBeTruthy();
    expect(container.querySelector(".bbox-shimmer.once")).toBeNull();
    expect(container.querySelector(".bbox-arriving")).toBeNull();
  });

  it("does not gate Auto or Manual draws", () => {
    const onAddRegion = vi.fn();
    const { container } = canvas({ onAddRegion });
    drag(container, [10, 10], [120, 60]);
    expect(onAddRegion).toHaveBeenCalledTimes(1);
    expect(container.querySelector("[data-testid='pending-region']")).toBeNull();
  });
});
