import { describe, it, expect, vi } from "vitest";
import { scanBlocksWorkflow, scanIsAdvisory, scanResultIsCurrent } from "./scanGate";
import { scanAssetLanguage } from "./api";

const scanning = { assetId: "a1", status: "scanning" as const };
const passed = { assetId: "a1", status: "passed" as const };
const mismatch = { assetId: "a1", status: "mismatch" as const };

describe("what the upload language scan may block", () => {
  it("does NOT block while the scan is still running", () => {
    // The defect this file exists for. The rule was `status !== "passed"`,
    // so an in-flight scan gated the Stepper and unrendered the Capture
    // control -- locking the user out of their own asset for the ~146s the
    // scan takes on a large upload, to produce one boolean.
    expect(scanBlocksWorkflow(scanning)).toBe(false);
  });

  it("blocks on a confirmed mismatch", () => {
    // A real answer, and continuing wastes the whole capture run.
    expect(scanBlocksWorkflow(mismatch)).toBe(true);
  });

  it("does not block once the scan passes", () => {
    expect(scanBlocksWorkflow(passed)).toBe(false);
  });

  it("does not block when no scan has run", () => {
    expect(scanBlocksWorkflow(null)).toBe(false);
  });

  it("still reports an in-flight scan as advisory", () => {
    // Shown, but never in the way.
    expect(scanIsAdvisory(scanning)).toBe(true);
    expect(scanIsAdvisory(mismatch)).toBe(false);
  });
});

describe("Capture stays available across an unresolved scan", () => {
  it("is enabled for the whole time the request is pending", async () => {
    // Holds the request unresolved, exactly as a 146s server-side OCR pass
    // would, and asserts the gate never closes while it hangs.
    let release!: (value: unknown) => void;
    const pending = new Promise((resolve) => { release = resolve; });
    const fetchMock = vi.fn(() => pending as Promise<Response>);
    vi.stubGlobal("fetch", fetchMock);

    const state = { assetId: "a1", status: "scanning" as const };
    const inFlight = scanAssetLanguage("a1").catch(() => null);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    // ...and while it hangs, the workflow is open.
    expect(scanBlocksWorkflow(state)).toBe(false);

    release(new Response("{}", { status: 200 }));
    await inFlight;
    expect(scanBlocksWorkflow(state)).toBe(false);
    vi.unstubAllGlobals();
  });

  it("passes an abort signal so a real capture can abandon it", async () => {
    // Two full OCR passes over one asset compete for the same CPU, so the
    // scan has to be abandonable rather than merely ignored.
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();
    await scanAssetLanguage("a1", controller.signal).catch(() => null);
    expect(fetchMock.mock.calls[0]?.[1]?.signal).toBe(controller.signal);
    vi.unstubAllGlobals();
  });
});

describe("late scan results are matched to what is on screen", () => {
  const started = { assetId: "a1", projectId: "p1" };

  it("accepts a result for the current asset and project", () => {
    expect(scanResultIsCurrent(started, { assetId: "a1", projectId: "p1" })).toBe(true);
  });

  it("rejects a result for an asset the user has left", () => {
    expect(scanResultIsCurrent(started, { assetId: "a2", projectId: "p1" })).toBe(false);
  });

  it("rejects a result for a project the user has left", () => {
    // Asset alone is not enough: a scan started under one project could
    // otherwise lock a source language on a different one.
    expect(scanResultIsCurrent(started, { assetId: "a1", projectId: "p2" })).toBe(false);
  });

  it("rejects a result when nothing is loaded any more", () => {
    expect(scanResultIsCurrent(started, { assetId: null, projectId: null })).toBe(false);
  });
});
