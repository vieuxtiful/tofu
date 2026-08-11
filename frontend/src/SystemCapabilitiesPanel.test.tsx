import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import SystemCapabilitiesPanel from "./SystemCapabilitiesPanel";
import type { SystemCapabilities } from "./api";

vi.mock("./api", () => ({
  fetchSystemCapabilities: vi.fn(),
}));

import { fetchSystemCapabilities } from "./api";

function makePayload(): SystemCapabilities {
  const mk = (id: string, label: string, ready = true) => ({
    id,
    label,
    available: ready,
    ready,
    version: null,
    reason: null,
  });
  return {
    schema_version: "1.0",
    build: { app: "ToFU", version: "0.0.0", python: "3.13", platform: "win32" },
    compute: { gpu: mk("cuda", "GPU", false) },
    ocr: {
      providers: [
        mk("easyocr", "Primary OCR"),
        mk("paddleocr", "Verifier OCR (CJK)", false),
      ],
    },
    scene: { active_backend: "classical_cv", providers: [mk("classical_cv", "Contour vision")] },
    inpainting: { providers: [mk("analytic", "Analytic inpaint")] },
    shaping: { advanced_available: true, providers: [mk("harfbuzz", "Complex-script shaping")] },
    semantics: { providers: [mk("deterministic_layout", "Deterministic layout")] },
    translation: { active_provider: "null", providers: [mk("null", "None configured", false)] },
    fonts: mk("font_library", "Fonts"),
    video: mk("video_pipeline", "Video"),
  };
}

describe("SystemCapabilitiesPanel", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows friendly labels instead of raw ids", async () => {
    vi.mocked(fetchSystemCapabilities).mockResolvedValue(makePayload());
    render(<SystemCapabilitiesPanel onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Primary OCR")).toBeTruthy());
    expect(screen.getByText("Verifier OCR (CJK)")).toBeTruthy();
    expect(screen.getByText("Contour vision")).toBeTruthy();
    expect(screen.getByText("None configured")).toBeTruthy();
    // raw ids must not be shown as the row label
    expect(screen.queryByText("easyocr")).toBeNull();
    expect(screen.queryByText("paddleocr")).toBeNull();
    expect(screen.queryByText("classical_cv")).toBeNull();
  });

  it("falls back to id when label is absent", async () => {
    const payload = makePayload();
    // strip label from one provider to exercise the fallback
    const ocr = payload.ocr.providers[0] as Record<string, unknown>;
    delete ocr.label;
    vi.mocked(fetchSystemCapabilities).mockResolvedValue(payload);
    render(<SystemCapabilitiesPanel onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("easyocr")).toBeTruthy());
  });
});
