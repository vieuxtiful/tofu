import { describe, expect, it, vi } from "vitest";
import { selectBatchFiles, uploadWithConcurrency } from "./assetBatch";

const files = (count: number) => Array.from(
  { length: count }, (_, index) => new File([String(index)], `${index}.png`),
);

describe("asset batch selection", () => {
  it("caps a project at thirty while preserving picker order", () => {
    const result = selectBatchFiles(files(8), 25);
    expect(result.accepted.map((file) => file.name)).toEqual(["0.png", "1.png", "2.png", "3.png", "4.png"]);
    expect(result.skipped).toHaveLength(3);
  });

  it("reports every success and failure without rolling back siblings", async () => {
    const settled = vi.fn();
    await uploadWithConcurrency(
      files(4),
      async (file) => {
        if (file.name === "2.png") throw new Error("bad file");
        return {
          asset_id: file.name, filename: file.name, asset_url: file.name,
          asset_info: { asset_type: "image", source: null, frame_count: 1, fps: null, duration: null },
        };
      },
      settled,
      2,
    );
    expect(settled).toHaveBeenCalledTimes(4);
    expect(settled.mock.calls.filter(([result]) => result.error)).toHaveLength(1);
  });
});
