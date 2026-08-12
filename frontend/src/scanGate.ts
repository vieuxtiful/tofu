/** What the upload language scan is allowed to stop the user doing.
 *
 * Extracted because the rule is one boolean guarding five call sites, and it
 * was wrong at all five: `scan.status !== "passed"` treated the IN-FLIGHT
 * state as a blocking verdict, so the Stepper refused to advance and the
 * Capture control was not even rendered while the scan ran. The scan is a
 * full `cicerone.detect()` on the server — CRAFT plus EasyOCR over the whole
 * image, measured at 146s on a 3200×3200 upload — so "still looking" locked
 * the user out of their own asset for over two minutes to produce a boolean.
 *
 * A module rather than an inline expression so the distinction is testable:
 * nothing in the repo renders App, and this rule is exactly the kind that
 * looks obviously right while being wrong in a state nobody rendered.
 */

export type ScanStatus = "scanning" | "passed" | "mismatch";

export type ScanLike = {
  assetId: string;
  projectId?: string;
  status: ScanStatus;
} | null;

/** Blocks navigation and capture.
 *
 * ONLY a confirmed mismatch. It is a real answer, and continuing into a
 * capture whose asset is in the wrong language wastes the entire run. An
 * unfinished scan is not an answer and must never be treated as one.
 */
export function scanBlocksWorkflow(scan: ScanLike): boolean {
  return scan?.status === "mismatch";
}

/** Shown, but never in the way. */
export function scanIsAdvisory(scan: ScanLike): boolean {
  return scan?.status === "scanning";
}

/** Whether a scan result that has just arrived still describes what is on
 *  screen.
 *
 *  Asset AND project. Guarding on the asset alone still lets a scan started
 *  under project A lock a source language on project B, or raise a mismatch
 *  dialog about an asset the user has already navigated away from — the scan
 *  outlives several user actions by design, because it takes minutes.
 */
export function scanResultIsCurrent(
  result: { assetId: string; projectId: string },
  current: { assetId: string | null | undefined; projectId: string | null | undefined },
): boolean {
  return Boolean(
    current.assetId
    && current.projectId
    && result.assetId === current.assetId
    && result.projectId === current.projectId,
  );
}
