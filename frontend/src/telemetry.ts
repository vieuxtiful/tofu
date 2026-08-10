// 🍢 telemetry — what the reviewer did, derived from what they already do
//
// Removing the scene-membership veto from detection recovered real text and
// grew the candidate set by roughly a third. Whether that trade is worth it
// is a question about REVIEWER TIME, and nothing the pipeline computes can
// answer it — mean IoU, garbage fraction and edit distance all describe the
// output, not the cost of checking it.
//
// So the outcomes have to come from here. The design rule is that they are
// DERIVED, never asked for: a reviewer who has to grade each region is doing
// data entry for a ranker that does not exist yet, and would stop. Deleting a
// region already says it was junk; dragging a handle already says the box was
// wrong; overtyping already says the recognizer was. Those gestures exist, and
// this only listens to them.
//
//   accepted          left untouched through to save/export
//   rejected          region deleted
//   edited_geometry   handles dragged  — localization failed
//   edited_text       transcript overtyped — recognition failed
//   merged            regions joined — over-segmentation upstream
//
// Fire-and-forget by construction. Telemetry must never block, slow, or fail
// an edit: a dropped event costs one row in a training set, a thrown one
// costs the user their action.

import { postReviewEvents } from "./api";

export type ReviewAction =
  | "accepted"
  | "rejected"
  | "edited_geometry"
  | "edited_text"
  | "merged";

export interface ReviewEvent {
  candidate_id: string;
  asset_id: string;
  action_type: ReviewAction;
  project_id?: string | null;
  session_id?: string;
  dwell_ms?: number;
  modified_bbox?: number[];
  /** snapshot of InstText.review_features as it was when reviewed */
  features?: Record<string, unknown> | null;
  /** snapshot of InstText.scene_eligibility */
  eligibility?: Record<string, unknown> | null;
  created_at?: number;
}

/** One id per page load, so a session's actions can be grouped. */
const SESSION_ID = `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

// Batched: emit happens on ordinary editing gestures, and a request per
// gesture would put telemetry in the interaction path.
const FLUSH_MS = 4000;
const FLUSH_AT = 25;

let queue: ReviewEvent[] = [];
let timer: ReturnType<typeof setTimeout> | null = null;

/** When a candidate was first shown, for dwell time. */
const firstSeen = new Map<string, number>();

export function markSeen(candidateId: string): void {
  if (!firstSeen.has(candidateId)) firstSeen.set(candidateId, Date.now());
}

export function flush(): void {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
  if (!queue.length) return;
  const batch = queue;
  queue = [];
  // Deliberately not awaited: a failed flush drops rows from a training set,
  // which is a cost worth paying to keep the editor responsive.
  void postReviewEvents(batch).catch(() => {});
}

export function emit(
  action: ReviewAction,
  candidateId: string,
  assetId: string,
  extra: Partial<ReviewEvent> = {},
): void {
  if (!candidateId || !assetId) return;
  const seen = firstSeen.get(candidateId);
  queue.push({
    candidate_id: candidateId,
    asset_id: assetId,
    action_type: action,
    session_id: SESSION_ID,
    dwell_ms: seen ? Date.now() - seen : undefined,
    created_at: Date.now() / 1000,
    ...extra,
  });
  if (queue.length >= FLUSH_AT) {
    flush();
    return;
  }
  if (!timer) timer = setTimeout(flush, FLUSH_MS);
}

/**
 * Record the regions a reviewer left alone.
 *
 * `accepted` is the only outcome with no gesture behind it — it is the
 * absence of one — so it can only be derived at a point where the reviewer
 * has declared they are done. Call on save or export with the ids that
 * survived and the ids that were touched; the difference is the accepted set.
 *
 * Without this the dataset would contain only complaints, and a ranker
 * trained on complaints alone learns that every candidate is bad.
 */
export function emitAccepted(
  assetId: string,
  survivingIds: string[],
  touchedIds: Set<string>,
  featuresFor?: (id: string) => Record<string, unknown> | null | undefined,
): void {
  for (const id of survivingIds) {
    if (touchedIds.has(id)) continue;
    emit("accepted", id, assetId, { features: featuresFor?.(id) ?? null });
  }
  flush();
}

// A closed tab must not take its evidence with it.
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", flush);
  window.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flush();
  });
}
