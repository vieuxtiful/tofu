import type { BBox } from "./api";

export type RepairCandidate = {
  id: string;
  provider: string;
  decision: string;
  url: string;
  bbox: BBox;
  cacheKey: string;
};

export type RepairReview = {
  id: string;
  provider: string;
  reason: string;
  candidates: RepairCandidate[];
};

export type LocalizedCandidatePreview = {
  status: "loading" | "ready" | "error";
  url?: string;
  error?: string;
};
