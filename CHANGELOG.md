# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker deployment: multi-stage Dockerfile, Dockerfile.paddle,
  Dockerfile.inpaint, Dockerfile.frontend, docker-compose.yml with
  profiles for CJK and GPU inpainting sidecars.

### Changed
- Version bumped to 1.0.0 (Production/Stable).
- FastAPI app version synchronized with library version.

## [1.0.0] - 2026-08-04

First production release. All Phase 1-5 work is included here.

### Added
- **Video pipeline**: temporal tracking edge cases — occlusion handling
  (`MAX_GAP_FRAMES`), rapid text change splitting
  (`TEXT_CHANGE_SPLIT_THRESHOLD`), birth/death smoothing
  (`MIN_CONFIRM_OBSERVATIONS`).
- **VideoWorkspace frontend**: timeline scrubber with keyframe/issue markers,
  track list with confidence and keyframe count, per-track style editing
  (opacity, color), side-by-side preview comparison, progress indicator with
  cancel/resume.
- **Video integration tests**: 13 end-to-end tests covering the full job
  lifecycle (create → timeline → track update → keyframe → cancel/resume →
  error paths).
- **API reference**: `docs/api.md` generated from FastAPI OpenAPI schema via
  `scripts/generate_api_docs.py`. 80 endpoints grouped by 24 tags.
- **Architecture diagram**: `docs/architecture.md` with Mermaid diagrams for
  system overview, image pipeline, video pipeline, temporal tracking states,
  three-venv architecture, and data flow.
- **CONTRIBUTING.md**: development setup, code style, test conventions, PR
  checklist, release process.
- **CI**: API docs freshness check added to `ci.yml` typecheck job.
- **Benchmark CI**: nightly `benchmark.yml` workflow for regression metric
  drift detection.
- **Frontend CI**: `frontend-test` job added to `ci.yml` for vitest.
- **E2E server tests**: 15 tests covering full lifecycle and error paths.
- **Frontend component tests**: 27 tests for Stepper, RegionTable,
  SemanticSubstitutionPanel.
- **Video regression baselines**: 4 SSIM-based tests with 27 baseline PNGs.

### Changed
- FastAPI app enriched with description and route tags for OpenAPI grouping.
- `detect_text_change` threshold raised from 0.35 to 0.6 to correctly split
  short-text swaps ("SALE" → "CLOSED") without splitting OCR noise.
- `detect_text_change` returns False for None inputs (missing readings are
  not content changes).
- `MISSED_KEYFRAMES_BEFORE_RETIREMENT` kept at 2 to preserve resume test
  compatibility.

### Fixed
- `drain()` now correctly manages `dormant` and `pending` tracks, recreating
  tracks from dormant state and preserving consensus history across
  occlusion gaps.
- TypeScript `TS2322` error in `SemanticSubstitutionPanel.test.tsx` resolved
  by explicitly typing `baseProps`.

## [0.1.0] - 2026-01-15

### Added
- **7-layer image pipeline**: ToFU (pre-flight validation), Scene (semantic
  context), Cicerone (text detection + OCR), Cleanse (text erasure +
  inpainting), Scribe (style-aware text rendering), Verify (quality
  verification), Memory (visual translation memory).
- **Auxiliary modules**: Savor (post-recognition glyph correction), Wasabi
  (CJK glyph normalization), Menu (gazetteer-assisted place-name recovery).
- **Three-venv architecture**: main venv (torch + EasyOCR + FastAPI),
  PaddleOCR venv (CJK detection), inpaint venv (LaMa neural repair).
- **Video pipeline**: Braise adaptive keyframe OCR tracker, compositor with
  preview/export parity, resumable checkpoints.
- **FastAPI server**: 81 REST endpoints for asset management, detection,
  translation, rendering, verification, video jobs, projects, snapshots,
  translation memory, glossary, and semantic units.
- **React frontend**: Vite + MUI + Tailwind, 5-step workflow (Upload →
  Capture → Translate → Render → Verify), VideoWorkspace for video editing.
- **Open VTM format**: open visual translation memory format v1.0 (MIT).
- **Evaluation harnesses**: `scripts/eval_*.py` for detection, render,
  cleanse, font match, memory, savor, paddle, verification corpus.
- **XLIFF/VTM interchange**: import/export for translation memory and
  industry-standard XLIFF format.
- **Regression baselines**: per-image metric baselines for detection,
  rendering, and verification quality.
- **Manual bounding-box snap**: click-to-snap region boundaries to text edges.
- **LLM comparison**: side-by-side comparison with Gemini and ChatGPT output.
- **Semantic substitution**: Basil semantic units for context-aware
  translation of signs, menus, and recurring text.
- **Region merging**: manual region merge with OCR fallback for fragmented
  detections.
- **Perspective quad**: perspective-aware text erasure and rendering for
  non-planar surfaces.
- **Font matching**: visual font matching with weight/italic/color profiling.
- **Multi-candidate OCR**: multiple OCR candidates per region with
  confidence scoring and arbitration.
- **Glossary**: project-level glossary for consistent terminology.
- **Project management**: multi-project support with asset organization.
- **Snapshot system**: save and restore translation session state.

### Known Limitations
- Video analysis requires FFmpeg; degrades gracefully when absent.
- PaddleOCR CJK detection requires separate `.venv-paddle`; degrades to
  reduced CJK recall when absent.
- Neural inpainting (LaMa) requires separate `.venv-inpaint` and CUDA;
  falls back to non-neural inpainting when absent.
