# AGENTS.md

A quick orientation for contributors and coding agents working in this repo.
Read this first — it covers how to build, test, and run ToFU, the layout, and
the gotchas that have actually bitten us.

## What this is

ToFU is context-aware visual text localization: detect text in an image, erase
it, and re-render it in another language while preserving the scene's visual
context. It is a Python library (`tofu-l10n`, in `src/tofu`) plus a FastAPI
server (`server/main.py`) and a React/Vite frontend (`frontend/`).

The pipeline runs a fixed stage order, each stage a module in `tofu.layers`:

```
tofu      pre-flight validation (glyph coverage, expansion feasibility)
scene     candidate text-bearing surfaces, then per-instance style
cicerone  text detection and recognition (OCR)
cleanse   source-text erasure and background reconstruction
scribe    target-language rendering in the detected style
verify    quality scoring
memory    visual translation memory
```

Layers exchange a `TextManifest` (see `src/tofu/core/types.py`) and nothing
else, which is what makes it practical to swap an OCR engine or inpainter
without touching the rest.

## Project layout

```
src/tofu/         the library (import root is src/, package is tofu)
  core/           types, pipeline orchestrator, event bus
  layers/         one module per pipeline stage (cicerone, cleanse, scribe, ...)
  video/          video analysis, temporal tracking, compositor
  utils/          imaging, geometry, glossary, interchange (XLIFF/VTM), textmatch
  resources/      shipped data (corrections/*.json)
server/           FastAPI app (main.py), DB, capabilities, requirements*.txt
frontend/         React 19 + Vite + MUI + Tailwind frontend (TypeScript)
scripts/          eval harnesses (eval_detect, eval_render, ...) + paddle/lama runners
tests/            pytest suite (unit + regression); tests/regression/ for baselines
images/           sample scenes used by tests and the README
```

## The three-venv architecture (read this before installing anything)

ToFU runs **three** interpreters on purpose. Do not collapse them.

| venv             | manifest                            | purpose                                  |
|------------------|-------------------------------------|------------------------------------------|
| `.venv`          | `server/requirements.txt`           | app + pipeline (CPU torch, EasyOCR)      |
| `.venv-paddle`   | `server/requirements-paddle.txt`    | PP-OCRv5 CJK detection — out-of-process  |
| `.venv-inpaint`  | `server/requirements-inpaint.txt`   | LaMa neural repair (CUDA torch) — OOP    |

`paddlepaddle` **force-replaces** the app venv's numpy/opencv on install
(measured, documented in `server/requirements.txt`). PaddleOCR and LaMa are
therefore never imported in-process — cicerone talks to PaddleOCR through
`scripts/paddle_worker.py` exchanging JSON over temp files, and cleanse talks
to LaMa through `scripts/inpaint_lama_runner.py`. The app only exchanges
temporary image/mask/output files with these runners; no request field can
pick a command, endpoint, model, or checkpoint.

When a sibling venv is absent, the backend degrades to `NullBackend` / a
non-neural path rather than raising. So a missing `.venv-paddle` shows up as
*lost CJK recall*, not a crash. Check `PaddleOCRBackend.is_available()` first
when CJK detection quality regresses (defaults to `<repo>/.venv-paddle`,
override with `TOFU_PADDLE_VENV`).

### Setup (Windows; the project is Windows-first)

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r server\requirements.txt

py -3.13 -m venv .venv-paddle
.venv-paddle\Scripts\python -m pip install -r server\requirements-paddle.txt

py -3.13 -m venv .venv-inpaint
.venv-inpaint\Scripts\python -m pip install -r server\requirements-inpaint.txt
```

easyocr's dependency tree is not yet reliable on 3.14; use 3.13 for `.venv`.

## Build / run commands

### Backend (library + server)

Install the library editable (pulls the `[server]` extra for FastAPI):

```
.venv\Scripts\python -m pip install -e ".[all,dev]"
```

Run the API (from `server/`, with the app venv active):

```
cd server
set PYTHONUTF8=1
..\.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

### Frontend

```
cd frontend
npm install
npm run dev      # vite dev server on http://localhost:5173, proxies /api -> :8000
npm run build    # tsc && vite build (the npm script; tsc with the project config)
npm run test     # vitest run
```

### Lint / typecheck

```
ruff check src/ tests/ server/
ruff format --check src/ tests/ server/
mypy src/tofu
npx tsc --noEmit          # from frontend/
```

`ruff` and `mypy` are not in the default venv extras; install them into
`.venv` (`pip install ruff mypy`) if you want to run them locally. The CI
workflow (`.github/workflows/ci.yml`) is the source of truth for what must
pass.

### Tests

```
.venv\Scripts\python -m pytest tests/ -q
```

`PYTHONUTF8=1` is required on Windows — EasyOCR's model-download progress bar
prints characters that crash a cp1252 console.

- `tests/test_paddle_bridge_live.py` self-skips when `.venv-paddle` is absent
  (`PaddleOCRBackend.is_available()` gate). It is the only test that needs the
  paddle venv.
- Inpaint tests do not require `.venv-inpaint` to run; they exercise the
  provider routing and treatment logic against stubs/non-neural paths. The
  live neural round-trip is not in the pytest suite.
- `tests/regression/` holds baseline images/metrics; regression tests compare
  against them.
- `tests/test_server_e2e.py` exercises the full server lifecycle (create →
  upload → detect → add region → translate → export → import) via FastAPI
  TestClient, plus error-path tests (404/422/400/415).
- `tests/test_video_integration.py` tests the video job lifecycle end-to-end
  (create job → timeline → track update → keyframe → cancel/resume → error
  paths) via FastAPI TestClient with synthetic tracks inserted directly into
  the DB. 13 tests.
- `tests/regression/test_video_regression.py` generates deterministic test
  clips, runs them through the compositor, and compares output frames against
  baselines with SSIM ≥ 0.95. Skipped if ffmpeg or scikit-image is unavailable.
- Frontend component tests: `frontend/src/*.test.tsx` run via `npm test`
  (vitest + @testing-library/react + happy-dom). 154 tests across Stepper,
  RegionTable, SemanticSubstitutionPanel, and other components.

### Eval harnesses (scripts/)

```
.venv\Scripts\python scripts\eval_detect.py images\my-image.png --tag baseline --ground-truth images\my-image.gt.json
.venv\Scripts\python scripts\eval_detect.py images\my-image.png --tag paddle --engine paddleocr --ground-truth images\my-image.gt.json
.venv\Scripts\python scripts\eval_render.py    # render-quality eval
.venv\Scripts\python scripts\eval_cleanse_providers.py
.venv\Scripts\python scripts\generate_api_docs.py    # regenerate docs/api.md from OpenAPI schema
```

Switch the OCR engine per request with the `engine` query param
(`/api/detect/stream?engine=paddleocr`) or the `OCR_ENGINE` env var
(`easyocr` or `paddleocr`).

## Known gotchas

1. **Install exactly one OpenCV distribution.** Multiple opencv wheels unpack
   into the same `site-packages/cv2/`, so whichever wrote last silently wins
   and the manifest stops describing what actually imports. This was a live
   bug: `opencv-contrib-python` 4.10 shadowed a pinned headless 5.0, and
   Scene's contour rescue pass missed the banner on `images/japan-street.jpeg`.
   The server never opens a GUI window and no contrib-only module is used
   anywhere in `src/`, `server/`, or `scripts/` — so `opencv-python-headless`
   is the correct and only choice. Do not "upgrade" to contrib to get a
   module; add it deliberately or not at all.

2. **`PYTHONUTF8=1` on Windows.** Required for the test suite and the server.
   EasyOCR's progress bar emits characters that crash a cp1252 console. CI
   sets this in the `Run tests` step env.

3. **PaddleOCR runs out-of-process.** Never `pip install paddlepaddle` into
   `.venv`. It goes in `.venv-paddle` only; cicerone shells out to it. Same
   for LaMa in `.venv-inpaint`. See the three-venv section above.

4. **`Any` for images crossing layer boundaries.** ~91 parameters are typed
   `Any` for image assets, so mypy cannot catch image type errors. mypy is
   intentionally not in strict mode yet — that is the honest state of the
   code rather than a lie in a config file. Introducing a real image protocol
   is tracked as a separate task (Phase 1.5 of the roadmap); do not sprinkle
   annotations to silence warnings.

5. **Explanatory comments are the point.** `ruff` ignores `E501` (long lines)
   on purpose — the long comments in this codebase document *why* a decision
   was made, often with the bug it prevents. Do not delete them to shorten
   lines, and do not add or remove comments unless asked.

6. **Version is single-sourced.** It lives in `src/tofu/__init__.py`
   (`__version__`) and is read from there by `pyproject.toml`. It used to be
   a literal in `interchange.py`, which drifted into disagreeing with itself
   (XLIFF headers said 0.2 while TMX headers in the same module said 0.1).
   Do not re-add a version literal anywhere.

7. **`requirements.txt` is pinned with `==`, not floored with `>=`.** Measured
   numbers in the technical paper have to be reproducible from this file.
   `requirements.lock.txt` is the fully-resolved transitive set for
   byte-identical rebuilds; regenerate it with `pip freeze` after any
   intentional dependency change. The *library* deps in `pyproject.toml` are
   floored (`>=`) on purpose — a library coexists with the caller's
   environment; the app pins because it reproduces measurements.

## CI

`.github/workflows/ci.yml` runs on push to `main`, PRs, and manual dispatch:
- `test` job: `pytest tests/ -q` across `ubuntu-latest`/`windows-latest` on
  Python 3.13, plus 3.11 and 3.12 on Linux only (catches syntax/stdlib drift
  without re-running the whole matrix twice). Installs with the CPU torch
  index, installs Noto fonts on Linux for complex-script shaping tests, sets
  `PYTHONUTF8=1`.
- `lint` job: `ruff check` + `ruff format --check` over `src/ tests/ server/`.
- `typecheck` job: `mypy src/tofu` (with `MYPYPATH=src`) + `npx tsc --noEmit`
  in `frontend/`.
- `build` job: `pip install -e ".[server]"` + `npm run build` in `frontend/`.
- `frontend-test` job: `npm test` (vitest) in `frontend/`.
- `package` job: builds the sdist+wheel, clean-room installs the wheel into a
  fresh venv and imports it with no `PYTHONPATH`, and asserts `py.typed`
  ships in the wheel.

`.github/workflows/benchmark.yml` runs nightly at 03:00 UTC and on manual
dispatch. It runs `tests/regression/` (excluding video) and posts a comment
on the PR if a `pr_number` input is provided and any metric regresses beyond
tolerance. The baseline lives in
`tests/fixtures/regression/baseline_metrics.json`.

Red CI blocks merge.

## Pre-commit hooks

`.pre-commit-config.yaml` mirrors the CI `lint` and `typecheck` jobs locally
so a green local commit is a green CI lint/typecheck run. Install once:

```
pip install pre-commit
pre-commit install
pre-commit run --all-files   # one-time full-repo check
```

Hooks: ruff (lint + autofix) and ruff-format (check only — does not rewrite on
commit) on staged `.py` files under `src/`/`tests/`/`server/`; mypy on staged
`src/tofu/*.py`; a local tsc hook that runs `npx tsc --noEmit` in `frontend/`
only when a `.ts`/`.tsx` file is staged (so Python-only commits skip Node).
Revs are pinned — bump deliberately, not automatically.
