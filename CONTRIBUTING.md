# Contributing to ToFU

Thanks for your interest in contributing! This guide covers everything you
need to submit a PR that passes CI on the first try.

## Development Setup

ToFU is Windows-first and uses **three** Python virtual environments to
isolate incompatible dependencies. See [AGENTS.md](AGENTS.md) for the full
three-venv architecture explanation.

### Quick Start

```powershell
# 1. Main venv (app + pipeline)
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r server\requirements.txt

# 2. Frontend
cd frontend
npm install
cd ..

# 3. Lint/typecheck tools (not in requirements.txt)
.venv\Scripts\python -m pip install ruff mypy
```

### Optional Venvs

Only needed if you're working on CJK detection or neural inpainting:

```powershell
# PaddleOCR (CJK detection)
py -3.13 -m venv .venv-paddle
.venv-paddle\Scripts\python -m pip install -r server\requirements-paddle.txt

# Neural inpainting (LaMa)
py -3.13 -m venv .venv-inpaint
.venv-inpaint\Scripts\python -m pip install -r server\requirements-inpaint.txt
```

### Running the Stack

```powershell
# Terminal 1: API server
cd server
set PYTHONUTF8=1
..\.venv\Scripts\python -m uvicorn main:app --reload --port 8000

# Terminal 2: Frontend dev server
cd frontend
npm run dev    # http://localhost:5173, proxies /api -> :8000
```

## Code Style

### Python

- **Linter**: [ruff](https://docs.astral.sh/ruff/) — run `ruff check src/ tests/ server/`
- **Formatter**: `ruff format --check src/ tests/ server/`
- **Line length**: 100 characters
- **Target**: Python 3.11+ (3.13 is the primary dev version)
- **Type checker**: `mypy src/tofu` (with `MYPYPATH=src`)

Configuration is in `pyproject.toml` under `[tool.ruff]`.

### TypeScript / React

- **Type checker**: `npx tsc --noEmit` (from `frontend/`)
- **Formatter**: Prettier (via Vite)
- **Test runner**: Vitest + @testing-library/react + happy-dom

### Comment Policy

- Comments explain *why*, not *what*. The code already says what.
- Use `##` for full-line comments (ruff format preserves alignment), `#` for inline.
- Don't add or remove comments unless asked. If you find you've accidentally
  deleted an existing comment, put it back.
- Comments are part of the codebase's institutional memory — they document
  gotchas, design decisions, and edge cases that aren't obvious from the code.

## Test Conventions

### Test Types

| Type | Location | Runner | Purpose |
|------|----------|--------|---------|
| Unit | `tests/test_*.py` | pytest | Module-level logic, mocked deps |
| Regression | `tests/regression/test_*.py` | pytest | Baseline metric comparison |
| E2E (server) | `tests/test_server_e2e.py` | pytest (TestClient) | Full API lifecycle |
| E2E (video) | `tests/test_video_integration.py` | pytest (TestClient) | Video job lifecycle |
| Frontend | `frontend/src/*.test.tsx` | vitest | Component rendering & interaction |
| Video regression | `tests/regression/test_video_regression.py` | pytest + SSIM | Compositor output baselines |

### Running Tests

```powershell
# All backend tests
set PYTHONUTF8=1
.venv\Scripts\python -m pytest tests/ -q

# Frontend tests
cd frontend
npm test

# Only regression tests
.venv\Scripts\python -m pytest tests/regression/ -q

# Only video tests
.venv\Scripts\python -m pytest tests/test_video_*.py tests/regression/test_video_regression.py -q
```

`PYTHONUTF8=1` is required on Windows — EasyOCR's progress bar prints
characters that crash a cp1252 console.

### Writing Tests

- **Unit tests**: Mock external dependencies (OCR engines, file I/O). Use the
  `fake_ocr` fixture pattern from `tests/test_video_analysis.py`.
- **Regression tests**: Compare against baselines in
  `tests/fixtures/regression/baseline_metrics.json`. To update baselines after
  an intentional improvement, run `scripts/generate_baseline.py` and commit
  the updated JSON.
- **E2E tests**: Use the `client` fixture (FastAPI TestClient with isolated
  tmp_path). See `tests/test_server_e2e.py` for the pattern.
- **Frontend tests**: Use `@testing-library/react` with `screen` queries.
  Prefer `data-*` attributes and `aria-label` selectors over text matching
  (text can appear in multiple places).

## PR Checklist

Before submitting a PR, verify:

- [ ] `ruff check src/ tests/ server/` passes
- [ ] `ruff format --check src/ tests/ server/` passes
- [ ] `mypy src/tofu` passes (with `MYPYPATH=src`)
- [ ] `npx tsc --noEmit` passes (from `frontend/`)
- [ ] `npm test` passes (from `frontend/`)
- [ ] `pytest tests/ -q` passes (with `PYTHONUTF8=1`)
- [ ] New features have tests
- [ ] README updated if user-facing
- [ ] `docs/api.md` regenerated if endpoints changed
  (`python scripts/generate_api_docs.py`)
- [ ] CHANGELOG.md entry added under `[Unreleased]`
- [ ] No secrets, API keys, or credentials committed

## CI

CI runs on push to `main`, PRs, and manual dispatch
(`.github/workflows/ci.yml`):

| Job | What it checks |
|-----|---------------|
| `test` | `pytest tests/ -q` on Python 3.11/3.12/3.13, Ubuntu + Windows |
| `lint` | `ruff check` + `ruff format --check` |
| `typecheck` | `mypy src/tofu` + `npx tsc --noEmit` + API docs freshness |
| `frontend-test` | `npm test` (vitest) |
| `build` | Server + frontend build |
| `package` | Wheel build + clean-room import |

A nightly benchmark workflow (`.github/workflows/benchmark.yml`) runs
regression tests and posts a PR comment if metrics drift.

**Red CI blocks merge.**

## Release Process

1. Update `__version__` in `src/tofu/__init__.py`
2. Update `app.version` in `server/main.py`
3. Add a `CHANGELOG.md` entry under the new version heading
4. Tag the release: `git tag v0.X.Y`
5. The `package` CI job builds the wheel; download from the `dist` artifact
6. Publish to PyPI: `python -m twine upload dist/tofu_l10n-0.X.Y-*.whl`

## Architecture

See [docs/architecture.md](docs/architecture.md) for system diagrams and
[docs/cross-layer-architecture.md](docs/cross-layer-architecture.md) for the
detailed layer-by-layer exposition.

The quick version: ToFU is a 7-layer image pipeline (tofu → scene → cicerone
→ cleanse → scribe → verify → memory) plus a video pipeline (Braise tracker +
compositor). All layers exchange a `TextManifest` and nothing else. The
FastAPI server owns persistence (SQLite + file storage); pipeline layers are
pure functions.
