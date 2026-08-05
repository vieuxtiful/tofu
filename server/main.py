## 🍢 ToFU — FastAPI backend
## vieuxtiful
"""
HTTP bridge between the frontend and the tofu pipeline.

endpoints:
  POST /api/assets                          upload an asset
  GET  /api/languages                       supported languages (with display names)
  GET  /api/fonts?lang=&limit=              ranked fonts for a language
  POST /api/validate                        ToFU layer-0 pre-flight report
  POST /api/detect                          run Cicerone, persist manifest
  GET  /api/manifest/{asset_id}             retrieve persisted manifest
  PUT  /api/manifest/{asset_id}             replace full manifest (auto-save)
  POST /api/manifest/{asset_id}/regions     add a region
  DELETE /api/manifest/{asset_id}/regions/{rid}  remove a region
  PATCH /api/manifest/{asset_id}/regions/{rid}   update a region
  POST /api/ocr-region                      OCR a specific bbox crop
  POST /api/export                          export to XLIFF/TMX/TSV/CSV/TXT/VTM
  POST /api/import                          import translated file
  POST /api/render                          run scene→cleanse→scribe→verify
  POST /api/process                         legacy full pipeline

run:  uvicorn main:app --reload --port 8000   (from server/)
"""

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tofu.core.pipeline import TofuPipeline
from tofu.core.events import PipelineEvent, PipelineEventStatus
from tofu.core.types import (
    PipelineCfg, LayerMode, infer_asset_info, TextManifest, InstText, BBox,
    RenderParams, StyleProfil, VldtnClass,
)
from tofu.layers.tofu import ToFU, lang_to_script
from tofu.layers.fonts import (
    FONT_EXTS, PACKS_SUBDIR, USER_SUBDIR, pantries, pantry, writable_pantry,
)
from tofu.layers.sift import sift
from tofu.layers import cicerone, memory, scribe, garnish, cleanse, scene, verify, inpaint_providers
from tofu.layers.cicerone import _to_easyocr_lang
from tofu.utils.manifest_store import save_manifest, load_manifest, _dict_to_manifest
from tofu.utils import interchange
from tofu.utils import glossary as glossary_utils
from tofu.video.media import MediaError, available as video_media_available, encode_vfr_sequence, make_proxy, mux_preview_audio, mux_rendered_video, probe, verify_output
from tofu.video.analysis import ANALYZER_REVISION, analyze as analyze_video
from tofu.video.checkpoint import (ResumeMismatch, TrackerCheckpoint,
                                   source_fingerprint as video_source_fingerprint)
from tofu.video.compositor import compose as compose_video
from tofu.video.plan import plan_inputs as video_plan_inputs
from tofu.video.temporal import settle as settle_tracks
from tofu.video.types import VideoManifest

import db

UPLOAD_DIR = ROOT / "server" / "uploads"
OUTPUT_DIR = ROOT / "server" / "outputs"
TM_THUMB_DIR = ROOT / "server" / "tm_thumbs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TM_THUMB_DIR.mkdir(parents=True, exist_ok=True)
GLOSSARY_DIR = UPLOAD_DIR / "glossaries"
GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()
VIDEO_WORKER_STOP = threading.Event()
VIDEO_WORKER_THREAD: Optional[threading.Thread] = None


def _font_dir() -> Optional[str]:
    """the one font-directory answer, shared with the eval harnesses.

    this chain used to be duplicated here and in scripts/, so a fix to
    one never reached the other. see tofu.layers.fonts.pantry().
    """
    return pantry()


def _font_dirs() -> List[str]:
    """EVERY font root the server serves from.

    The server takes all of them where the eval harnesses take the first:
    a harness names one root and joins paths against it, so widening it
    would change what those measure, while the server's job is to offer
    the user every face actually available.
    """
    return pantries()


_validator: Optional[ToFU] = None


def get_validator() -> ToFU:
    global _validator
    if _validator is None:
        _validator = ToFU(font_library_path=_font_dirs())
    return _validator


def _font_roots() -> List[Path]:
    """Resolved font roots, for deciding whether a path may be served."""
    roots = [Path(d).resolve() for d in _font_dirs()]
    # The writable roots may not exist yet but are still legitimate, so
    # they are named here rather than inferred from what is on disk.
    for subdir in (USER_SUBDIR, PACKS_SUBDIR):
        roots.append((ROOT / "server" / "font-library" / subdir).resolve())
    return roots


def _is_served_font(path: Path) -> bool:
    """Is this file inside a directory the server is willing to serve?

    /api/font-file previously resolved ANY readable path, with
    application/octet-stream as the fallback media type -- so it would
    hand back any file on the host that a caller could name. Harmless
    while fonts only ever came from fixed system directories; not harmless
    once callers can also PUT files on the box.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in _font_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(jsonable(k)): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj


def _asset_path(asset_id: str) -> Path:
    matches = [
        p for p in UPLOAD_DIR.glob(f"{asset_id}.*")
        if not p.name.endswith(".manifest.json")
    ]
    if not matches:
        raise HTTPException(404, f"asset '{asset_id}' not found")
    return matches[0]


def _normalized_asset_bbox(asset_id: str, raw: Dict[str, Any]) -> BBox:
    """Validate and clip a client rectangle to the uploaded image.

    Canvas gestures are already clamped in the browser, but API callers and
    stale browser state are not. Keeping this at the HTTP boundary prevents
    malformed rectangles from becoming Pillow crops, OCR-worker failures, or
    invalid persisted manifest geometry.
    """
    required = ("x", "y", "width", "height")
    try:
        values = {key: int(raw[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, "bbox requires integer x, y, width, and height") from exc
    if values["width"] <= 0 or values["height"] <= 0:
        raise HTTPException(422, "bbox width and height must be positive")

    path = _asset_path(asset_id)
    try:
        from PIL import Image
        with Image.open(path) as image:
            image_width, image_height = image.size
    except Exception as exc:
        raise HTTPException(422, f"cannot determine asset dimensions: {type(exc).__name__}") from exc

    left = max(0, values["x"])
    top = max(0, values["y"])
    right = min(image_width, values["x"] + values["width"])
    bottom = min(image_height, values["y"] + values["height"])
    if right <= left or bottom <= top:
        raise HTTPException(422, "bbox does not intersect the asset")
    return BBox(x=left, y=top, width=right - left, height=bottom - top)


def _persist_tm_updates(pid: Optional[str], drafts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """endpoint-owned I/O for memory.update()'s draft records: save each
    thumbnail crop to disk, write the row to tm_records, and return a
    JSON-safe summary (the raw PIL 'thumb_crop' never survives past this
    function -- jsonable_encoder has no handler for it, the same class of
    bug as Phase 5's numpy-scalar leak). no-ops (but still strips
    thumb_crop) when the asset isn't attached to a project, since
    tm_records requires a project_id."""
    out: List[Dict[str, Any]] = []
    for d in drafts:
        thumb_crop = d.get("thumb_crop")
        record_id = None
        thumb_path = None
        if pid is not None:
            if thumb_crop is not None:
                thumb_name = f"{uuid.uuid4().hex}.png"
                try:
                    thumb_crop.save(TM_THUMB_DIR / thumb_name)
                    thumb_path = thumb_name
                except Exception:
                    thumb_path = None
            record_id = db.store_tm_record(
                pid, d["asset_id"], d["region_id"], d["source_text"],
                d["normalized_text"], d.get("source_lang"), d["target_lang"],
                d["target_text"], d.get("style_fingerprint"), d.get("phash"),
                thumb_path, d["qa_score"],
            )
        out.append({k: v for k, v in d.items() if k != "thumb_crop"} | {"record_id": record_id})
    return out


def _attach_tm_suggestions(pid: Optional[str], manifest: TextManifest, source_path) -> int:
    """post-detect TM lookup: populates InstText.tm_suggestion for regions
    with a confident memory match. returns the number of regions matched
    (the 'seen before' signal). no-ops outside a project (no TM to
    search) or a manifest with no source language recorded yet."""
    if pid is None:
        return 0
    candidates = db.find_tm_candidates(pid, manifest.targ_lang or "en", manifest.src_lang)
    if not candidates:
        return 0
    matches = memory.lookup(manifest, str(source_path), manifest.targ_lang or "en", candidates)
    for inst in manifest.instances:
        m = matches.get(inst.id)
        inst.tm_suggestion = m
    return len(matches)


def _resolve_auto_fonts(manifest: TextManifest, default_targ_lang: Optional[str] = None) -> int:
    """populate InstText.resolved_font_family for every region left on
    "auto" (style_profile.font_family is None/unset): what does auto
    ACTUALLY render with, right now, for this region's own text/weight/
    italic? uses scribe.resolve_auto_font() -- the identical resolution
    render() itself performs, including picking a real bold/italic
    sibling face when style_profile.font_weight/italic asks for one on
    an otherwise-auto family -- so the Font column label and the
    Translate-tab preview both show something guaranteed to match the
    real render, not a guess. cheap (registry codepoint lookups, no font
    file I/O), so it's safe to recompute on every call rather than
    tracking what changed since the last one.

    resolves against target_text when one exists (what will actually be
    drawn) and falls back to the source text otherwise (capture-time,
    before any translation is entered, still shows a reasonable guess).
    explicit font_family picks are left untouched -- resolved_font_family
    is a display hint only, never itself treated as an override.
    """
    registry = get_validator().font_registry
    if registry is None:
        return 0
    resolved = 0
    for inst in manifest.instances:
        text = inst.target_text or inst.text or ""
        lang = inst.target_language or default_targ_lang or manifest.targ_lang
        weight = inst.style_profile.font_weight if inst.style_profile else None
        italic = bool(inst.style_profile.italic) if inst.style_profile else False
        explicit = inst.style_profile.font_family if inst.style_profile else None
        if explicit:
            # An explicit pick needs no path resolution, but it still goes
            # through resolve_face() at render time, which is where a
            # requested italic with no real italic sibling becomes a
            # SHEAR. The preview has to know that, so the flag is computed
            # for every region even though the path is not.
            _, synthetic = scribe.resolve_face(registry, explicit, weight, italic)
            inst.resolved_synthetic_italic = bool(synthetic)
            continue
        path, synthetic = scribe.resolve_auto_font_face(
            registry, lang, text, weight=weight, italic=italic,
        )
        if path != inst.resolved_font_family:
            inst.resolved_font_family = path
        inst.resolved_synthetic_italic = bool(synthetic)
        if path:
            resolved += 1
    return resolved


@contextlib.contextmanager
def _matched_faces_applied(manifest: TextManifest):
    """Render regions still on "auto" with the face Basil's bouquet agreed on.

    The frontend's resolution ladder (doppelganger.ts) has always ranked
    ``font_match.recommended_substitute`` above the generic auto default,
    while /api/render only ever consulted ``style_profile.font_family``.
    So the Translate preview showed the matched face and the render drew
    the fallback: a localiser approved one thing and shipped another, and
    the ladder had to carry a ``useMatch: false`` mode purely to describe
    the discrepancy.  This closes it from the render side.

    It does not move the boundary that keeps font matching honest.  The
    assignment lasts exactly as long as the render and is undone in the
    ``finally``, so the PERSISTED style_profile still holds nothing but a
    human's explicit pick -- the invariant tests/test_font_matching.py
    asserts.  A region that HAS an explicit pick is never touched: rung 1
    outranks rung 2 here for the same reason it does in the ladder.

    Mutating in place rather than rendering a copy is deliberate: scribe
    writes ``glyph_fallback`` back onto the instances it drew, and a copy
    thrown away after the render would swallow those flags.
    """
    touched: List[Tuple[Any, Optional[StyleProfil]]] = []
    for inst in manifest.instances:
        style = inst.style_profile
        if style is not None and style.font_family:
            continue  # rung 1: a human already chose, and that always wins
        match = inst.font_match or {}
        if match.get("status") == "unavailable":
            continue
        path = (match.get("recommended_substitute") or {}).get("font_path")
        if not path:
            continue
        touched.append((inst, style))
        if style is None:
            inst.style_profile = StyleProfil(font_family=path)
        else:
            style.font_family = path
    try:
        yield [inst.id for inst, _ in touched]
    finally:
        for inst, style in touched:
            if style is None:
                inst.style_profile = None
            else:
                style.font_family = None


def _revalidate_regions(
    validator: ToFU,
    path: Any,
    manifest: TextManifest,
    targ_lang: str,
    font: Optional[str],
) -> tuple:
    """Re-validate every effective target language with real per-region
    typography context, without saying the same thing N times.

    Most of what ToFU reports at this stage — script support, render
    quality — is a fact about a (language, font, size, effects)
    combination, not about one box. Validating once per INSTANCE meant a
    30-region Japanese scene produced 30 identical warnings, each stamped
    with a different region id, which buries the one finding that is
    genuinely per-region among 29 copies of a language-level one.

    So: validate once per distinct context, then merge issues that are
    the same finding, carrying the regions they came from on region_ids.

    returns (merged_issues, languages, validate_calls).
    """
    contexts: Dict[tuple, List[str]] = {}
    for inst in manifest.instances:
        if inst.dnt or not inst.target_text:
            continue
        lang = inst.target_language or targ_lang
        font_px = (
            inst.characteristics.size
            if inst.characteristics and inst.characteristics.size else None
        )
        sp = inst.style_profile
        effects = []
        if sp:
            if sp.shadow: effects.append("shadow")
            if sp.stroke_width: effects.append("stroke")
            if sp.italic: effects.append("italic")
        contexts.setdefault((lang, font_px, tuple(effects)), []).append(inst.id)

    merged: Dict[tuple, VldtnClass] = {}
    for (lang, font_px, effects), region_ids in contexts.items():
        ctx: Dict[str, Any] = {"font": font} if font else {}
        if font_px:
            ctx["font_px"] = font_px
        if effects:
            ctx["effects"] = list(effects)
        for issue in validator.validate(str(path), lang, ctx or None).issues:
            # a region-scoped issue (ToFU_005/007) keeps its own anchor;
            # anything else is a property of the context, so it collects
            # every region that shares that context
            owners = [issue.region_id] if issue.region_id else list(region_ids)
            key = (issue.code, issue.message, issue.region_id)
            existing = merged.get(key)
            if existing is None:
                issue.region_id = owners[0]
                issue.region_ids = list(owners)
                merged[key] = issue
            else:
                existing.region_ids.extend(
                    r for r in owners if r not in existing.region_ids
                )

    return list(merged.values()), list({k[0] for k in contexts}), len(contexts)


def _infer_src_lang(manifest: TextManifest) -> str:
    """dominant detected language, weighted by region area — a storefront
    sign outvotes a handful of small incidental latin fragments.

    the vote itself lives in cicerone (where the evidence is produced and
    where build_manifest now applies it); this wrapper only keeps the
    server's "always answer something" contract.
    """
    return cicerone.taste_the_room(manifest.instances) or "en"


LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "uk": "Ukrainian",
    "el": "Greek", "ar": "Arabic", "fa": "Persian", "he": "Hebrew",
    "ja": "Japanese", "ko": "Korean",
    "zh-cn": "Chinese (Simplified)", "zh-sg": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "zh-hk": "Chinese (Traditional)",
    "zh-mo": "Chinese (Traditional)",
    "sr-latn": "Serbian (Latin)", "sr-cyrl": "Serbian (Cyrillic)",
    "hi": "Hindi", "th": "Thai", "vi": "Vietnamese", "bn": "Bengali",
    "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "si": "Sinhala",
    "my": "Burmese", "km": "Khmer", "lo": "Lao", "am": "Amharic",
    "ti": "Tigrinya", "hy": "Armenian", "ka": "Georgian", "mn": "Mongolian",
    "kk": "Kazakh", "uz": "Uzbek", "az": "Azerbaijani", "tr": "Turkish",
    "nl": "Dutch", "sv": "Swedish", "no": "Norwegian", "da": "Danish",
    "fi": "Finnish", "is": "Icelandic", "pl": "Polish", "cs": "Czech",
    "sk": "Slovak", "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
    "sr": "Serbian", "hr": "Croatian", "sl": "Slovenian", "et": "Estonian",
    "lv": "Latvian", "lt": "Lithuanian",
    "en-US": "English (US)", "es-MX": "Spanish (Mexico)",
    "es-US": "Spanish (US)", "pt-BR": "Portuguese (Brazil)",
    "fr-CA": "French (Canada)",
    # full bcp-47 locale tags (wizard canonical form).  bare codes above
    # are kept for projects created before the locale-tag refactor.
    "en-GB": "English (UK)", "es-ES": "Spanish (Spain)",
    "fr-FR": "French (France)", "de-DE": "German (Germany)",
    "it-IT": "Italian (Italy)", "pt-PT": "Portuguese (Portugal)",
    "nl-NL": "Dutch (Netherlands)", "sv-SE": "Swedish (Sweden)",
    "no-NO": "Norwegian (Norway)", "da-DK": "Danish (Denmark)",
    "fi-FI": "Finnish (Finland)", "is-IS": "Icelandic (Iceland)",
    "pl-PL": "Polish (Poland)", "cs-CZ": "Czech (Czechia)",
    "sk-SK": "Slovak (Slovakia)", "hu-HU": "Hungarian (Hungary)",
    "ro-RO": "Romanian (Romania)", "bg-BG": "Bulgarian (Bulgaria)",
    "sr-RS": "Serbian (Serbia)", "sr-Latn-RS": "Serbian (Latin)",
    "sr-Cyrl-RS": "Serbian (Cyrillic)",
    "hr-HR": "Croatian (Croatia)", "sl-SI": "Slovenian (Slovenia)",
    "et-EE": "Estonian (Estonia)", "lv-LV": "Latvian (Latvia)",
    "lt-LT": "Lithuanian (Lithuania)", "el-GR": "Greek (Greece)",
    "ru-RU": "Russian (Russia)", "uk-UA": "Ukrainian (Ukraine)",
    "tr-TR": "Turkish (Türkiye)",
    "ja-JP": "Japanese (Japan)", "ko-KR": "Korean (Korea)",
    "zh-CN": "Chinese (Simplified)", "zh-SG": "Chinese (Singapore)",
    "zh-TW": "Chinese (Traditional)", "zh-HK": "Chinese (Hong Kong)",
    "zh-MO": "Chinese (Macau)", "mn-MN": "Mongolian (Mongolia)",
    "hi-IN": "Hindi (India)", "bn-BD": "Bengali (Bangladesh)",
    "pa-IN": "Punjabi (India)", "gu-IN": "Gujarati (India)",
    "mr-IN": "Marathi (India)", "ta-IN": "Tamil (India)",
    "te-IN": "Telugu (India)", "kn-IN": "Kannada (India)",
    "ml-IN": "Malayalam (India)", "si-LK": "Sinhala (Sri Lanka)",
    "am-ET": "Amharic (Ethiopia)", "ti-ER": "Tigrinya (Eritrea)",
    "vi-VN": "Vietnamese (Vietnam)", "th-TH": "Thai (Thailand)",
    "my-MM": "Burmese (Myanmar)", "km-KH": "Khmer (Cambodia)",
    "lo-LA": "Lao (Laos)",
    "ar-SA": "Arabic (Saudi Arabia)", "fa-IR": "Persian (Iran)",
    "he-IL": "Hebrew (Israel)", "hy-AM": "Armenian (Armenia)",
    "ka-GE": "Georgian (Georgia)", "kk-KZ": "Kazakh (Kazakhstan)",
    "uz-UZ": "Uzbek (Uzbekistan)", "az-AZ": "Azerbaijani (Azerbaijan)",
}


@contextlib.asynccontextmanager
async def app_lifespan(_: FastAPI):
    global VIDEO_WORKER_THREAD
    VIDEO_WORKER_STOP.clear()
    # Nothing of ours holds a lease yet, so any row still marked 'running' belongs to a
    # worker that died.  Requeue before starting ours, or a crash on the final attempt
    # would strand the operation as permanently 'running'.
    db.requeue_abandoned_video_operations()
    worker_id = f"desktop-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    VIDEO_WORKER_THREAD = threading.Thread(target=_video_worker_loop, args=(worker_id,), daemon=True, name="tofu-video-worker")
    VIDEO_WORKER_THREAD.start()
    try: yield
    finally:
        VIDEO_WORKER_STOP.set()
        if VIDEO_WORKER_THREAD: VIDEO_WORKER_THREAD.join(timeout=5)


app = FastAPI(title="ToFU", version="0.2.0", lifespan=app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/tm_thumbs", StaticFiles(directory=TM_THUMB_DIR), name="tm_thumbs")

FRONTEND_URL = os.environ.get("TOFU_FRONTEND_URL", "http://localhost:5173")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """opening the API port in a browser lands on the app, not a JSON 404."""
    return RedirectResponse(FRONTEND_URL)


@app.get("/api/capabilities")
def capabilities():
    """One side-effect-free runtime contract for frontend feature gating."""
    from capabilities import build_capabilities
    from tofu.layers.basil import provider_statuses as semantic_provider_statuses

    return build_capabilities(
        get_validator,
        app_version=app.version,
        inpaint_statuses=inpaint_providers.provider_statuses(),
        semantic_statuses=semantic_provider_statuses(),
    )


def _startup_font_check() -> None:
    """verify scribe's fallback fonts resolve on this host; log the winner."""
    try:
        from PIL import ImageFont
        from tofu.layers.scribe import FALLBACK_FONTS
    except ImportError:
        print("[tofu] WARNING: Pillow unavailable; scribe will pass assets through")
        return
    for cand in FALLBACK_FONTS:
        try:
            ImageFont.truetype(cand, 12)
            print(f"[tofu] scribe fallback font resolved: {cand}")
            return
        except Exception:
            continue
    print("[tofu] WARNING: no scribe fallback font resolved; "
          "text will render with PIL's bitmap default")


_startup_font_check()


# --- request models ---

class ValidateRequest(BaseModel):
    asset_id: str
    targ_lang: str
    font: Optional[str] = None

class ProcessRequest(BaseModel):
    asset_id: str
    targ_lang: str
    font: Optional[str] = None
    qa_threshold: Optional[float] = None
    translations: Optional[Dict[str, str]] = None

class DetectRequest(BaseModel):
    asset_id: str
    gpu: Optional[bool] = None
    languages: Optional[List[str]] = None  ## source-language hints (tofu codes)

class GroundTruthUpdate(BaseModel):
    ground_truth: List[str] = []

class OcrRegionRequest(BaseModel):
    asset_id: str
    bbox: Dict[str, int]

class RefineRegionRequest(BaseModel):
    asset_id: str
    bbox: Dict[str, int]
    engine: Optional[str] = None
    scale: Optional[int] = 2

class FontMatchRequest(BaseModel):
    # Commercial catalog matching transmits a text crop to the configured
    # provider, so it is opt-in even when a server API key is available.
    allow_external: bool = False

class ExportRequest(BaseModel):
    asset_id: str
    format: str
    variant: Optional[str] = "standard"
    targ_lang: Optional[str] = ""

class RenderRequest(BaseModel):
    asset_id: str
    targ_lang: str
    font: Optional[str] = None
    qa_threshold: Optional[float] = None

class PreviewRenderRequest(BaseModel):
    asset_id: str
    targ_lang: str
    manifest: Optional[Dict[str, Any]] = None
    # Garnish/treatment controls do not alter scene interpretation or the
    # Cleanse cache key.  Their preview can safely skip that expensive pass.
    fast_path: bool = False
    # Workspace-only inspection control. It never changes the manifest or
    # final render; Cleanup can expose the patched Cleanse base directly.
    show_localized_text: bool = True

class CandidatePreviewRequest(BaseModel):
    manifest: Optional[Dict[str, Any]] = None
    targ_lang: Optional[str] = None

class InpaintRequest(BaseModel):
    asset_id: str
    polygon: List[List[int]] = []
    points: List[List[int]] = []
    mode: str = "auto"
    radius: int = 18
    hardness: float = 0.85
    blur_strength: float = 0.5
    opacity: float = 1.0
    clone_source: Optional[List[int]] = None
    # The localized canvas can be ahead of autosave.  Supplying this snapshot
    # keeps manual treatment on the exact same Cleanse base as the preview,
    # without mutating the stored manifest.
    manifest: Optional[Dict[str, Any]] = None

class CandidateApplyRequest(BaseModel):
    cache_key: Optional[str] = None

class TreatmentRestoreRequest(BaseModel):
    patch_ids: List[str] = []

class LocalizedBaselineRequest(BaseModel):
    manifest: Dict[str, Any]
    patch_ids: List[str] = []

class ApproveRequest(BaseModel):
    asset_id: str
    targ_lang: str
    overall_score: Optional[float] = None
    regions_total: Optional[int] = None
    rendered: Optional[int] = None
    dnt: Optional[int] = None

class ProjectCreate(BaseModel):
    name: str
    target_lang: str
    asset_kind: str = "image"  ## "image" | "video" — video pipeline lands later

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    target_lang: Optional[str] = None
    source_lang: Optional[str] = None
    ground_truth: Optional[List[str]] = None
    archived: Optional[bool] = None

class SnapshotCreate(BaseModel):
    asset_id: str
    reason: str = "manual"

class RegionCreate(BaseModel):
    x: int
    y: int
    width: int
    height: int
    text: Optional[str] = None
    target_text: Optional[str] = None

class RegionMerge(BaseModel):
    # No `reread` switch. It defaulted to True and no client ever sent it, so
    # the join-only branch was unreachable -- untested surface that read as a
    # supported option. cicerone.reread_merged_region already falls back to
    # joining the parts whenever the whole-region read fails to beat them, so
    # the choice this exposed was never the caller's to make.
    region_ids: List[str]

class RegionUpdate(BaseModel):
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    text: Optional[str] = None
    target_text: Optional[str] = None
    dnt: Optional[bool] = None
    target_language: Optional[str] = None
    language: Optional[str] = None  ## per-region source language
    font: Optional[str] = None      ## per-region font (style_profile.font_family)
    excluded: Optional[bool] = None
    target_orientation: Optional[str] = None
    word_order: Optional[str] = None
    segmentation_mask: Optional[Dict[str, Any]] = None

class SemanticSubstitutionRequest(BaseModel):
    """A user-supplied phrase may have a grammatical order that differs
    from its source sign's spatial order.  ``apply`` is deliberately false
    by default: planning must never overwrite individual target fields."""
    target_text: str
    targ_lang: Optional[str] = None
    apply: bool = False


class SemanticRepairRequest(BaseModel):
    """The user's verdict on a proposed cross-region source correction.

    Basil only ever proposes: when a gazetteer entity is spelled across
    fragmented regions and one of them was misrecognised, the corrected
    reading waits here until someone says so explicitly.
    """
    accepted: bool


class TranslationRunRequest(BaseModel):
    region_ids: Optional[List[str]] = None
    target_lang: Optional[str] = None


class TranslationDecisionItem(BaseModel):
    region_id: str
    action: str
    attempt_id: Optional[str] = None
    text: Optional[str] = None


class TranslationDecisionRequest(BaseModel):
    manifest_revision: str
    decisions: List[TranslationDecisionItem]


class VideoJobCreate(BaseModel):
    asset_id: str
    project_id: Optional[str] = None
    chunk_size: int = 240


class VideoKeyframeRequest(BaseModel):
    id: Optional[str] = None
    track_id: str
    frame_index: int
    scope: str = "frame"
    end_frame: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    quad: Optional[List[List[float]]] = None
    opacity: Optional[float] = None
    style: Optional[Dict[str, Any]] = None
    effects: Optional[Dict[str, Any]] = None
    mask_path: Optional[str] = None
    expected_job_revision: Optional[int] = None


class VideoTrackUpdate(BaseModel):
    source_text: Optional[str] = None
    target_text: Optional[str] = None
    target_language: Optional[str] = None
    status: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    expected_revision: Optional[int] = None


class VideoRenderRequest(BaseModel):
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


# --- upload + languages + fonts ---

@app.post("/api/assets")
async def upload_asset(file: UploadFile = File(...), project_id: Optional[str] = None):
    if project_id and db.get_project(project_id) is None:
        raise HTTPException(404, f"project '{project_id}' not found")
    suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
    asset_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{asset_id}{suffix}"
    hasher = hashlib.sha256(); uploaded_bytes = 0
    max_upload = int(os.environ.get("TOFU_MAX_UPLOAD_BYTES", str(100 * 1024**3)))
    try:
        with dest.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                uploaded_bytes += len(chunk)
                if uploaded_bytes > max_upload:
                    raise HTTPException(413, f"asset exceeds configured upload limit ({max_upload} bytes)")
                hasher.update(chunk); output.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    content_hash = hasher.hexdigest()
    info = infer_asset_info(str(dest))

    # persist an empty manifest immediately so GET /api/manifest never 404s
    # ("failed to fetch manifest" on fresh uploads) and autosave has a target
    img_dim = None
    try:
        from PIL import Image
        with Image.open(dest) as im:
            img_dim = (im.width, im.height)
    except Exception:
        pass
    # reject undecodable image uploads outright — every downstream layer
    # (scan, detect, render) would 500 on them otherwise
    if info.asset_type.value == "image" and img_dim is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(415, "file is not a decodable image")
    empty = TextManifest(
        asset_id=asset_id, total_regions=0, instances=[],
        img_dim=img_dim, asset_type=info.asset_type,
        frame_count=info.frame_count, fps=info.fps, duration=info.duration,
    )
    save_manifest(UPLOAD_DIR, asset_id, empty)

    if project_id:
        db.link_asset(project_id, asset_id, file.filename, content_hash)
        db.log_event(project_id, "asset-uploaded",
                     f"uploaded '{file.filename}' ({asset_id})")
    return {
        "asset_id": asset_id, "filename": file.filename,
        "asset_info": jsonable(info),
        "asset_url": f"/uploads/{asset_id}{suffix}",
    }


def _prepare_video_job(job_id: str, source: Path, *, operation_id: Optional[str] = None,
                       worker_id: Optional[str] = None, cancellation_generation: int = 0,
                       chunk_size: int = 240, mode: str = "restart") -> None:
    """Background ingest stage; state is durable across client disconnects."""
    try:
        job = db.update_video_job(job_id, status="running", stage="proxy", progress=0.05)
        if not job or job.get("cancel_requested"):
            db.update_video_job(job_id, status="cancelled", stage="cancelled")
            return
        artifact_dir = OUTPUT_DIR / "video" / job_id
        proxy = artifact_dir / "proxy.mp4"
        ## Re-encoding the whole clip on every resume would cost more than the
        ## analysis a resume exists to save. The proxy depends only on the
        ## source, so an existing one is still correct.
        if not (mode == "resume" and proxy.is_file()):
            make_proxy(source, proxy)
        now = time.time()
        with db._conn() as con:
            con.execute("INSERT OR REPLACE INTO video_artifacts (id,job_id,kind,path,status,created_at) VALUES (?,?,?,?,?,?)",
                        (f"{job_id}-proxy", job_id, "proxy", str(proxy), "ready", now))
        current = db.get_video_job(job_id)
        if not current or current.get("cancel_requested"):
            db.update_video_job(job_id, status="cancelled", stage="cancelled")
            return
        db.update_video_job(job_id, status="running", stage="analysis", progress=0.15)
        vm = VideoManifest(**current["manifest"])
        def cancelled() -> bool:
            job_cancelled = bool((db.get_video_job(job_id) or {}).get("cancel_requested"))
            operation_cancelled = bool(operation_id and db.video_operation_cancelled(operation_id, cancellation_generation))
            return job_cancelled or operation_cancelled or VIDEO_WORKER_STOP.is_set()
        def report(value: float) -> None:
            db.update_video_job(job_id, progress=.15 + .7 * value)
            if operation_id and worker_id: db.heartbeat_video_operation(operation_id, worker_id)
        revision = int(current.get("dependency_revision", 1))
        fingerprint = video_source_fingerprint(source)
        ## Source-language priority applies to video exactly as it does to
        ## stills: the charset drives what the recognizer can read, and every
        ## keyframe here goes through the same cicerone.detect() as an image.
        ## Analysis ran language-blind until this was threaded through.
        hints = _project_lang_hints(current["asset_id"])
        language_params = list(hints) if hints else None
        state = None
        resume_language_changed = False
        if mode == "resume":
            state = TrackerCheckpoint.from_dict(db.latest_video_checkpoint(
                job_id, analyzer_revision=ANALYZER_REVISION, dependency_revision=revision,
                source_fingerprint=fingerprint))
            ## A different reader charset produces different text for the same
            ## pixels, so resuming across a language change would stitch two
            ## incompatible analyses into one timeline. Drop the checkpoint and
            ## pay for a cold restart instead -- the same trade the
            ## ResumeMismatch path already makes for decoder desynchronization.
            if state is not None and (state.params or {}).get("languages") != language_params:
                resume_language_changed = True
                state = None
        ## Scoped when there is something to keep, total when there is not. The
        ## unconditional wipe that used to live here is what made every resume a
        ## restart from frame zero.
        db.reset_video_analysis(job_id, from_frame=state.next_frame if state else 0)

        def checkpoint(chunk_index: int, end_frame: int, chunk_tracks: List[Dict[str, Any]],
                       chunk_observations: List[Dict[str, Any]], tracker: TrackerCheckpoint) -> None:
            digest = hashlib.sha256(json.dumps(chunk_observations, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            db.commit_video_analysis_chunk(job_id, chunk_index, chunk_tracks, chunk_observations,
                                           digest, revision, checkpoint=tracker.to_dict(),
                                           analyzer_revision=ANALYZER_REVISION)
            if operation_id and worker_id: db.heartbeat_video_operation(operation_id, worker_id)

        ## Every resume leaves a trace either way. Determinism is the claim this
        ## architecture makes; a resume whose outcome is not recorded cannot be
        ## audited when the export looks wrong.
        provenance: List[Dict[str, Any]] = []
        if resume_language_changed:
            provenance.append(
                {"code": "resume_language_changed", "severity": "warning", "frame_index": 0,
                 "detail": f"project source language is now {language_params or 'auto'}; "
                           "the checkpoint was analyzed with a different reader charset, "
                           "so analysis restarted from the first frame"})
        try:
            analyze_video(
                str(source), vm, languages=hints, cancelled=cancelled, progress=report,
                checkpoint=checkpoint,
                chunk_size=chunk_size, resume=state, job_id=job_id, fingerprint=fingerprint,
                seek_mode=str((state.params or {}).get("seek_mode", "fast")) if state else "fast",
                on_resume=lambda evidence: provenance.append(
                    {"code": "resume_reconciled", "severity": "info",
                     "frame_index": state.next_frame if state else 0,
                     "detail": json.dumps(evidence, default=str)[:500]}),
            )
        except ResumeMismatch as mismatch:
            ## Never continue on a mismatch. A silently degraded resume produces
            ## track-identity churn, which reaches the viewer as duplicated or
            ## flickering subtitles -- far worse than redoing the work.
            provenance = [{"code": "resume_cold_restart", "severity": "warning",
                           "frame_index": state.next_frame if state else 0,
                           "detail": f"{mismatch.reason}; reanalyzed from the start"}]
            db.reset_video_analysis(job_id, from_frame=0)
            analyze_video(str(source), vm, languages=hints, cancelled=cancelled, progress=report,
                          checkpoint=checkpoint, chunk_size=chunk_size,
                          job_id=job_id, fingerprint=fingerprint)
        ## Settle over the ROWS, not over whatever this process happens to hold:
        ## after a resume the earlier chunks' tracks were never in this memory.
        settled, issues = settle_tracks(db.list_video_tracks(job_id))
        db.finalize_video_analysis(job_id, settled, issues + provenance)
        db.mark_video_upgrade(job_id, completed=True)
        db.update_video_job(job_id, status="ready", stage="review", progress=1.0)
    except InterruptedError:
        db.update_video_job(job_id, status="cancelled", stage="cancelled")
    except Exception as exc:
        db.update_video_job(job_id, status="failed", stage="ingest", error=f"{type(exc).__name__}: {exc}")
        raise


def _require_video_disk(source: Path, multiplier: float, label: str) -> Dict[str, int]:
    source_bytes = source.stat().st_size
    required = max(512 * 1024**2, int(source_bytes * multiplier))
    free = shutil.disk_usage(OUTPUT_DIR).free
    if free < required:
        raise HTTPException(507, f"insufficient disk for {label}: requires about {required} bytes, {free} available")
    return {"source_bytes": source_bytes, "estimated_required_bytes": required, "available_bytes": free}


@app.post("/api/video/jobs")
def create_video_job(req: VideoJobCreate):
    path = _asset_path(req.asset_id)
    if infer_asset_info(str(path)).asset_type.value != "video":
        raise HTTPException(400, "asset is not a supported video")
    if not video_media_available():
        raise HTTPException(503, "video ingest requires ffmpeg and ffprobe on PATH")
    try:
        manifest = probe(path, req.asset_id).to_dict()
    except MediaError as exc:
        raise HTTPException(415, str(exc)) from exc
    estimate = _require_video_disk(path, 2.5, "video analysis")
    manifest["storage_estimate"] = estimate
    job = db.create_video_job(req.asset_id, req.project_id, manifest, max(1, min(req.chunk_size, 1000)))
    revision = int(job.get("dependency_revision", 1))
    db.create_video_operation(job["id"], "analysis", 0, max(0, manifest["frame_count"] - 1),
                              spec={"asset_id": req.asset_id, "chunk_size": max(1, min(req.chunk_size, 1000))}, dependency_revision=revision,
                              dedupe_key=f"analysis:{revision}")
    return job


@app.get("/api/video/jobs/{job_id}")
def get_video_job(job_id: str):
    job = db.get_video_job(job_id)
    if not job:
        raise HTTPException(404, "video job not found")
    job["proxy_url"] = db.video_artifact_url(job_id, "proxy")
    job["preview_url"] = db.video_artifact_url(job_id, "preview")
    job["export_url"] = db.video_artifact_url(job_id, "export")
    return job


@app.get("/api/video/assets/{asset_id}/latest-job")
def get_latest_video_job(asset_id: str):
    job = db.latest_video_job(asset_id)
    if not job:
        raise HTTPException(404, "video job not found")
    job["proxy_url"] = db.video_artifact_url(job["id"], "proxy")
    job["preview_url"] = db.video_artifact_url(job["id"], "preview")
    job["export_url"] = db.video_artifact_url(job["id"], "export")
    return job


@app.get("/api/video/jobs/{job_id}/events")
def stream_video_job(job_id: str):
    if not db.get_video_job(job_id):
        raise HTTPException(404, "video job not found")
    def events():
        last = None
        while True:
            job = db.get_video_job(job_id)
            if not job:
                yield "event: error\ndata: {\"detail\":\"job removed\"}\n\n"
                return
            snapshot = {k: job.get(k) for k in ("id", "status", "stage", "progress", "error", "updated_at")}
            encoded = json.dumps(snapshot)
            if encoded != last:
                yield f"event: progress\ndata: {encoded}\n\n"
                last = encoded
            if job["status"] in {"ready", "completed", "failed", "cancelled"}:
                yield f"event: terminal\ndata: {encoded}\n\n"
                return
            time.sleep(.5)
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/video/jobs/{job_id}/cancel")
def cancel_video_job(job_id: str):
    if not db.get_video_job(job_id):
        raise HTTPException(404, "video job not found")
    db.cancel_video_operations(job_id)
    return db.update_video_job(job_id, cancel_requested=1, status="cancelling", stage="cancelling")


@app.post("/api/video/jobs/{job_id}/resume")
def resume_video_job(job_id: str):
    job = db.get_video_job(job_id)
    if not job:
        raise HTTPException(404, "video job not found")
    if job["status"] == "running":
        raise HTTPException(409, "video job is already running")
    revision = int(job.get("dependency_revision", 1))
    chunk_size = int(job.get("chunk_size") or 240)
    state = db.latest_video_checkpoint(
        job_id, analyzer_revision=ANALYZER_REVISION, dependency_revision=revision,
        source_fingerprint=video_source_fingerprint(_asset_path(job["asset_id"])))
    resume_from = int(state.get("next_frame", 0)) if state else 0
    db.update_video_job(job_id, cancel_requested=0, status="queued", error=None)
    ## Without a dedupe key two clicks queued two analysis operations for one job.
    operation = db.create_video_operation(
        job_id, "analysis", resume_from, max(0, int(job["manifest"].get("frame_count", 0)) - 1),
        spec={"asset_id": job["asset_id"], "chunk_size": chunk_size, "mode": "resume"},
        dependency_revision=revision,
        dedupe_key=f"analysis:{revision}:resume:{state.get('chunk_index', -1) if state else -1}")
    return {**(db.get_video_job(job_id) or {}), "operation_id": operation,
            "resumed_from_frame": resume_from, **db.video_analysis_progress(job_id)}


@app.post("/api/video/jobs/{job_id}/upgrade")
def upgrade_video_job(job_id: str):
    job = db.get_video_job(job_id)
    if not job: raise HTTPException(404, "video job not found")
    db.mark_video_upgrade(job_id, completed=False)
    revision = int(job.get("dependency_revision", 1)) + 1
    db.update_video_job(job_id, cancel_requested=0, status="queued", stage="analysis_upgrade", error=None)
    db.create_video_operation(job_id, "analysis", 0, max(0, int(job["manifest"].get("frame_count", 0))-1),
                              spec={"asset_id": job["asset_id"], "chunk_size": int(job.get("chunk_size") or 240), "upgrade": True},
                              dependency_revision=revision)
    return db.get_video_job(job_id)


@app.get("/api/video/jobs/{job_id}/timeline")
def get_video_timeline(job_id: str, start: int = 0, end: int = 300, limit: int = 1000, cursor: int = 0):
    if not db.get_video_job(job_id):
        raise HTTPException(404, "video job not found")
    if start < 0 or end < start:
        raise HTTPException(400, "invalid frame range")
    if cursor < 0: raise HTTPException(400, "invalid timeline cursor")
    return db.list_video_timeline(job_id, start, end, max(1, min(limit, 5000)), cursor)


@app.put("/api/video/jobs/{job_id}/keyframes")
def put_video_keyframe(job_id: str, req: VideoKeyframeRequest):
    if req.scope not in {"frame", "range", "track"}:
        raise HTTPException(400, "scope must be frame, range, or track")
    if req.scope == "range" and (req.end_frame is None or req.end_frame < req.frame_index):
        raise HTTPException(400, "range scope requires a valid end_frame")
    frame_count = db.video_frame_count(job_id)
    if frame_count is None:
        raise HTTPException(404, "video job not found")
    track = db.get_video_track(job_id, req.track_id)
    if not track:
        raise HTTPException(404, "video track not found")
    job = db.get_video_job(job_id)
    if req.expected_job_revision is not None and job and int(job.get("dependency_revision", 1)) != req.expected_job_revision:
        raise HTTPException(409, {"code": "stale_video_revision", "current_revision": job.get("dependency_revision")})
    end = req.end_frame if req.end_frame is not None else req.frame_index
    if req.frame_index < 0 or end >= frame_count:
        raise HTTPException(400, "keyframe range is outside the video")
    if req.scope != "track" and (req.frame_index < track["start_frame"] or end > track["end_frame"]):
        raise HTTPException(400, "keyframe range is outside the track")
    if req.bbox is not None:
        values = (req.bbox.get("x"), req.bbox.get("y"), req.bbox.get("width"), req.bbox.get("height"))
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
            raise HTTPException(400, "bbox must contain finite positive geometry")
    if req.mask_path:
        candidate = Path(req.mask_path).resolve(); owned = (OUTPUT_DIR / "video" / job_id).resolve()
        try: candidate.relative_to(owned)
        except ValueError as exc: raise HTTPException(400, "mask must be an artifact owned by this video job") from exc
        if not candidate.is_file(): raise HTTPException(400, "mask artifact does not exist")
    data = req.model_dump(exclude={"expected_job_revision"})
    data["id"] = data["id"] or uuid.uuid4().hex[:16]
    data["created_at"] = time.time()
    db.upsert_video_keyframe(job_id, data)
    return data


@app.patch("/api/video/jobs/{job_id}/tracks/{track_id}")
def patch_video_track(job_id: str, track_id: str, req: VideoTrackUpdate):
    track_before = db.get_video_track(job_id, track_id)
    if not track_before: raise HTTPException(404, "video track not found")
    if req.expected_revision is not None and int(track_before.get("revision", 1)) != req.expected_revision:
        raise HTTPException(409, {"code": "stale_track_revision", "current_revision": track_before.get("revision")})
    changes = req.model_dump(exclude_unset=True, exclude={"expected_revision"})
    if "status" in changes and changes["status"] not in {"recognized", "review_required", "translated", "approved", "excluded"}:
        raise HTTPException(400, "invalid track status")
    font = (changes.get("style") or {}).get("font_family") if isinstance(changes.get("style"), dict) else None
    if font and Path(font).is_absolute() and not _is_served_font(Path(font)):
        raise HTTPException(400, "font path is outside the served font library")
    track = db.update_video_track(job_id, track_id, changes)
    if not track:
        raise HTTPException(404, "video track not found")
    return track


def _render_video_artifact(job_id: str, kind: str, start: int, end: int, operation_id: Optional[str] = None,
                           worker_id: Optional[str] = None, cancellation_generation: int = 0) -> None:
    temp: Optional[Path] = None
    try:
        if operation_id: db.update_video_operation(operation_id, "running")
        job = db.get_video_job(job_id)
        if not job: return
        source = _asset_path(job["asset_id"]); data = db.video_composition_data(job_id, start, end)
        folder = OUTPUT_DIR / "video" / job_id; folder.mkdir(parents=True, exist_ok=True)
        ## The artifact is named for the PLAN, not just the job revision. The
        ## plan hashes everything the render is a function of -- track edits,
        ## keyframes, analyzer and renderer revisions, the font library -- so a
        ## filename can no longer promise output the inputs no longer produce.
        inputs = video_plan_inputs(job, data["tracks"], data["keyframes"],
                                   analyzer_revision=ANALYZER_REVISION,
                                   font_registry=get_validator().font_registry)
        revision = inputs.revision; artifact_token = operation_id or uuid.uuid4().hex[:12]
        temp = folder / f"{kind}-r{revision}-{artifact_token}-intermediate.mp4"
        destination = folder / f"{kind}-r{revision}-{artifact_token}.mp4"
        db.update_video_job(job_id, stage=f"{kind}_render", progress=0.0, error=None)
        with tempfile.TemporaryDirectory(prefix=f"tofu-{kind}-frames-", dir=folder) as frame_temp:
            frame_dir = Path(frame_temp)
            compose_video(source, temp, job["manifest"], data["tracks"], data["observations"], data["keyframes"],
                          start_frame=start, end_frame=end, frame_dir=frame_dir, inputs=inputs,
                          progress=lambda value: (db.update_video_job(job_id, progress=.75*value),
                                                  operation_id and worker_id and db.heartbeat_video_operation(operation_id, worker_id)),
                          cancelled=lambda: bool((db.get_video_job(job_id) or {}).get("cancel_requested")) or
                              bool(operation_id and db.video_operation_cancelled(operation_id, cancellation_generation)) or VIDEO_WORKER_STOP.is_set())
            db.update_video_job(job_id, stage=f"{kind}_encode", progress=.8)
            pts = job["manifest"].get("frame_pts") or []
            segment_pts = pts[start:end+1] if pts else [i/(job["manifest"].get("fps") or 30) for i in range(end-start+1)]
            encode_vfr_sequence(frame_dir, segment_pts, temp)
        if kind == "export": mux_rendered_video(temp, source, destination,
                                                 color=job["manifest"].get("color"),
                                                 rotation=int(job["manifest"].get("rotation") or 0))
        else:
            pts = job["manifest"].get("frame_pts") or []
            start_seconds = pts[start] if start < len(pts) else start/(job["manifest"].get("fps") or 30)
            end_seconds = pts[end] if end < len(pts) else end/(job["manifest"].get("fps") or 30)
            last_delta = (pts[end]-pts[end-1]) if pts and 0 < end < len(pts) else 1/(job["manifest"].get("fps") or 30)
            mux_preview_audio(temp, source, destination, start_seconds, max(.001,end_seconds-start_seconds+last_delta))
        pts = job["manifest"].get("frame_pts") or []
        last_delta = (pts[end]-pts[end-1]) if pts and 0 < end < len(pts) else 1/(job["manifest"].get("fps") or 30)
        expected_duration = ((pts[end]-pts[start]+last_delta) if pts and end < len(pts) else (end-start+1)/(job["manifest"].get("fps") or 30))
        verification = verify_output(destination, expected_width=int(job["manifest"]["width"]),
                                     expected_height=int(job["manifest"]["height"]),
                                     expected_duration=max(.001, expected_duration), max_drift=.02)
        db.save_video_artifact(job_id, kind, str(destination), verification)
        for obsolete in db.prune_video_artifact_records(job_id, kind, 2 if kind == "export" else 1):
            candidate = Path(obsolete)
            try: candidate.resolve().relative_to((OUTPUT_DIR / "video" / job_id).resolve())
            except ValueError: continue
            candidate.unlink(missing_ok=True)
        db.update_video_job(job_id, status="ready", stage="review", progress=1.0)
        if operation_id: db.update_video_operation(operation_id, "completed")
    except InterruptedError:
        db.update_video_job(job_id, status="cancelled", stage="cancelled")
        if operation_id: db.update_video_operation(operation_id, "cancelled")
    except Exception as exc:
        db.update_video_job(job_id, status="ready", stage="review", error=f"{kind} failed: {type(exc).__name__}: {exc}")
        if operation_id: db.update_video_operation(operation_id, "failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if temp: temp.unlink(missing_ok=True)


@app.post("/api/video/jobs/{job_id}/preview")
def render_video_preview(job_id: str, req: VideoRenderRequest):
    count = db.video_frame_count(job_id)
    if count is None: raise HTTPException(404, "video job not found")
    start = max(0, req.start_frame or 0); end = min(count-1, req.end_frame if req.end_frame is not None else start+150)
    if end < start: raise HTTPException(400, "invalid preview range")
    job = db.get_video_job(job_id)
    if job: _require_video_disk(_asset_path(job["asset_id"]), .4, "video preview")
    job = db.get_video_job(job_id); revision = int(job.get("dependency_revision", 1)) if job else 1
    db.create_video_operation(job_id, "preview", start, end, spec={"start": start, "end": end}, dependency_revision=revision)
    return db.update_video_job(job_id, status="rendering", stage="preview_queued", progress=0.0)


@app.post("/api/video/jobs/{job_id}/export")
def export_video(job_id: str):
    count = db.video_frame_count(job_id)
    if count is None: raise HTTPException(404, "video job not found")
    job = db.get_video_job(job_id)
    if job: _require_video_disk(_asset_path(job["asset_id"]), 1.75, "video export")
    job = db.get_video_job(job_id); revision = int(job.get("dependency_revision", 1)) if job else 1
    db.create_video_operation(job_id, "export", 0, count-1, spec={"container": "mp4", "codec": "h264"},
                              dependency_revision=revision, dedupe_key=f"export:{revision}:mp4:h264")
    return db.update_video_job(job_id, status="rendering", stage="export_queued", progress=0.0)


def _video_worker_loop(worker_id: str) -> None:
    while not VIDEO_WORKER_STOP.is_set():
        operation = db.claim_video_operation(worker_id)
        if not operation:
            VIDEO_WORKER_STOP.wait(.5); continue
        op_id = operation["id"]; generation = int(operation.get("cancellation_generation", 0))
        try:
            if operation["kind"] == "analysis":
                job = db.get_video_job(operation["job_id"])
                if not job: raise RuntimeError("video job disappeared")
                _prepare_video_job(job["id"], _asset_path(job["asset_id"]), operation_id=op_id,
                                   worker_id=worker_id, cancellation_generation=generation,
                                   chunk_size=int(operation.get("spec", {}).get("chunk_size")
                                                  or job.get("chunk_size") or 240),
                                   mode=str(operation.get("spec", {}).get("mode", "restart")))
            elif operation["kind"] in {"preview", "export"}:
                _render_video_artifact(operation["job_id"], operation["kind"], operation["start_frame"], operation["end_frame"],
                                       operation_id=op_id, worker_id=worker_id, cancellation_generation=generation)
            else: raise RuntimeError(f"unknown video operation kind {operation['kind']}")
            status = "cancelled" if db.video_operation_cancelled(op_id, generation) else "completed"
            db.finish_video_operation(op_id, worker_id, status)
        except Exception as exc:
            db.finish_video_operation(op_id, worker_id, "failed", f"{type(exc).__name__}: {exc}")


@app.get("/api/assets/check-duplicate")
async def check_duplicate_asset(hash: str):
    match = db.find_asset_by_hash(hash)
    if match is None:
        return {"duplicate": False, "project_id": None, "project_name": None}
    return {
        "duplicate": True,
        "project_id": match["project_id"],
        "project_name": match["project_name"],
    }


@app.get("/api/languages")
def languages():
    return {
        "languages": [
            {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
            for code in sorted(lang_to_script.keys())
        ]
    }


# The full catalog is the whole installed library sifted and scored, which
# is a real cost the first time and free afterwards: the registry is a
# process singleton and the fonts on disk do not change while we run.
_CATALOG_CACHE: Dict[Tuple[str, str], List[dict]] = {}


@app.get("/api/fonts")
def fonts(lang: str, limit: int = 24, full: bool = False):
    """Ranked fonts for one target language.

    Two modes, deliberately distinct.  The default is the dropdown's short
    ranked list, capped at ``limit`` families.  ``full=true`` is the Font
    Manager's complete catalog: every family, each tagged with a browsing
    category, and no ``fonts`` face list (the manager picks per family, so
    computing a flat ranked face list for it would be pure waste).
    """
    script = lang_to_script.get(lang)
    if script is None:
        raise HTTPException(400, f"unknown language '{lang}'")
    validator = get_validator()
    if validator.font_registry is None:
        return {"script": script, "fonts": [], "families": []}

    if full:
        cached = _CATALOG_CACHE.get((script, lang))
        if cached is None:
            cached = validator.font_registry.families_with_weights(
                script, lang, limit=None
            )
            for fam in cached:
                fam["category"] = sift(fam["best_path"], fam["family"])
            _CATALOG_CACHE[(script, lang)] = cached
        return {"script": script, "fonts": [], "families": cached}

    ranked = validator.font_registry.recommend(script, lang, limit=limit)
    families = validator.font_registry.families_with_weights(script, lang, limit=limit)
    for fam in families:
        fam["category"] = sift(fam["best_path"], fam["family"])
    return {
        "script": script,
        "fonts": [{"path": p, "coverage": round(c, 4)} for p, c in ranked],
        "families": families,
    }


# OS/2 fsType, the vendor's embedding permission. Bit 1 is the one that
# matters: the OpenType spec makes Restricted exclusive of the others, but
# real fonts do not honour that -- measured across 337 installed faces, one
# ships fsType=14, which is Restricted AND Preview&Print AND Editable at
# once. A check that tests "is Preview&Print or Editable set" passes that
# font. Bit 1 is therefore tested FIRST and alone.
FSTYPE_RESTRICTED = 0x0002
FSTYPE_PREVIEW_PRINT = 0x0004
FSTYPE_EDITABLE = 0x0008
FSTYPE_NO_SUBSET = 0x0100
FSTYPE_BITMAP_ONLY = 0x0200


def _embedding_permission(path: Path, face_part: str = "") -> Dict[str, Any]:
    """What the font's vendor permits, read from OS/2 fsType.

    Fails OPEN on an unreadable table: a font whose permissions cannot be
    determined is treated as installable, matching how the rest of the
    registry degrades. The alternative -- refusing everything unparseable
    -- would break every face on a machine without fontTools.
    """
    result = {
        "fs_type": None, "restricted": False,
        "subsettable": True, "bitmap_only": False, "readable": False,
    }
    try:
        from fontTools.ttLib import TTCollection, TTFont

        if face_part.isdigit():
            font = TTCollection(str(path), lazy=True).fonts[int(face_part)]
        else:
            font = TTFont(str(path), lazy=True, fontNumber=0)
        if "OS/2" not in font:
            return result
        value = int(font["OS/2"].fsType or 0)
    except Exception:
        return result
    result.update({
        "fs_type": value,
        "readable": True,
        "restricted": bool(value & FSTYPE_RESTRICTED),
        "subsettable": not (value & FSTYPE_NO_SUBSET),
        "bitmap_only": bool(value & FSTYPE_BITMAP_ONLY),
    })
    return result


def _unsafe_archive_member(name: str) -> bool:
    """Would extracting this member write outside the target directory?

    Deliberately string-level rather than via pathlib. On Windows
    ``Path("/abs/evil.ttf").is_absolute()`` is **False** -- an absolute
    Windows path needs a drive letter -- so a root-anchored POSIX member
    sails straight through the obvious check. Zip entries are also
    specified with forward slashes, but nothing stops a hostile archive
    using backslashes, which pathlib then treats as separators on Windows
    and as ordinary characters elsewhere.

    So: reject a leading separator of either kind, a drive letter, and any
    '..' segment under either separator.
    """
    if not name:
        return True
    normalised = name.replace("\\", "/")
    if normalised.startswith("/"):
        return True
    if len(name) > 1 and name[1] == ":":          # C:\... or C:/...
        return True
    return ".." in normalised.split("/")


def _register_font_dir(directory: Path) -> int:
    """Discover a directory into the live registry and invalidate caches.

    _CATALOG_CACHE's own comment says it is safe because "fonts on disk do
    not change while we run". Runtime installation is precisely what breaks
    that, so every path that adds or removes faces clears it here rather
    than relying on each endpoint to remember.

    Scribe's _font_cache deliberately is NOT cleared: it keys on
    (font_family, size), a newly installed file is a new path, and a new
    path is a cache miss. It would only go stale if a file were REPLACED at
    an existing path, which the upload endpoint refuses to do.
    """
    validator = get_validator()
    registry = validator.font_registry
    if registry is None:
        return 0
    loaded = registry.discover(str(directory))
    _CATALOG_CACHE.clear()
    return loaded


@app.post("/api/fonts/upload")
async def upload_font(file: UploadFile = File(...)):
    """Install one font file for this server, available immediately.

    No OS-level installation is involved, and none is needed: scribe
    renders through ImageFont.truetype(path), the preview is served over
    /api/font-file, and FontRegistry keys on absolute paths. The system
    font directory was only ever a discovery convenience.
    """
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() not in FONT_EXTS:
        raise HTTPException(
            400, f"expected a font file ({', '.join(sorted(FONT_EXTS))})"
        )
    target_dir = writable_pantry(USER_SUBDIR)
    target = target_dir / name
    if target.exists():
        # Replacing a file in place would strand scribe's (font_family,
        # size) cache on the previous bytes for the rest of the process.
        raise HTTPException(409, f"'{name}' is already installed")

    target.write_bytes(await file.read())
    permission = _embedding_permission(target)
    registry = get_validator().font_registry
    if registry is None:
        target.unlink(missing_ok=True)
        raise HTTPException(503, "no font registry on this server")
    loaded = registry.load_font(str(target))
    if not loaded:
        # A file fontTools cannot parse would fail every future discover()
        # of this directory, silently, forever.
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"'{name}' could not be read as a font")
    _CATALOG_CACHE.clear()
    return {
        "installed": name, "faces": loaded, "path": str(target),
        "source": USER_SUBDIR, "embedding": permission,
        # Restricted fonts install and render locally; what they must not do
        # is get served to browsers for preview. Said plainly rather than
        # dropped silently.
        "preview_blocked": permission["restricted"],
    }


@app.get("/api/fonts/packs")
def list_font_packs():
    packs = []
    root = ROOT / "server" / "font-library" / PACKS_SUBDIR
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            meta: Dict[str, Any] = {"name": entry.name}
            manifest = entry / "pack.json"
            if manifest.is_file():
                try:
                    meta.update(json.loads(manifest.read_text(encoding="utf-8")))
                except Exception:
                    meta["error"] = "pack.json is unreadable"
            meta["face_files"] = sum(
                1 for p in entry.rglob("*") if p.suffix.lower() in FONT_EXTS
            )
            packs.append(meta)
    return {"packs": packs}


@app.post("/api/fonts/packs/install")
async def install_font_pack(file: UploadFile = File(...)):
    """Install a zip of fonts plus an optional pack.json."""
    import zipfile

    name = Path(Path(file.filename or "").name).stem
    if not name:
        raise HTTPException(400, "pack needs a filename")
    root = writable_pantry(PACKS_SUBDIR)
    target = root / name
    if target.exists():
        raise HTTPException(409, f"pack '{name}' is already installed")

    payload = io.BytesIO(await file.read())
    try:
        archive = zipfile.ZipFile(payload)
    except zipfile.BadZipFile:
        raise HTTPException(400, "pack must be a zip archive")
    for member in archive.namelist():
        # Zip Slip: an archive member may name an absolute path or climb
        # out with '..', and extractall would happily write there.
        if _unsafe_archive_member(member):
            raise HTTPException(400, f"unsafe path in archive: {member}")
    target.mkdir(parents=True)
    archive.extractall(target)

    loaded = _register_font_dir(target)
    if not loaded:
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(400, "archive contained no readable fonts")
    return {"installed": name, "faces": loaded}


@app.delete("/api/fonts/packs/{name}")
def remove_font_pack(name: str):
    safe = Path(name).name
    target = ROOT / "server" / "font-library" / PACKS_SUBDIR / safe
    if not target.is_dir():
        raise HTTPException(404, f"no pack '{safe}'")
    registry = get_validator().font_registry
    removed = 0
    if registry is not None:
        for path in target.rglob("*"):
            if path.suffix.lower() in FONT_EXTS:
                # Without this the registry keeps offering families whose
                # files are gone, and the failure surfaces at render time
                # inside ImageFont.truetype instead of at the choice.
                removed += registry.unload_font(str(path))
    shutil.rmtree(target, ignore_errors=True)
    _CATALOG_CACHE.clear()
    return {"removed": safe, "faces_unloaded": removed}


_FACE_CACHE: Dict[str, bytes] = {}


@app.get("/api/font-file")
def serve_font_file(path: str):
    """serve a font file for @font-face preview in the frontend.

    FontRegistry keys the Nth face of a collection as "<file>#<n>"
    (fonts.py's load_font), and those keys are exactly what reaches the
    client on style_profile.font_family / resolved_font_family. Path(...)
    on such a key is not a file, so every collection face used to 404 --
    and CJK families on Windows are overwhelmingly .ttc, which is
    precisely where the preview most needs the real face.

    A browser @font-face cannot select a face INSIDE a collection, so
    serving the whole .ttc would silently preview face 0: msgothic.ttc#1
    (MS UI Gothic) drawn as face 0 (MS Gothic) is a different typeface at
    different metrics, which desynchronizes both the fit measurement and
    the paint from what scribe draws. Extract the requested face into a
    standalone font instead. Cached in-process because the extraction is
    pure CPU over an unchanging file and the panel preloads whole
    families at once.
    """
    file_part, _, face_part = path.partition("#")
    p = Path(file_part)
    if not p.is_file():
        raise HTTPException(404, "font not found")
    if not _is_served_font(p):
        # 404 rather than 403: whether some path outside the font roots
        # exists is not this endpoint's to disclose.
        raise HTTPException(404, "font not found")
    if _embedding_permission(p, face_part)["restricted"]:
        # fsType bit 1. Serving the bytes for @font-face preview puts the
        # whole face on every client that asks; a face whose vendor
        # forbade embedding must not be redistributed by us.
        raise HTTPException(403, "this font's embedding permissions forbid serving it")
    if not face_part:
        ext = p.suffix.lower()
        media = {
            ".ttf": "font/ttf",
            ".otf": "font/otf",
            ".ttc": "font/collection",
            ".otc": "font/collection",
        }.get(ext, "application/octet-stream")
        return Response(content=p.read_bytes(), media_type=media)

    cached = _FACE_CACHE.get(path)
    if cached is None:
        try:
            from fontTools.ttLib import TTCollection

            faces = TTCollection(str(p)).fonts
            face = faces[int(face_part)]
        except (ImportError, ValueError, IndexError) as exc:
            raise HTTPException(404, f"font face not found: {exc}")
        buf = io.BytesIO()
        face.save(buf)
        cached = buf.getvalue()
        _FACE_CACHE[path] = cached
    return Response(content=cached, media_type="font/ttf")


@app.post("/api/validate")
def validate(req: ValidateRequest):
    path = _asset_path(req.asset_id)
    context = {"font": req.font} if req.font else None
    manifest = load_manifest(UPLOAD_DIR, req.asset_id)
    if manifest is not None and not manifest.semantic_units:
        try:
            from tofu.layers.basil import unify_manifest
            unify_manifest(manifest)
            save_manifest(UPLOAD_DIR, req.asset_id, manifest)
        except Exception:
            pass
    # Preflight is the decision gate, so make persisted Cicerone evidence
    # available even for projects scanned before glyph retrieval was added.
    # This is local-only and never changes the user's selected font.
    if manifest is not None and any(inst.text and inst.font_match is None for inst in manifest.instances):
        try:
            from tofu.layers.font_matching import identify_manifest_fonts
            identify_manifest_fonts(str(path), manifest, get_validator().font_registry)
            save_manifest(UPLOAD_DIR, req.asset_id, manifest)
        except Exception as exc:
            # A silent failure here looks exactly like "this asset has no
            # font evidence", which is what a manifest that never got any
            # also looks like.  Say which one it is.
            print(f"[tofu] font evidence backfill failed for {req.asset_id}: "
                  f"{type(exc).__name__}: {exc}")
    report = get_validator().validate(str(path), req.targ_lang, context, manifest)
    return jsonable(report)


# --- projects (localization management: track, retrieve, load) ---

@app.post("/api/projects")
def create_project(req: ProjectCreate):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "project name must not be empty")
    if req.asset_kind not in ("image", "video"):
        raise HTTPException(400, f"unknown asset kind '{req.asset_kind}'")
    if req.asset_kind == "video":
        if not video_media_available():
            raise HTTPException(409, "video localization requires ffmpeg and ffprobe on PATH")
    return db.create_project(name, req.target_lang, req.asset_kind)


@app.get("/api/projects")
def list_projects(archived: Optional[bool] = None, query: Optional[str] = None,
                  sort: str = "updated"):
    if sort not in {"updated", "created", "name"}:
        raise HTTPException(400, "sort must be updated, created, or name")
    return {"projects": db.list_projects(archived=archived, query=query, sort=sort)}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    project = db.get_project(pid)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    # active asset needs a servable URL + manifest presence for session restore
    active = project.get("active_asset")
    if active:
        try:
            path = _asset_path(active["asset_id"])
            active["asset_url"] = f"/uploads/{path.name}"
        except HTTPException:
            active["asset_url"] = None
        active["has_manifest"] = load_manifest(UPLOAD_DIR, active["asset_id"]) is not None
    return project


@app.patch("/api/projects/{pid}")
def update_project(pid: str, req: ProjectUpdate):
    project = db.update_project(
        pid, name=req.name, target_lang=req.target_lang, source_lang=req.source_lang,
        ground_truth=(
            _normalize_ground_truth(req.ground_truth)
            if req.ground_truth is not None else None
        ),
        archived=req.archived,
    )
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return project


@app.post("/api/projects/{pid}/archive")
def archive_project(pid: str):
    project = db.set_project_archived(pid, True)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return project


@app.post("/api/projects/{pid}/restore")
def restore_project(pid: str):
    project = db.set_project_archived(pid, False)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return project


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    if not db.delete_project(pid):
        raise HTTPException(404, f"project '{pid}' not found")
    return {"ok": True}


@app.get("/api/projects/{pid}/history")
def project_history(pid: str, asset_id: Optional[str] = None):
    if db.get_project(pid) is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return {
        "snapshots": db.list_snapshots(pid, asset_id),
        "events": db.list_events(pid),
    }


@app.get("/api/projects/{pid}/memory")
def project_memory(pid: str):
    """translation-memory browser panel: every stored record for this
    project, newest first, with a thumb_url for the crop preview."""
    if db.get_project(pid) is None:
        raise HTTPException(404, f"project '{pid}' not found")
    records = db.list_tm_records(pid)
    for r in records:
        r["thumb_url"] = f"/tm_thumbs/{r['thumb_path']}" if r.get("thumb_path") else None
    return {"records": records, "count": len(records)}


@app.delete("/api/memory/{record_id}")
def delete_memory_record(record_id: int):
    if not db.delete_tm_record(record_id):
        raise HTTPException(404, f"TM record '{record_id}' not found")
    return {"ok": True}


@app.post("/api/projects/{pid}/snapshots")
def create_snapshot(pid: str, req: SnapshotCreate):
    """snapshot the asset's current manifest on demand — called before any
    destructive step (session replacement, restore) so data is never lost."""
    if db.get_project(pid) is None:
        raise HTTPException(404, f"project '{pid}' not found")
    manifest = load_manifest(UPLOAD_DIR, req.asset_id)
    if manifest is None:
        return {"snapshot_id": None, "note": "no manifest to snapshot"}
    from tofu.utils.manifest_store import _manifest_to_dict
    sid = db.add_snapshot(pid, req.asset_id, _manifest_to_dict(manifest), reason=req.reason)
    db.log_event(pid, "snapshot", f"{req.reason} snapshot of {req.asset_id}")
    return {"snapshot_id": sid}


@app.post("/api/snapshots/{sid}/restore")
def restore_snapshot(sid: int):
    """write a snapshot's manifest back as the asset's live manifest.
    The pre-restore state is itself snapshotted first (restore-backup)."""
    snap = db.get_snapshot(sid)
    if snap is None:
        raise HTTPException(404, f"snapshot {sid} not found")
    from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict
    current = load_manifest(UPLOAD_DIR, snap["asset_id"])
    if current is not None:
        db.add_snapshot(snap["project_id"], snap["asset_id"],
                        _manifest_to_dict(current), reason="restore-backup")
    manifest = _dict_to_manifest(snap["manifest"])
    save_manifest(UPLOAD_DIR, snap["asset_id"], manifest)
    db.log_event(snap["project_id"], "restore",
                 f"restored snapshot #{sid} for {snap['asset_id']}")
    db.set_active_asset(snap["project_id"], snap["asset_id"])
    return jsonable(manifest)


def _scan_named_a_language(manifest) -> bool:
    """Did the scan FIND a language, or merely fall back to the latin default?

    The scan runs unhinted on purpose -- it is an upload guard, and feeding it
    the project's own answer would make it agree with itself. But an English
    charset cannot emit Cyrillic, Greek, Arabic or Devanagari, so on such an
    asset it returns confident-looking latin garbage, which _script_lang_for
    then labels 'en' because that is the reader's own language. That 'en' is a
    DEFAULT, not a finding, and reporting it as a detection is what told a user
    their Russian billboard was English and overwrote a correct project source.

    Two things count as actually naming a language: a non-latin script was
    emitted (only a reader that covers it can do that), or the latin heuristic
    positively identified one. Note the heuristic can never answer "en" -- it
    only reports a language that BEATS the english default -- so a genuinely
    English asset abstains here rather than auto-locking. That is the intended
    trade: no verdict is better than a confident wrong one.
    """
    detector = cicerone.ScriptDetector()
    texts = [i.text or "" for i in manifest.instances]
    if any(detector.detect_script(t) not in (None, "latin") for t in texts):
        return True
    return cicerone.guess_latin_language(texts) is not None


@app.post("/api/assets/{asset_id}/scan-language")
def scan_language(asset_id: str):
    """background language scan (lexiq-parity upload guard): detect the
    asset's dominant language via cicerone (read-only — the manifest on
    disk is untouched) and compare it against the project's source
    language. when the project source is unset ('auto'), the first
    successful scan locks it."""
    path = _asset_path(asset_id)
    info = infer_asset_info(str(path))
    pid = db.project_for_asset(asset_id)
    project = db.get_project(pid) if pid else None
    project_src = project["source_lang"] if project else None

    try:
        import easyocr  # noqa: F401
    except ImportError:
        return {"engine": "null", "detected_lang": None, "match": None,
                "project_source_lang": project_src, "locked": False}

    try:
        # adaptive=False: the auto-probe suffices for language IDENTITY;
        # the full tuned re-detection is capture's job, not the scan's
        manifest = cicerone.detect(
            str(path), info, adaptive=False, font_registry=get_validator().font_registry
        )
    except Exception as exc:
        raise HTTPException(422, f"language scan could not read the asset: {exc}")
    detected = _infer_src_lang(manifest) if manifest.instances else None
    if detected and not _scan_named_a_language(manifest):
        detected = None

    locked = False
    match: Optional[bool] = None
    if detected:
        if project and not project_src:
            db.update_project(pid, source_lang=detected)
            project_src = detected
            locked = True
            match = True
        elif project_src:
            # base-language comparison: script detection cannot reliably
            # split regional variants (zh-cn vs zh-tw), so they must not
            # false-positive as mismatches
            match = detected.split("-")[0] == project_src.split("-")[0]
    if pid:
        db.log_event(pid, "language-scan",
                     f"{asset_id}: detected {detected or 'no text'}"
                     f" (project source: {project_src or 'auto'})"
                     + (" — locked" if locked else "")
                     + (" — MISMATCH" if match is False else ""))
    return {"engine": "easyocr", "detected_lang": detected, "match": match,
            "project_source_lang": project_src, "locked": locked}


@app.delete("/api/projects/{pid}/assets/{asset_id}")
def delete_project_asset(pid: str, asset_id: str):
    """unlink an asset from its project. the snapshot ledger and the
    uploaded file are retained — deletion never destroys session data."""
    if db.get_project(pid) is None:
        raise HTTPException(404, f"project '{pid}' not found")
    if not db.unlink_asset(pid, asset_id):
        raise HTTPException(404, f"asset '{asset_id}' not in project")
    db.log_event(pid, "asset-removed", f"removed asset {asset_id} from project")
    return {"ok": True}


@app.patch("/api/assets/{asset_id}/ground-truth")
def update_asset_ground_truth(asset_id: str, req: GroundTruthUpdate):
    asset = db.update_asset_ground_truth(
        asset_id, _normalize_ground_truth(req.ground_truth)
    )
    if asset is None:
        raise HTTPException(404, f"asset '{asset_id}' is not linked to a project")
    db.log_event(
        asset["project_id"], "ground-truth",
        f"updated {len(asset['ground_truth'])} asset Ground Truth term(s) for {asset_id}",
    )
    return asset


@app.delete("/api/snapshots/{sid}")
def delete_snapshot(sid: int):
    snap = db.get_snapshot(sid)
    if snap is None:
        raise HTTPException(404, f"snapshot {sid} not found")
    db.delete_snapshot(sid)
    db.log_event(snap["project_id"], "snapshot-deleted",
                 f"deleted {snap['reason']} snapshot #{sid} of {snap['asset_id']}")
    return {"ok": True}


@app.post("/api/projects/{pid}/assets/{asset_id}/activate")
def activate_asset(pid: str, asset_id: str):
    """switch the project's active session to a previously uploaded asset."""
    project = db.get_project(pid)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    if not any(a["asset_id"] == asset_id for a in project["assets"]):
        raise HTTPException(404, f"asset '{asset_id}' not in project")
    db.set_active_asset(pid, asset_id)
    path = _asset_path(asset_id)
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    db.log_event(pid, "session-loaded", f"loaded asset {asset_id}")
    return {
        "asset_id": asset_id,
        "asset_url": f"/uploads/{path.name}",
        "manifest": jsonable(manifest) if manifest else None,
    }


# --- detect + manifest CRUD ---

def _normalize_ground_truth(values: Optional[List[str]]) -> List[str]:
    """Whitespace-delimited, stable, exact-Unicode term normalization."""
    result: List[str] = []
    seen = set()
    for value in values or []:
        for term in str(value).split():
            if term and term not in seen:
                seen.add(term)
                result.append(term)
    return result


def _ground_truth_pool(asset_id: str, lang: Optional[str]) -> List[tuple]:
    """Effective project+asset pool, with asset scope winning duplicates."""
    pid = db.project_for_asset(asset_id)
    project = db.get_project(pid) if pid else None
    if not project:
        return []
    effective: Dict[str, str] = {
        term: "project" for term in _normalize_ground_truth(project.get("ground_truth"))
    }
    active = next(
        (item for item in project.get("assets", []) if item["asset_id"] == asset_id),
        None,
    )
    for term in _normalize_ground_truth(active.get("ground_truth") if active else None):
        effective[term] = "asset"
    source_lang = lang or project.get("source_lang")
    return [(term, source_lang, scope) for term, scope in effective.items()]

def _project_lang_hints(asset_id: str) -> Optional[List[str]]:
    """source-language priority protection: when the asset's project has a
    locked source language, detection MUST start language-tuned — the
    charset drives what the recognizer can read, and the language guard
    already ensures assets match the project source."""
    pid = db.project_for_asset(asset_id)
    if not pid:
        return None
    project = db.get_project(pid)
    if project and project.get("source_lang"):
        return [project["source_lang"]]
    return None


@app.post("/api/detect")
def detect(req: DetectRequest):
    path = _asset_path(req.asset_id)
    info = infer_asset_info(str(path))
    hints = req.languages or _project_lang_hints(req.asset_id)
    backend = None
    if cicerone._engine_from_env() == "paddleocr":
        backend = (
            cicerone.PaddleOCRBackend(languages=hints or ["en"], gpu=False)
            if cicerone.PaddleOCRBackend.is_available() else cicerone.NullBackend()
        )
    else:
        try:
            import easyocr  # noqa: F401
            langset = cicerone.expand_langset(hints) if hints else ("en",)
            backend = cicerone.EasyOCRBackend(
                languages=langset,
                gpu=req.gpu if req.gpu is not None else False,
            )
        except ImportError:
            backend = cicerone.NullBackend()

    # scene pre-pass: candidate surfaces constrain text detection
    from tofu.layers import scene
    try:
        scene_regions = scene.analyze_regions(str(path))
    except Exception:
        scene_regions = []

    manifest = cicerone.detect(
        str(path), info, backend=backend, scene_regions=scene_regions,
        languages=hints,
        ground_truth_pool=_ground_truth_pool(
            req.asset_id, hints[0] if hints else None
        ),
        font_registry=get_validator().font_registry,
    )
    manifest.src_lang = _infer_src_lang(manifest)
    # scene enrichment at CAPTURE time (not just render): style/background
    # profiles + typography power the capture tooltips and region table
    try:
        manifest = scene.analyze(str(path), manifest)
    except Exception:
        pass  # enrichment is best-effort; detection results stand alone
    # Visual retrieval complements typography's weight/slant profile.  It is
    # local-only during scan; commercial catalog lookup requires a later,
    # explicit editor consent action.
    try:
        from tofu.layers.font_matching import identify_manifest_fonts
        identify_manifest_fonts(str(path), manifest, get_validator().font_registry)
    except Exception:
        pass
    tm_matched = _lookup_tm_for_manifest(req.asset_id, manifest, path)
    _resolve_auto_fonts(manifest)  # after TM lookup so manifest.targ_lang is set
    save_manifest(UPLOAD_DIR, req.asset_id, manifest)
    # A Render-entry baseline is meaningful only for the exact detected
    # manifest it was captured from.  A fresh scan replaces that starting
    # point, so a later Reset must never resurrect edits from a prior scan.
    _localized_baseline_index(req.asset_id).unlink(missing_ok=True)
    _record_detection(req.asset_id, manifest)
    payload = jsonable(manifest)
    payload["engine"] = _engine_name(backend)
    payload["tm_matched"] = tm_matched
    return payload


def _lookup_tm_for_manifest(asset_id: str, manifest: TextManifest, source_path) -> int:
    """wraps _attach_tm_suggestions with the project's target language
    (detect-time manifests don't carry a targ_lang yet -- that's chosen
    at render time -- so the project default stands in as the language
    a capture-time 'seen before' suggestion is most useful for)."""
    pid = db.project_for_asset(asset_id)
    if not pid:
        return 0
    project = db.get_project(pid)
    targ_lang = project["target_lang"] if project else None
    if not targ_lang:
        return 0
    manifest.targ_lang = manifest.targ_lang or targ_lang
    return _attach_tm_suggestions(pid, manifest, source_path)


def _engine_name(backend) -> str:
    """which OCR engine actually ran — the UI warns when detection was a
    silent no-op because no real engine is installed on the server."""
    if isinstance(backend, cicerone.EasyOCRBackend):
        return "easyocr"
    if isinstance(backend, cicerone.PaddleOCRBackend):
        return "paddleocr"
    return "null"


def _record_detection(asset_id: str, manifest: TextManifest) -> None:
    """project bookkeeping after a detection pass: history event, autosave
    snapshot, and auto-set the project's source language (Tofu detects it —
    users only pick the target)."""
    pid = db.project_for_asset(asset_id)
    if not pid:
        return
    db.log_event(pid, "detection",
                 f"detected {len(manifest.instances)} region(s), "
                 f"source language: {manifest.src_lang or 'unknown'}")
    from tofu.utils.manifest_store import _manifest_to_dict
    db.add_snapshot(pid, asset_id, _manifest_to_dict(manifest), reason="autosave")
    if manifest.src_lang:
        db.update_project(pid, source_lang=manifest.src_lang)


@app.get("/api/detect/stream")
def detect_stream(
    asset_id: str,
    languages: Optional[str] = None,
    gpu: bool = False,
    engine: Optional[str] = None,
):
    """SSE variant of /api/detect: emits progress events per stage/pass.

    GET (not POST) so the browser's native EventSource can consume it;
    languages is a comma-separated list of tofu codes. The engine
    parameter accepts 'easyocr' (default) or 'paddleocr'. shares
    merge_detections + build_manifest with /api/detect, so both produce
    identical manifests from identical inputs.
    """
    import json as _json
    import os
    from fastapi.responses import StreamingResponse
    from tofu.layers import scene

    # normalize engine selection for backend factory
    if engine:
        os.environ["OCR_ENGINE"] = engine.lower()

    path = _asset_path(asset_id)  # 404s before the stream opens
    info = infer_asset_info(str(path))
    lang_hints = (
        [l.strip() for l in languages.split(",") if l.strip()]
        if languages else None
    )
    # source-language priority: a locked project source language tunes the
    # reader charset from pass 1 (and skips the adaptive rediscovery stage)
    if lang_hints is None:
        lang_hints = _project_lang_hints(asset_id)

    def event(data: Dict[str, Any]) -> str:
        return f"data: {_json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        try:
            yield event({"stage": "scene", "status": "running"})
            try:
                regions = scene.analyze_regions(str(path))
            except Exception:
                regions = []
            yield event({
                "stage": "scene", "status": "complete",
                "regions": [jsonable(r) for r in regions],
            })

            # choose backend via env or default factory
            if cicerone._engine_from_env() == "paddleocr":
                if cicerone.PaddleOCRBackend.is_available():
                    backend = (
                        cicerone.PaddleOCRBackend(languages=lang_hints, gpu=gpu)
                        if lang_hints else cicerone.get_backend()
                    )
                else:
                    backend = cicerone.NullBackend()
            else:
                try:
                    import easyocr  # noqa: F401
                    langset = (
                        cicerone.expand_langset(lang_hints) if lang_hints else ("en",)
                    )
                    backend = cicerone.EasyOCRBackend(languages=langset, gpu=gpu)
                except ImportError:
                    backend = cicerone.NullBackend()

            # Run the ONE pipeline, forwarding its stage callbacks as SSE.
            #
            # This endpoint used to re-implement cicerone.detect() inline so it
            # could interleave progress events, and the two copies drifted
            # seven ways -- the worst being that assess_multi_candidate_ocr
            # never ran here at all, so every region detected through the app
            # carried no arbitration record and no hypothesis score, while the
            # eval harness (which calls detect()) measured a pipeline the app
            # was not running. Detection now happens in a worker thread and
            # its callback payloads are drained onto this generator.
            events: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
            outcome: Dict[str, Any] = {}

            def run_detection():
                try:
                    outcome["manifest"] = cicerone.detect(
                        str(path), asset_info=info, backend=backend,
                        languages=list(lang_hints) if lang_hints else None,
                        scene_regions=regions,
                        ground_truth_pool=_ground_truth_pool(
                            asset_id, lang_hints[0] if lang_hints else None
                        ),
                        font_registry=get_validator().font_registry,
                        on_stage=events.put,
                    )
                except BaseException as exc:  # surfaced on the main thread
                    outcome["error"] = exc
                finally:
                    events.put(None)

            worker = threading.Thread(target=run_detection, daemon=True)
            worker.start()
            while True:
                payload = events.get()
                if payload is None:
                    break
                yield event(payload)
            worker.join()
            if "error" in outcome:
                raise outcome["error"]
            manifest = outcome["manifest"]

            # scene enrichment at capture time: profiles + typography
            if manifest.instances:
                yield event({"stage": "enrich", "status": "running"})
                try:
                    manifest = scene.analyze(str(path), manifest)
                except Exception:
                    pass
                yield event({
                    "stage": "enrich", "status": "complete",
                    "regions": len(manifest.instances),
                })

            if manifest.instances:
                yield event({"stage": "font_match", "status": "running"})
                try:
                    from tofu.layers.font_matching import identify_manifest_fonts
                    matched_fonts = identify_manifest_fonts(
                        str(path), manifest, get_validator().font_registry
                    )
                except Exception:
                    matched_fonts = 0
                yield event({"stage": "font_match", "status": "complete", "matched": matched_fonts})

            # Re-evaluate Latin language from the final recognized text.
            # Savor/Wasabi/menu may have repaired the weak first-pass text;
            # cicerone.detect() performs this same final pass internally.
            cicerone.label_latin_languages(
                manifest.instances, lang_hints[0] if lang_hints else None
            )
            manifest.src_lang = _infer_src_lang(manifest)

            tm_matched = 0
            if manifest.instances:
                yield event({"stage": "memory", "status": "running"})
                tm_matched = _lookup_tm_for_manifest(asset_id, manifest, path)
                yield event({"stage": "memory", "status": "complete", "matched": tm_matched})

            # what does "auto" render with, right now, for each region?
            # near-instant (registry lookups, no font-file I/O) -- no
            # SSE stage of its own, folded in silently before save
            _resolve_auto_fonts(manifest)

            save_manifest(UPLOAD_DIR, asset_id, manifest)
            _localized_baseline_index(asset_id).unlink(missing_ok=True)
            _record_detection(asset_id, manifest)
            yield event({
                "stage": "complete",
                "manifest": jsonable(manifest),
                "engine": _engine_name(backend),
                "tm_matched": tm_matched,
            })
        except Exception as exc:
            yield event({"stage": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/manifest/{asset_id}")
def get_manifest(asset_id: str):
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    # Existing projects predate material labels.  Enrich those manifests on
    # read from the source pixels only (no OCR/re-detection), so the Capture
    # surface inspector can say "brick / masonry" or "painted sign / panel"
    # instead of exposing implementation labels such as text_cluster.
    missing_material = (
        any(region.material is None for region in manifest.scene_regions)
        or any((inst.background_profile is not None and inst.background_profile.material is None)
               for inst in manifest.instances)
    )
    if missing_material:
        try:
            from tofu.layers import scene
            manifest = scene.analyze(str(_asset_path(asset_id)), manifest)
            save_manifest(UPLOAD_DIR, asset_id, manifest)
        except Exception:
            # Material is presentation enrichment.  A non-image or optional
            # vision dependency must never block opening a valid project.
            pass
    # Older manifests predate semantic reading-unit registration.  Upgrade
    # that metadata on read without re-running OCR and without touching
    # target_text: users keep every existing translation exactly as entered.
    #
    # Units registered before the pairing verdict existed are also stale:
    # they were grouped by box adjacency alone, so a manifest can be holding
    # a unit that spans several unrelated signs.  Re-registering is cheap and
    # is the same deterministic pass, so upgrade those too rather than
    # leaving a project permanently on the old grouping.
    if not manifest.semantic_units or any(unit.pairing is None for unit in manifest.semantic_units):
        try:
            from tofu.layers.basil import unify_manifest
            unify_manifest(manifest)
            save_manifest(UPLOAD_DIR, asset_id, manifest)
        except Exception:
            pass
    # Upgrade approved Basil plans created before block→cube anchoring.  This
    # is a deterministic metadata/projection repair: no OCR or geometry is
    # changed, but every client and Scribe now see the same plated order.
    try:
        from tofu.layers.basil import migrate_legacy_plating
        if migrate_legacy_plating(manifest):
            save_manifest(UPLOAD_DIR, asset_id, manifest)
    except Exception:
        pass
    return jsonable(manifest)


@app.get("/api/semantic/providers")
def semantic_providers(project_id: Optional[str] = None):
    """Report only locally provisioned semantic/translation seams.

    It is intentionally not an installer: language models are large and must
    pass ToFU's benchmark gate before a deployment enables them.
    """
    from tofu.layers.basil import provider_statuses, set_active_glossary
    effective, metadata = glossary_utils.resolve_active_glossary(GLOSSARY_DIR, project_id)
    set_active_glossary(metadata if effective else None)
    return {"providers": provider_statuses()}


@app.post("/api/semantic-units/{asset_id}/{unit_id}/substitution")
def semantic_substitution(asset_id: str, unit_id: str, req: SemanticSubstitutionRequest):
    """Plan or explicitly apply target spans to immutable source anchors."""
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    from tofu.layers import basil
    target_lang = req.targ_lang or manifest.targ_lang
    project_id = db.project_for_asset(asset_id)
    effective_lexicon, glossary_meta = glossary_utils.resolve_active_glossary(GLOSSARY_DIR, project_id)
    basil.set_active_glossary(glossary_meta if effective_lexicon else None)
    try:
        plan = basil.plan_substitution(
            manifest, unit_id, req.target_text, target_lang,
            external_lexicon=effective_lexicon,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    if req.apply:
        try:
            basil.apply_substitution(manifest, plan, target_lang)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        manifest.targ_lang = target_lang or manifest.targ_lang
        _resolve_auto_fonts(manifest, target_lang)
    # Persist unit registration and, on apply, the provenance. A plan never
    # changes targets or geometry, so merely opening Translation substitution
    # cannot alter what Render will produce.
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    return {"manifest": jsonable(manifest), "plan": plan, "applied": bool(req.apply)}


@app.post("/api/semantic-units/{asset_id}/{unit_id}/repair")
def semantic_repair(asset_id: str, unit_id: str, req: SemanticRepairRequest):
    """Accept or reject Basil's proposed cross-region source correction.

    The decision changes only the semantic unit's own source text. Region
    ids, boxes and OCR text stay exactly as Cicerone left them, so a
    rejected proposal costs nothing and a later re-detection loses nothing.
    """
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    from tofu.layers import basil
    try:
        basil.accept_repair(manifest, unit_id, req.accepted)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    return {"manifest": jsonable(manifest), "accepted": bool(req.accepted)}


def _glossary_path(scope: str, project_id: Optional[str] = None) -> Path:
    if scope not in {"global", "project"}:
        raise HTTPException(422, "scope must be 'global' or 'project'")
    if scope == "project":
        if not project_id or not re.fullmatch(r"[A-Za-z0-9_-]+", project_id):
            raise HTTPException(422, "project scope requires a valid project_id")
        return GLOSSARY_DIR / f"project_{project_id}.json"
    return GLOSSARY_DIR / "global.json"


def _glossary_meta(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {key: value for key, value in data.items() if key != "lexicon"}
    except Exception:
        return None


@app.post("/api/glossary/upload")
async def upload_glossary(
    file: UploadFile = File(...),
    scope: str = "project",
    project_id: Optional[str] = None,
    mode: str = "auxiliary",
    src_lang: Optional[str] = None,
    targ_lang: Optional[str] = None,
):
    if mode not in {"auxiliary", "merge", "replace"}:
        raise HTTPException(422, "mode must be auxiliary, merge, or replace")
    path = _glossary_path(scope, project_id)
    filename = file.filename or "glossary.txt"
    raw = await file.read()
    try:
        parsed = glossary_utils.parse_glossary(filename, raw, src_lang, targ_lang)
    except glossary_utils.GlossaryParseError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not parsed.entries:
        raise HTTPException(422, "glossary contains no usable source/target entries with language pairs")
    lexicon = glossary_utils.to_basil_lexicon(parsed)
    payload = {
        "mode": mode,
        "source_format": parsed.source_format,
        "uploaded_at": datetime.now().astimezone().isoformat(),
        "filename": filename,
        "entry_count": parsed.entry_count,
        "language_pairs": [list(pair) for pair in parsed.language_pairs],
        "lexicon": glossary_utils.serializable_lexicon(lexicon),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return {key: payload[key] for key in ("entry_count", "language_pairs", "mode", "source_format", "filename")}


@app.get("/api/glossary/status")
def glossary_status(project_id: Optional[str] = None):
    global_meta = _glossary_meta(_glossary_path("global"))
    project_meta = _glossary_meta(_glossary_path("project", project_id)) if project_id else None
    effective, metadata = glossary_utils.resolve_active_glossary(GLOSSARY_DIR, project_id)
    return {
        "global": global_meta,
        "project": project_meta,
        "effective_mode": metadata.get("mode") if effective else None,
        "effective_entry_count": metadata.get("entry_count", 0) if effective else 0,
        "effective_language_pairs": metadata.get("language_pairs", []) if effective else [],
    }


@app.delete("/api/glossary/{scope}")
def delete_glossary(scope: str, project_id: Optional[str] = None):
    path = _glossary_path(scope, project_id)
    if path.exists():
        path.unlink()
    return {"ok": True}


@app.post("/api/font-match/{asset_id}")
def font_match(asset_id: str, req: FontMatchRequest):
    """Refresh local font evidence and, only with explicit consent, query a
    configured commercial font catalog for unavailable/licensed candidates."""
    path = _asset_path(asset_id)
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    from tofu.layers.font_matching import identify_manifest_fonts, external_catalog_match
    local = identify_manifest_fonts(str(path), manifest, get_validator().font_registry)
    external = external_catalog_match(str(path), manifest, get_validator().font_registry) if req.allow_external else 0
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    return {"manifest": jsonable(manifest), "local_matched": local, "external_matched": external}


@app.put("/api/manifest/{asset_id}")
def put_manifest(asset_id: str, manifest_data: Dict[str, Any]):
    from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict
    manifest = _dict_to_manifest(manifest_data)
    manifest.asset_id = asset_id
    manifest.total_regions = len(manifest.instances)
    # re-resolve "auto" fonts on every save: target_text/target_language
    # are exactly what changes during Translate-step editing, and
    # resolution is cheap (registry lookups, no font-file I/O) -- no
    # need to diff old vs new to decide what changed
    _resolve_auto_fonts(manifest)
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    # autosave ledger: every accepted write is snapshotted (deduped) so a
    # session can always be rolled back
    pid = db.project_for_asset(asset_id)
    if pid:
        db.add_snapshot(pid, asset_id, _manifest_to_dict(manifest), reason="autosave")
    resolved_fonts = {
        inst.id: inst.resolved_font_family
        for inst in manifest.instances if inst.resolved_font_family
    }
    return {"ok": True, "total_regions": manifest.total_regions, "resolved_fonts": resolved_fonts}


def _translation_manifest_revision(manifest: TextManifest) -> str:
    from tofu.utils.manifest_store import _manifest_to_dict
    payload = json.dumps(
        _manifest_to_dict(manifest), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _translation_glossary_revision(project_id: Optional[str]) -> Optional[str]:
    paths = [_glossary_path("global")]
    if project_id:
        paths.append(_glossary_path("project", project_id))
    digest = hashlib.sha256()
    found = False
    for path in paths:
        if path.is_file():
            found = True
            digest.update(path.read_bytes())
    return digest.hexdigest() if found else None


@app.post("/api/assets/{asset_id}/translation-runs")
def create_translation_run(asset_id: str, req: TranslationRunRequest):
    """Create local TM proposals. No network provider is invoked here."""
    from tofu.utils.translation_workflow import append_attempt, make_tm_attempt
    from tofu.utils.manifest_store import _manifest_to_dict

    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    pid = db.project_for_asset(asset_id)
    project = db.get_project(pid) if pid else None
    target_lang = req.target_lang or manifest.targ_lang or (
        project.get("target_lang") if project else None
    )
    if not target_lang:
        raise HTTPException(422, "target language is required")
    manifest.targ_lang = target_lang
    _lookup_tm_for_manifest(asset_id, manifest, _asset_path(asset_id))
    selected = set(req.region_ids or [inst.id for inst in manifest.instances])
    unknown = selected - {inst.id for inst in manifest.instances}
    if unknown:
        raise HTTPException(404, f"unknown region(s): {', '.join(sorted(unknown))}")
    base_revision = _translation_manifest_revision(manifest)
    glossary_revision = _translation_glossary_revision(pid)
    attempts = []
    for inst in manifest.instances:
        if inst.id not in selected:
            continue
        attempt = make_tm_attempt(
            inst, inst.target_language or target_lang,
            manifest_revision=base_revision,
            glossary_revision=glossary_revision,
        )
        if attempt is not None:
            append_attempt(inst, attempt)
            attempts.append({"region_id": inst.id, **attempt})
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    revision = _translation_manifest_revision(manifest)
    if pid:
        db.add_snapshot(pid, asset_id, _manifest_to_dict(manifest), reason="autosave")
        db.log_event(pid, "translation-run", f"created {len(attempts)} local TM proposal(s)")
    return {
        "run_id": uuid.uuid4().hex,
        "asset_id": asset_id,
        "provider_mode": "local_tm_only",
        "manifest_revision": revision,
        "attempts": attempts,
        "unresolved_region_ids": sorted(selected - {item["region_id"] for item in attempts}),
    }


@app.post("/api/assets/{asset_id}/translation-decisions")
def apply_translation_decisions(asset_id: str, req: TranslationDecisionRequest):
    """Apply server-owned attempts with optimistic concurrency protection."""
    from tofu.utils.translation_workflow import apply_decision, find_attempt
    from tofu.utils.manifest_store import _manifest_to_dict

    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    current_revision = _translation_manifest_revision(manifest)
    if req.manifest_revision != current_revision:
        raise HTTPException(409, "manifest changed after translation proposals were loaded")
    by_id = {inst.id: inst for inst in manifest.instances}
    applied = []
    for item in req.decisions:
        inst = by_id.get(item.region_id)
        if inst is None:
            raise HTTPException(404, f"unknown region '{item.region_id}'")
        attempt = find_attempt(inst, item.attempt_id) if item.attempt_id else None
        if item.attempt_id and attempt is None:
            raise HTTPException(404, f"unknown translation attempt '{item.attempt_id}'")
        try:
            decision = apply_decision(
                inst, action=item.action, attempt=attempt, text=item.text,
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        applied.append({"region_id": inst.id, **decision})
    _resolve_auto_fonts(manifest)
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    pid = db.project_for_asset(asset_id)
    if pid:
        db.add_snapshot(pid, asset_id, _manifest_to_dict(manifest), reason="autosave")
        db.log_event(pid, "translation-decision", f"applied {len(applied)} translation decision(s)")
    return {
        "asset_id": asset_id,
        "manifest_revision": _translation_manifest_revision(manifest),
        "decisions": applied,
        "manifest": jsonable(manifest),
    }


@app.post("/api/manifest/{asset_id}/regions")
def add_region(asset_id: str, req: RegionCreate):
    bbox = _normalized_asset_bbox(asset_id, req.dict())
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        manifest = TextManifest(
            asset_id=asset_id,
            total_regions=0,
            instances=[],
            img_dim=None,
            scene_regions=[],
        )
    existing_nums = [int(i.id[1:]) for i in manifest.instances if i.id.startswith("r")]
    next_num = max(existing_nums, default=0) + 1
    new_inst = InstText(
        id=f"r{next_num}",
        bounding_box=bbox,
        text=req.text, target_text=req.target_text,
        reading_order=len(manifest.instances),
    )
    manifest.instances.append(new_inst)
    manifest.total_regions = len(manifest.instances)
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    return jsonable(new_inst)


@app.post("/api/manifest/{asset_id}/regions/merge")
def merge_regions(asset_id: str, req: RegionMerge):
    """Fold several regions into one, re-reading the union where it helps.

    merge_baseline_runs already joins the words of a line automatically,
    and declines wherever the geometry is ambiguous -- across a column
    gutter, over a gap wider than a word space. This is the manual door
    for those, and for any grouping only a person knows is one unit.

    The survivor is the FIRST member in reading order, which keeps its id
    and its correction/provenance history; the rest are marked excluded
    exactly as delete_region marks them, so cleanse() still erases their
    pixels even though the merged box already covers them.
    """
    if len(req.region_ids) < 2:
        raise HTTPException(422, "merging needs at least two regions")
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    by_id = {i.id: i for i in manifest.instances}
    members = []
    for rid in req.region_ids:
        inst = by_id.get(rid)
        if inst is None:
            raise HTTPException(404, f"region '{rid}' not found")
        if inst.bounding_box is None:
            raise HTTPException(422, f"region '{rid}' has no bounding box")
        members.append(inst)

    boxes = [i.bounding_box for i in members]
    order = cicerone._fragment_reading_order(boxes)
    members = [members[i] for i in order]
    boxes = [i.bounding_box for i in members]
    union = BBox(
        x=min(b.x for b in boxes), y=min(b.y for b in boxes),
        width=max(b.x + b.width for b in boxes) - min(b.x for b in boxes),
        height=max(b.y + b.height for b in boxes) - min(b.y for b in boxes),
    )
    lang = members[0].detected_language or members[0].language or manifest.src_lang
    try:
        engine = cicerone.EasyOCRBackend(cicerone.expand_langset([lang] if lang else ["en"]))
    except Exception:
        engine = None
    text, confidence, source = cicerone.reread_merged_region(
        str(_asset_path(asset_id)), union,
        [i.text or "" for i in members], boxes, engine=engine,
    )

    survivor = members[0]
    survivor.bounding_box = union
    survivor.text = text
    if confidence is not None:
        survivor.confidence = confidence
    for spare in members[1:]:
        spare.excluded = True
    manifest.total_regions = sum(1 for i in manifest.instances if not i.excluded)
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    pid = db.project_for_asset(asset_id)
    if pid:
        db.log_event(pid, "region-merge",
                     f"merged {len(members)} regions into {survivor.id} ({source})")
    return {
        "ok": True, "region": jsonable(survivor), "source": source,
        "merged_ids": [i.id for i in members[1:]],
        "total_regions": manifest.total_regions,
    }


@app.delete("/api/manifest/{asset_id}/regions/{rid}")
def delete_region(asset_id: str, rid: str):
    """"remove" a region from the workspace. the instance is marked
    excluded rather than actually dropped from the manifest: cleanse()
    still erases its source pixels (nothing is left half-translated on
    the canvas), but scribe() never renders it and the UI never lists
    it -- matches every other "delete" in this app while still letting
    a user keep a region out of the export without leaving its source
    text sitting untouched in the final image."""
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    inst = next((i for i in manifest.instances if i.id == rid), None)
    if inst is None:
        raise HTTPException(404, f"region '{rid}' not found")
    inst.excluded = True
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    active_count = sum(1 for i in manifest.instances if not i.excluded)
    return {"ok": True, "total_regions": active_count}


@app.patch("/api/manifest/{asset_id}/regions/{rid}")
def update_region(asset_id: str, rid: str, req: RegionUpdate):
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    inst = next((i for i in manifest.instances if i.id == rid), None)
    if inst is None:
        raise HTTPException(404, f"region '{rid}' not found")
    geometry_changed = any(
        value is not None for value in (req.x, req.y, req.width, req.height)
    )
    if geometry_changed:
        current = inst.bounding_box
        inst.bounding_box = _normalized_asset_bbox(asset_id, {
            "x": current.x if req.x is None else req.x,
            "y": current.y if req.y is None else req.y,
            "width": current.width if req.width is None else req.width,
            "height": current.height if req.height is None else req.height,
        })
    if req.text is not None: inst.text = req.text
    if req.target_text is not None: inst.target_text = req.target_text
    if req.dnt is not None: inst.dnt = req.dnt
    if req.excluded is not None: inst.excluded = req.excluded
    if req.target_language is not None: inst.target_language = req.target_language
    if req.language is not None: inst.language = req.language
    if req.font is not None:
        from tofu.core.types import StyleProfil
        if inst.style_profile is None:
            inst.style_profile = StyleProfil()
        inst.style_profile.font_family = req.font
    supplied = getattr(req, "model_fields_set", getattr(req, "__fields_set__", set()))
    if "target_orientation" in supplied or "word_order" in supplied:
        if inst.style_profile is None:
            inst.style_profile = StyleProfil()
        if "target_orientation" in supplied:
            inst.style_profile.target_orientation = req.target_orientation
        if "word_order" in supplied:
            inst.style_profile.word_order = req.word_order
    if "segmentation_mask" in supplied and req.segmentation_mask is None:
        inst.segmentation_mask = None
    elif req.segmentation_mask is not None:
        from tofu.core.types import Mask
        polygon = req.segmentation_mask.get("polygon")
        if not polygon or len(polygon) < 3:
            raise HTTPException(422, "segmentation_mask.polygon requires at least three points")
        holes = req.segmentation_mask.get("holes")
        inst.segmentation_mask = Mask(
            polygon=[tuple(point) for point in polygon],
            confidence=float(req.segmentation_mask.get("confidence", 1.0)),
            holes=[[tuple(point) for point in hole] for hole in holes] if holes else None,
        )
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    return jsonable(inst)


# --- OCR on a specific region ---

@app.post("/api/ocr-region")
def ocr_region(req: OcrRegionRequest):
    path = _asset_path(req.asset_id)
    bbox = _normalized_asset_bbox(req.asset_id, req.bbox)
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise HTTPException(500, "Pillow/numpy not available")
    try:
        img = Image.open(str(path)).convert("RGB")
    except Exception as exc:
        raise HTTPException(500, f"cannot open asset image: {exc}")
    crop = img.crop((bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height))
    try:
        import easyocr  # noqa: F401
    except ImportError:
        # engine absence is a server-config problem, not "no text here" —
        # the UI must not tell users to type text in manually
        return {"text": "", "confidence": 0.0, "detected_language": None,
                "engine_missing": True}
    crop_np = np.asarray(crop)
    # determine OCR language from manifest if available
    manifest = load_manifest(UPLOAD_DIR, req.asset_id)
    src_lang = manifest.src_lang if manifest else "en"
    ocr_langs = [src_lang] if src_lang else ["en"]
    try:
        from tofu.layers.cicerone import EasyOCRBackend
        backend = EasyOCRBackend(languages=ocr_langs, gpu=False)
        detections = backend.detect(crop_np)
        if detections:
            det = detections[0]
            return {"text": det.text, "confidence": det.confidence, "detected_language": det.language}
    except ImportError:
        pass
    except Exception as exc:
        # fall back to raw easyocr if backend wrapper fails
        try:
            import easyocr
            reader = easyocr.Reader([_to_easyocr_lang(ocr_langs[0])], gpu=False)
            results = reader.readtext(crop_np)
            if results:
                _, text, conf = results[0]
                return {"text": text, "confidence": float(conf), "detected_language": ocr_langs[0]}
        except Exception:
            pass
    return {"text": "", "confidence": 0.0, "detected_language": None}


# --- refine/snap a manual bbox into detected text boxes ---

@app.post("/api/detect/refine")
def refine_region(req: RefineRegionRequest):
    """detect and recognize text inside a user-drawn bbox; return the
    detected sub-boxes in original-image coordinates so the UI can snap
    a rough rectangle to precise text polygons."""
    path = _asset_path(req.asset_id)
    region = _normalized_asset_bbox(req.asset_id, req.bbox)

    import os
    if req.engine:
        os.environ["OCR_ENGINE"] = req.engine.lower()

    from tofu.layers import cicerone

    manifest = load_manifest(UPLOAD_DIR, req.asset_id)
    src_lang = manifest.src_lang if manifest else None

    if cicerone._engine_from_env() == "paddleocr":
        if not cicerone.PaddleOCRBackend.is_available():
            raise HTTPException(500, "PaddleOCR is not available (isolated venv missing)")
        langset = [src_lang] if src_lang else ["en"]
        backend = cicerone.PaddleOCRBackend(languages=langset, gpu=False)
    else:
        try:
            import easyocr  # noqa: F401
            langset = cicerone.expand_langset([src_lang]) if src_lang else ("en",)
            backend = cicerone.EasyOCRBackend(languages=langset, gpu=False)
        except ImportError:
            raise HTTPException(500, "EasyOCR is not installed")

    try:
        per_region = backend.detect_in_regions(str(path), [region], pad=0)
    except Exception as exc:
        raise HTTPException(500, f"detection failed: {exc}")

    dets = per_region[0] if per_region else []
    dets.sort(key=lambda d: d.confidence, reverse=True)

    return {
        "regions": [
            {
                "polygon": d.polygon,
                "bbox": {
                    "x": min(p[0] for p in d.polygon),
                    "y": min(p[1] for p in d.polygon),
                    "width": max(p[0] for p in d.polygon) - min(p[0] for p in d.polygon),
                    "height": max(p[1] for p in d.polygon) - min(p[1] for p in d.polygon),
                },
                "text": d.text,
                "confidence": d.confidence,
            }
            for d in dets
        ]
    }


# --- export + import ---

@app.post("/api/export")
def export(req: ExportRequest):
    manifest = load_manifest(UPLOAD_DIR, req.asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{req.asset_id}'")
    fmt = req.format.lower()
    src_lang = manifest.src_lang or "en"
    if fmt == "xliff":
        content = interchange.export_xliff(manifest, src_lang, req.targ_lang or "", req.variant or "standard")
        ext, media = "xliff", "application/xml"
    elif fmt == "tmx":
        pairs = [
            (i.id, i.text or "", i.target_text or "",
             f"bbox:{i.bounding_box.x},{i.bounding_box.y}",
             i.language or i.detected_language or src_lang,
             i.target_language or req.targ_lang or "")
            for i in manifest.instances if not getattr(i, "dnt", False)
        ]
        content = interchange.export_tmx(pairs, src_lang, req.targ_lang or "")
        ext, media = "tmx", "application/xml"
    elif fmt == "tsv":
        content = interchange.export_tsv(manifest); ext, media = "tsv", "text/tab-separated-values"
    elif fmt == "csv":
        content = interchange.export_csv(manifest); ext, media = "csv", "text/csv"
    elif fmt == "txt":
        content = interchange.export_txt(manifest); ext, media = "txt", "text/plain"
    elif fmt == "vtm":
        # Visual Translation Memory (spec/vtm-1.0.md). Unlike the formats
        # above, this carries geometry and typography as well as the string
        # pair -- the record needed to re-render a translation the way the
        # source looked, rather than merely to look it up.
        import json as _json

        from tofu.utils import vtm as vtm_mod
        try:
            document = vtm_mod.export_vtm(manifest, src_lang, req.targ_lang or "")
        except ValueError as exc:
            # the asset's pixel dimensions are required and unrecoverable
            # here: a bbox with no resolution cannot be interpreted later
            raise HTTPException(422, str(exc))
        content = _json.dumps(document, ensure_ascii=False, indent=2)
        ext, media = "vtm.json", "application/json"
    else:
        raise HTTPException(400, f"unknown format '{fmt}'")
    filename = f"{req.asset_id}.{ext}"
    return Response(content=content, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/api/import")
async def import_file(asset_id: str = "", file: UploadFile = File(...)):
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    raw = await file.read()
    try:
        content = interchange.decode_translation_bytes(raw)
    except UnicodeDecodeError:
        raise HTTPException(422, "translation file is not valid UTF-8, UTF-16, or UTF-32 text")
    filename = file.filename or "import.txt"
    fmt = interchange.detect_format(filename)
    mapping = None
    try:
        if fmt == "xliff":
            mapping = interchange.import_xliff_for_manifest(content, manifest)
            translations = mapping["translations"]
        else:
            translations = interchange.import_file(filename, content)
    except Exception as exc:
        raise HTTPException(422, f"could not parse {fmt.upper()} translation file: {type(exc).__name__}: {exc}")
    manifest_ids = {i.id for i in manifest.instances if not i.dnt}
    imported_ids = set(translations.keys())
    missing = list(manifest_ids - imported_ids)
    extra = list(imported_ids - manifest_ids)
    if mapping:
        extra.extend(str(item.get("id")) for item in mapping["unresolved"] if item.get("id"))
    for inst in manifest.instances:
        if inst.id in translations:
            inst.target_text = translations[inst.id]
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    pid = db.project_for_asset(asset_id)
    if pid:
        from tofu.utils.manifest_store import _manifest_to_dict
        db.add_snapshot(pid, asset_id, _manifest_to_dict(manifest), reason="import")
        db.log_event(pid, "import",
                     f"imported {len(translations)} translation(s) from '{filename}'")
    return {
        "imported": len(imported_ids & manifest_ids), "missing": missing,
        "extra": extra, "translations": translations, "format": fmt,
        "matched_by": mapping["matched_by"] if mapping else {"id": len(translations)},
        "unresolved": mapping["unresolved"] if mapping else [],
        "empty_targets": mapping["empty_targets"] if mapping else 0,
    }


# --- render (scene → cleanse → scribe → verify) ---

def _patch_index(asset_id: str) -> Path:
    return OUTPUT_DIR / f"{asset_id}.patches.json"


def _patch_archive_index(asset_id: str) -> Path:
    """Durable patch metadata for localized-canvas redo/reset.

    Active treatment is intentionally a small ordered layer.  Removing a
    patch from it must not destroy its PNG or metadata because redo and a
    render-entry reset need to restore that exact non-deterministic repair.
    """
    return OUTPUT_DIR / f"{asset_id}.patch-archive.json"


def _localized_baseline_index(asset_id: str) -> Path:
    return OUTPUT_DIR / f"{asset_id}.localized-baseline.json"


def _candidate_dir() -> Path:
    path = OUTPUT_DIR / "cleanse-cache" / "candidates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_index(asset_id: str) -> Path:
    return _candidate_dir() / f"{asset_id}.json"


def _load_candidates(asset_id: str) -> List[Dict[str, Any]]:
    try:
        value = json.loads(_candidate_index(asset_id).read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except Exception:
        return []


def _save_candidate(asset_id: str, record: Dict[str, Any]) -> None:
    records = [item for item in _load_candidates(asset_id) if item.get("id") != record["id"]]
    records.append(record)
    _candidate_index(asset_id).write_text(json.dumps(records, sort_keys=True), encoding="utf-8")


def _candidate_observer(asset_id: str, cache_key: str):
    """Return a server-owned sink for review-only neural repair overlays."""
    def save(info: Dict[str, Any], candidate, mask):
        from PIL import Image
        import numpy as np

        region = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(region)
        if not len(xs):
            return None
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        rgba = np.dstack([
            np.asarray(candidate, dtype=np.uint8)[y0:y1, x0:x1],
            (region[y0:y1, x0:x1].astype(np.uint8) * 255),
        ])
        # Candidate identity incorporates the actual pixels.  A stochastic
        # provider can therefore offer distinct variants without mutating the
        # selected Cleanse base or overwriting a prior review artifact.
        candidate_id = hashlib.sha256(
            f"{asset_id}:{cache_key}:{info['group_key']}:{info['provider']}".encode("utf-8")
            + rgba.tobytes()
        ).hexdigest()[:24]
        filename = f"{asset_id}-{candidate_id}.png"
        path = _candidate_dir() / filename
        if not path.exists():
            Image.fromarray(rgba, "RGBA").save(path)
        relative = f"cleanse-cache/candidates/{filename}"
        record = {
            "id": candidate_id,
            "cache_key": cache_key,
            "group_key": info["group_key"],
            "group_ids": info["group_ids"],
            "provider": info["provider"],
            "decision": info["decision"],
            "execution": info["execution"],
            "quality_gate": info["quality_gate"],
            "bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "file": relative,
            "url": f"/outputs/{relative}",
            "created_at": int(time.time() * 1000),
        }
        _save_candidate(asset_id, record)
        # This returned object is JSON-safe and gets preserved in per-region
        # provenance/cache sidecars for the localized-canvas review strip.
        return {key: record[key] for key in ("id", "provider", "decision", "bbox", "url", "cache_key")}
    return save


def _find_candidate(asset_id: str, candidate_id: str) -> Optional[Dict[str, Any]]:
    return next((item for item in _load_candidates(asset_id) if item.get("id") == candidate_id), None)


def _load_patches(asset_id: str) -> List[Dict[str, Any]]:
    try:
        return json.loads(_patch_index(asset_id).read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_patch_archive(asset_id: str) -> List[Dict[str, Any]]:
    try:
        value = json.loads(_patch_archive_index(asset_id).read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except Exception:
        # Existing projects predate the archive; their active patches are a
        # valid seed and become archived on the next treatment mutation.
        return _load_patches(asset_id)


def _archive_patches(asset_id: str, patches: List[Dict[str, Any]]) -> None:
    known = {patch.get("id"): patch for patch in _load_patch_archive(asset_id) if patch.get("id")}
    for patch in patches:
        if patch.get("id"):
            known[patch["id"]] = patch
    _patch_archive_index(asset_id).write_text(
        json.dumps(list(known.values()), sort_keys=True), encoding="utf-8"
    )


def _flatten_patches(asset_id: str) -> Optional[Path]:
    patches = _load_patches(asset_id)
    target = OUTPUT_DIR / f"{asset_id}.patches.png"
    if not patches:
        target.unlink(missing_ok=True)
        return None
    from PIL import Image
    with Image.open(_asset_path(asset_id)) as src:
        layer = Image.new("RGBA", src.size, (0, 0, 0, 0))
    for patch in patches:
        crop = Image.open(OUTPUT_DIR / patch["file"]).convert("RGBA")
        b = patch["bbox"]
        layer.alpha_composite(crop, (b["x"], b["y"]))
    layer.save(target)
    return target


def _patch_revision(asset_id: str) -> str:
    return hashlib.sha256(json.dumps(_load_patches(asset_id), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _composite_patches(asset_id: str, image):
    from PIL import Image
    flattened = OUTPUT_DIR / f"{asset_id}.patches.png"
    if not flattened.exists():
        return image
    return Image.alpha_composite(image.convert("RGBA"), Image.open(flattened).convert("RGBA"))


def _cleanse_cache_key(asset_id: str, manifest: TextManifest) -> str:
    """Hash only inputs which determine the erased base, never translations."""
    path = _asset_path(asset_id)
    geometry = []
    for inst in manifest.instances:
        b = inst.bounding_box
        geometry.append({"id": inst.id, "bbox": dataclasses.asdict(b) if b else None,
                         "polygon": inst.segmentation_mask.polygon if inst.segmentation_mask else None,
                         "background": dataclasses.asdict(inst.background_profile) if inst.background_profile else None,
                         "dnt": inst.dnt})
    payload = {
        "schema": "cleanse-provider-router-v4",
        "asset": [path.stat().st_mtime_ns, path.stat().st_size],
        "regions": geometry,
        # A provider promotion/configuration changes the pixels Cleanse is
        # permitted to generate, so it is part of cache identity.  This call
        # only reports configuration; it never imports or downloads weights.
        "providers": inpaint_providers.provider_statuses(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


def _cleansed_base(asset_id: str, manifest: TextManifest):
    """Shared cache for preview and final render; never writes a manifest."""
    from PIL import Image
    key = _cleanse_cache_key(asset_id, manifest)
    cache_dir = OUTPUT_DIR / "cleanse-cache"
    cache_dir.mkdir(exist_ok=True)
    image_path = cache_dir / f"{asset_id}-{key}.png"
    # The sidecar is keyed with the pixels.  A mutable per-asset sidecar made
    # a cache hit capable of restoring repair evidence for the wrong geometry
    # or provider revision.
    sidecar = cache_dir / f"{asset_id}-{key}.json"
    if image_path.exists():
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            provenance = cached.get("repair_provenance", {})
            for inst in manifest.instances:
                if inst.id in provenance:
                    inst.repair_provenance = provenance[inst.id]
        except Exception:
            # The cached image remains valid even if an old/partial sidecar
            # cannot be read.  The next clean recomputes evidence.
            pass
        return Image.open(image_path).convert("RGBA")
    cleaned = cleanse.erase(str(_asset_path(asset_id)), manifest,
                            candidate_observer=_candidate_observer(asset_id, key))
    if cleaned is None or not hasattr(cleaned, "save"):
        raise RuntimeError("cleanse produced no image")
    cleaned.save(image_path)
    provenance = {
        inst.id: inst.repair_provenance
        for inst in manifest.instances if inst.repair_provenance is not None
    }
    sidecar.write_text(json.dumps({
        "schema": "cleanse-cache-v4",
        "key": key,
        "image": image_path.name,
        "repair_provenance": provenance,
    }, sort_keys=True), encoding="utf-8")
    return cleaned.convert("RGBA")


@app.post("/api/inpaint")
def inpaint(req: InpaintRequest):
    if len(req.polygon) < 3 and not req.points:
        raise HTTPException(422, "a lasso polygon or brush stroke is required")
    _asset_path(req.asset_id)
    from tofu.layers.inpaint import make_patch
    try:
        # Treatment operates on the same cleansed/treatment base that the
        # localized canvas displays.  It must not sample source text back into
        # a region that Cleanse already removed.
        manifest = _dict_to_manifest(req.manifest) if req.manifest else load_manifest(UPLOAD_DIR, req.asset_id)
        if manifest is None:
            raise HTTPException(404, f"no manifest for asset '{req.asset_id}'")
        manifest.asset_id = req.asset_id
        base = _composite_patches(req.asset_id, _cleansed_base(req.asset_id, manifest))
        crop, bbox, strategy = make_patch(
            base, [tuple(p) for p in req.polygon], req.mode,
            [tuple(p) for p in req.points], req.radius, req.hardness,
            req.blur_strength, req.opacity,
            tuple(req.clone_source) if req.clone_source and len(req.clone_source) == 2 else None,
        )
        patch_id = uuid.uuid4().hex[:12]
        filename = f"{req.asset_id}.patch-{patch_id}.png"
        crop.save(OUTPUT_DIR / filename)
        patches = _load_patches(req.asset_id)
        patches.append({"id": patch_id, "file": filename, "bbox": bbox,
                        "polygon": req.polygon, "points": req.points,
                        "mode": req.mode, "strategy": strategy,
                        "radius": req.radius, "hardness": req.hardness,
                        "opacity": req.opacity, "clone_source": req.clone_source,
                        "parent_revision": _patch_revision(req.asset_id)})
        _archive_patches(req.asset_id, patches)
        _patch_index(req.asset_id).write_text(json.dumps(patches), encoding="utf-8")
        _flatten_patches(req.asset_id)
        return {"id": patch_id, "bbox": bbox, "patches": patches,
                "revision": _patch_revision(req.asset_id)}
    except Exception as exc:
        raise HTTPException(422, f"inpaint failed: {type(exc).__name__}: {exc}")


@app.post("/api/inpaint/candidate/{asset_id}/{candidate_id}")
def apply_repair_candidate(asset_id: str, candidate_id: str,
                           req: Optional[CandidateApplyRequest] = None):
    """Apply an editor-approved neural candidate as an undoable patch.

    Approval is explicit and affects only the treatment layer.  It never
    flips a provider promotion flag or rewrites the deterministic Cleanse
    cache, so later benchmark evidence remains attributable to the model.
    """
    _asset_path(asset_id)
    record = _find_candidate(asset_id, candidate_id)
    if record is None:
        raise HTTPException(404, "repair candidate not found for this asset")
    if req and req.cache_key and req.cache_key != record.get("cache_key"):
        raise HTTPException(409, "repair candidate belongs to a stale Cleanse preview")
    patches = _load_patches(asset_id)
    # Candidate selection is an explicit but idempotent editor action.  A
    # double-click or retry must return the existing treatment, never stack
    # the same RGBA overlay twice.
    existing = next((patch for patch in patches if patch.get("candidate_id") == candidate_id), None)
    if existing is not None:
        return {"id": existing["id"], "bbox": existing["bbox"], "patches": patches,
                "revision": _patch_revision(asset_id), "already_applied": True}
    candidate_root = _candidate_dir().resolve()
    source = (OUTPUT_DIR / str(record.get("file", ""))).resolve()
    if candidate_root not in source.parents or not source.is_file():
        raise HTTPException(422, "repair candidate artifact is unavailable")
    try:
        from PIL import Image
        patch_id = uuid.uuid4().hex[:12]
        filename = f"{asset_id}.patch-{patch_id}.png"
        Image.open(source).convert("RGBA").save(OUTPUT_DIR / filename)
        patches.append({
            "id": patch_id, "file": filename, "bbox": record["bbox"],
            "mode": "review_candidate", "strategy": record.get("provider", "neural"),
            "candidate_id": candidate_id, "parent_revision": _patch_revision(asset_id),
        })
        _archive_patches(asset_id, patches)
        _patch_index(asset_id).write_text(json.dumps(patches), encoding="utf-8")
        _flatten_patches(asset_id)
        return {"id": patch_id, "bbox": record["bbox"], "patches": patches,
                "revision": _patch_revision(asset_id), "already_applied": False}
    except Exception as exc:
        raise HTTPException(422, f"could not apply repair candidate: {type(exc).__name__}: {exc}")


@app.delete("/api/inpaint/{asset_id}")
@app.delete("/api/inpaint/{asset_id}/{patch_id}")
def undo_inpaint(asset_id: str, patch_id: Optional[str] = None):
    _asset_path(asset_id)
    patches = _load_patches(asset_id)
    removed = patches if patch_id is None else [p for p in patches if p.get("id") == patch_id]
    remaining = [] if patch_id is None else [p for p in patches if p.get("id") != patch_id]
    _archive_patches(asset_id, removed)
    # Patch PNGs stay in the private output directory until the asset itself
    # is cleaned up.  Deleting them here made redo/reset silently impossible.
    _patch_index(asset_id).write_text(json.dumps(remaining), encoding="utf-8")
    _flatten_patches(asset_id)
    return {"ok": True, "patches": remaining, "revision": _patch_revision(asset_id)}


@app.get("/api/treatment/{asset_id}")
def treatment_state(asset_id: str):
    _asset_path(asset_id)
    patches = _load_patches(asset_id)
    return {"patches": patches, "revision": _patch_revision(asset_id)}


@app.put("/api/treatment/{asset_id}")
def restore_treatment(asset_id: str, req: TreatmentRestoreRequest):
    """Restore an ordered treatment layer for unified localized undo/redo."""
    _asset_path(asset_id)
    archive = {patch.get("id"): patch for patch in _load_patch_archive(asset_id) if patch.get("id")}
    active = {patch.get("id"): patch for patch in _load_patches(asset_id) if patch.get("id")}
    archive.update(active)
    missing = [patch_id for patch_id in req.patch_ids if patch_id not in archive]
    if missing:
        raise HTTPException(422, f"treatment history is missing patch(es): {', '.join(missing)}")
    selected = [archive[patch_id] for patch_id in req.patch_ids]
    for patch in selected:
        if not (OUTPUT_DIR / str(patch.get("file", ""))).is_file():
            raise HTTPException(422, "treatment patch pixels are unavailable")
    _archive_patches(asset_id, selected)
    _patch_index(asset_id).write_text(json.dumps(selected), encoding="utf-8")
    _flatten_patches(asset_id)
    return {"patches": selected, "revision": _patch_revision(asset_id)}


@app.get("/api/localized-baseline/{asset_id}")
def get_localized_baseline(asset_id: str):
    _asset_path(asset_id)
    path = _localized_baseline_index(asset_id)
    if not path.is_file():
        raise HTTPException(404, "localized baseline has not been captured")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(422, f"localized baseline is unreadable: {type(exc).__name__}")


@app.put("/api/localized-baseline/{asset_id}")
def capture_localized_baseline(asset_id: str, req: LocalizedBaselineRequest):
    """Persist the first Render-entry state; manual edits never overwrite it."""
    _asset_path(asset_id)
    path = _localized_baseline_index(asset_id)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    # Validate snapshot before persisting it.  The baseline belongs to this
    # asset even if an accidental client payload claims another id.
    manifest = _dict_to_manifest(req.manifest)
    manifest.asset_id = asset_id
    payload = {"schema": 1, "manifest": jsonable(manifest), "patch_ids": req.patch_ids}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


@app.get("/api/inpainting/providers")
def inpainting_providers():
    """Local provider capabilities; never probes or downloads model weights."""
    return {"providers": inpaint_providers.provider_statuses()}


@app.post("/api/preview/render")
def preview_render(req: PreviewRenderRequest):
    """Build an isolated preview from an optional unsaved manifest."""
    path = _asset_path(req.asset_id)
    manifest = _dict_to_manifest(req.manifest) if req.manifest else load_manifest(UPLOAD_DIR, req.asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{req.asset_id}'")
    manifest.asset_id = req.asset_id
    manifest.targ_lang = req.targ_lang
    try:
        # Garnish and treatment edits keep the same geometry, material and
        # Cleanse base.  Re-running Scene/typography on every slider tick is
        # unnecessary latency; an incomplete manifest still takes the safe
        # full path.
        if not req.fast_path or not manifest.scene_regions:
            manifest = scene.analyze(str(path), manifest)
        cleansed = _cleansed_base(req.asset_id, manifest)
        patched = _composite_patches(req.asset_id, cleansed)
        if req.show_localized_text:
            with _matched_faces_applied(manifest):
                localized = scribe.render(patched, manifest, req.targ_lang, font_registry=get_validator().font_registry)
            localized = garnish.apply(localized, manifest, patched, get_validator().font_registry)
        else:
            localized = patched.copy()
        if localized is None or not hasattr(localized, "save"):
            raise RuntimeError("preview produced no image")
        # Preview requests can overlap while an editor drags a control.  A
        # fixed output name lets an older server request overwrite the bytes
        # later read by a newer response, even when the client ignores the
        # stale response.  Content-address the artifact by every composition
        # input so browser cancellation is an optimisation, not correctness.
        fingerprint = {
            "asset_id": req.asset_id,
            "target": req.targ_lang,
            "manifest": jsonable(manifest),
            "patch_revision": _patch_revision(req.asset_id),
            "cleanse_cache_key": _cleanse_cache_key(req.asset_id, manifest),
            "show_localized_text": req.show_localized_text,
        }
        version = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        name = f"{req.asset_id}-{req.targ_lang}-{version}.preview.png"
        localized.save(OUTPUT_DIR / name)
        return {
            "output_url": f"/outputs/{name}?v={version}",
            "text_manifest": jsonable(manifest),
            "cleanse_cache_key": _cleanse_cache_key(req.asset_id, manifest),
        }
    except Exception as exc:
        raise HTTPException(500, f"preview render failed: {type(exc).__name__}: {exc}")


@app.post("/api/preview/candidate/{asset_id}/{candidate_id}")
def preview_candidate_localized(asset_id: str, candidate_id: str, req: CandidatePreviewRequest):
    """Preview a background-only repair with current translated text layered on top."""
    record = _find_candidate(asset_id, candidate_id)
    if record is None:
        raise HTTPException(404, "repair candidate not found")
    manifest = _dict_to_manifest(req.manifest) if req.manifest else load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    manifest.asset_id = asset_id
    target = req.targ_lang or manifest.targ_lang or "en"
    try:
        from PIL import Image
        # Candidate artifacts are created from an already analysed manifest.
        # Re-running Scene for each thumbnail is pure latency once the caller
        # has supplied those regions.
        if not manifest.scene_regions:
            manifest = scene.analyze(str(_asset_path(asset_id)), manifest)
        # Candidate review must be composed over the same retained treatment
        # layer as the localized canvas, otherwise its text and surface are a
        # different image from the one the user will eventually apply to.
        base = _composite_patches(asset_id, _cleansed_base(asset_id, manifest)).convert("RGBA")
        candidate = Image.open(OUTPUT_DIR / record["file"]).convert("RGBA")
        b = record["bbox"]
        base.alpha_composite(candidate, (int(b["x"]), int(b["y"])))
        with _matched_faces_applied(manifest):
            localized = scribe.render(base, manifest, target, font_registry=get_validator().font_registry)
        localized = garnish.apply(localized, manifest, base, get_validator().font_registry)
        crop = localized.crop((int(b["x"]), int(b["y"]), int(b["x"] + b["width"]), int(b["y"] + b["height"])))
        # Candidate previews are requested while text/style edits are being
        # debounced.  A deterministic name lets an older request overwrite a
        # newer candidate composite, so make the artifact content-addressed by
        # the exact in-memory manifest and candidate evidence instead.
        fingerprint = {
            "candidate_id": candidate_id,
            "candidate_file": record.get("file"),
            "target": target,
            "manifest": req.manifest if req.manifest is not None else jsonable(manifest),
        }
        version = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        name = f"{asset_id}-{candidate_id}-{version}.localized-preview.png"
        crop.save(OUTPUT_DIR / name)
        return {"preview_url": f"/outputs/{name}?v={version}"}
    except Exception as exc:
        raise HTTPException(422, f"candidate preview failed: {type(exc).__name__}: {exc}")


@app.post("/api/render")
def render(req: RenderRequest):
    """render via TofuPipeline (MANUAL cicerone seeded with the stored
    manifest): tofu re-validation, scene enrichment, cleanse, scribe,
    verify, and memory all run through the one orchestrator, which also
    produces the structured logs/errors this endpoint returns."""
    path = _asset_path(req.asset_id)
    manifest = load_manifest(UPLOAD_DIR, req.asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{req.asset_id}'")
    manifest.targ_lang = req.targ_lang

    cfg = PipelineCfg(cicerone_mode=LayerMode.MANUAL)
    if req.qa_threshold is not None:
        cfg.qa_threshold = req.qa_threshold
    pipeline = TofuPipeline(cfg, font_registry=get_validator().font_registry)
    pipeline.set_manual_manifest(manifest)
    result = pipeline.process(str(path), req.targ_lang, font=req.font)

    # The synchronous endpoint is still retained for API clients, but its
    # image must be byte-for-byte derived from the same cleanse/patch/scribe
    # composition as preview and the SSE renderer.  Pipeline remains the
    # authority for validation, logging and memory decisions.
    if result.text_manifest is not None and not result.errors:
        try:
            shared_base = _composite_patches(
                req.asset_id, _cleansed_base(req.asset_id, result.text_manifest)
            )
            with _matched_faces_applied(result.text_manifest):
                result.output_asset = scribe.render(
                    shared_base, result.text_manifest, req.targ_lang,
                    font_registry=get_validator().font_registry,
                )
            result.verification_report = verify.build_verification_report(
                result.output_asset,
                result.text_manifest,
                get_validator().font_registry,
            )
            result.output_asset = garnish.apply(result.output_asset, result.text_manifest, shared_base, get_validator().font_registry)
            result.qa_report = verify.assess(
                result.output_asset, result.text_manifest, str(path), shared_base
            )
            result.success = bool(result.qa_report and result.qa_report.overall_score is not None
                                  and result.qa_report.overall_score >= cfg.qa_threshold)
            result.logs.append({"ts": datetime.now().strftime("%H:%M:%S"), "stage": "compose",
                                "level": "info", "message": "used shared preview/final cleanse cache and patch layer"})
        except Exception as exc:
            result.errors.append(f"shared render composition failed: {type(exc).__name__}: {exc}")

    logs = result.logs
    errors = result.errors

    # save output (endpoint concern: URL + metadata passthrough)
    output_url = None
    localized = result.output_asset
    t0 = time.time()
    if localized is not None and hasattr(localized, "save"):
        try:
            out_name = f"{req.asset_id}-{req.targ_lang}.png"
            save_kwargs = {}
            if hasattr(localized, "info"):
                if localized.info.get("dpi"): save_kwargs["dpi"] = localized.info["dpi"]
                if localized.info.get("exif"): save_kwargs["exif"] = localized.info["exif"]
                if localized.info.get("icc_profile"): save_kwargs["icc_profile"] = localized.info["icc_profile"]
            localized.save(OUTPUT_DIR / out_name, **save_kwargs)
            # The filename is stable across renders, so give the browser a
            # new URL after every successful write. Otherwise a perspective
            # edit followed by final render can display the previous PNG.
            output_url = f"/outputs/{out_name}?v={time.time_ns()}"
            logs.append({
                "ts": datetime.now().strftime("%H:%M:%S"), "stage": "save",
                "level": "info", "message": f"wrote {out_name}",
                "duration_ms": int((time.time() - t0) * 1000),
            })
        except Exception as exc:
            msg = f"save failed: {type(exc).__name__}: {exc}"
            errors.append(msg)
            logs.append({
                "ts": datetime.now().strftime("%H:%M:%S"), "stage": "save",
                "level": "error", "message": msg,
            })
    elif not errors:
        errors.append("render produced no output image (non-image asset or Pillow missing)")
        logs.append({
            "ts": datetime.now().strftime("%H:%M:%S"), "stage": "save",
            "level": "error", "message": "no renderable output produced",
        })

    pid = db.project_for_asset(req.asset_id)
    if pid:
        db.log_event(pid, "render",
                     f"rendered {req.asset_id} → {req.targ_lang}"
                     f" ({'ok' if output_url else 'failed'})")
        if result.memory_updates:
            # pipeline.py already logged the draft count via _run_memory;
            # this just does the actual db/thumbnail persistence
            _persist_tm_updates(pid, result.memory_updates)

    return {
        "output_url": output_url,
        "verification_report": (
            jsonable(result.verification_report)
            if result.verification_report else None
        ),
        "qa_report": jsonable(result.qa_report) if result.qa_report else None,
        "qa_passed": result.success and output_url is not None,
        "qa_threshold": cfg.qa_threshold,
        "validation_report": (
            jsonable(result.validation_report) if result.validation_report else None
        ),
        "text_manifest": jsonable(result.text_manifest or manifest),
        "logs": logs,
        "errors": errors,
    }


def _render_sse_event_payload(core_event: PipelineEvent) -> Optional[Dict[str, Any]]:
    """Adapt one canonical core event to the stable render SSE contract."""
    stage = core_event.stage
    operation = core_event.operation
    if stage in {"pipeline", "cicerone", "memory"}:
        return None
    if stage == "tofu" and operation == "revalidate":
        return None
    if stage == "scene" and operation == "prepass":
        return None

    payload: Dict[str, Any] = {
        "stage": stage,
        "status": {
            PipelineEventStatus.STARTED: "running",
            PipelineEventStatus.COMPLETED: "complete",
            PipelineEventStatus.WARNING: "complete",
            PipelineEventStatus.FAILED: "error",
            PipelineEventStatus.PAUSED: "paused",
        }[core_event.status],
    }
    if stage == "tofu" and operation == "preflight":
        if "passed" in core_event.payload:
            payload["passed"] = core_event.payload["passed"]
    if stage == "verify" and operation == "assess":
        payload["score"] = core_event.payload.get("overall_score")
    return payload


@app.get("/api/render/stream")
def render_stream(
    asset_id: str,
    targ_lang: str,
    font: Optional[str] = None,
    qa_threshold: Optional[float] = None,
    region_ids: Optional[str] = None,
):
    """SSE variant of /api/render: emits progress per layer (tofu, scene,
    tofu_regions, cleanse, scribe, verify) + a final payload shaped like
    /api/render's response. GET (not POST) so the browser's native
    EventSource can consume it, mirroring /api/detect/stream. Full renders
    consume the canonical TofuPipeline event observer; partial region renders
    retain their specialized prior-output composition path below.

    also closes the Phase-4-deferred item: after scene enrichment, every
    DISTINCT effective target language across regions (target_language
    overrides included) gets its own ToFU.validate() pass with the
    region's real font_px (Phase 1 typography) and effects context —
    the original pre-flight only ever validated the single request-level
    target language, never a region override, and never with the size/
    effects context that makes the render-quality prediction meaningful.

    region_ids (comma-separated, optional): the QA Inspector's per-region
    re-render action. When given, cleanse+scribe run ONLY on that instance
    subset, and the starting image is the asset's existing localized output
    (if one is on disk) rather than the raw source — so untouched regions
    keep their prior render instead of reverting to source text. Falls back
    to the raw source when no prior output exists yet (nothing to composite
    onto).
    """
    import dataclasses as _dc
    import json as _json
    from fastapi.responses import StreamingResponse
    from tofu.layers import scene, cleanse, scribe, verify

    path = _asset_path(asset_id)  # 404s before the stream opens
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    manifest.targ_lang = targ_lang
    threshold = qa_threshold if qa_threshold is not None else 0.8

    target_ids = set(region_ids.split(",")) if region_ids else None
    render_path = path
    if target_ids is not None:
        prior_output = OUTPUT_DIR / f"{asset_id}-{targ_lang}.png"
        if prior_output.exists():
            render_path = prior_output

    def event(data: Dict[str, Any]) -> str:
        return f"data: {_json.dumps(data, ensure_ascii=False)}\n\n"

    def canonical_gen():
        """Bridge synchronous core callbacks to a live SSE iterator.

        The worker owns processing.  The request iterator only serializes
        typed core events, so stage ordering cannot drift from Pipeline.
        """
        import queue
        import threading

        messages: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        region_validation: Dict[str, Any] = {
            "issues": [], "languages": [], "contexts": 0,
        }

        def observe(core_event: PipelineEvent) -> None:
            # Internal orchestration details remain available to other
            # observers, while this adapter preserves the established public
            # render-stream stage vocabulary.
            stage = core_event.stage
            operation = core_event.operation
            payload = _render_sse_event_payload(core_event)
            if payload is not None:
                messages.put(payload)

            # Per-region validation is a render API concern layered on top of
            # canonical Scene enrichment.  Emit it at the same public seam as
            # before without reproducing any core stage ordering.
            if (
                stage == "scene"
                and operation == "enrich"
                and core_event.status == PipelineEventStatus.COMPLETED
            ):
                messages.put({"stage": "tofu_regions", "status": "running"})
                messages.put({
                    "stage": "tofu_regions",
                    "status": "complete",
                    "issues": len(region_validation["issues"]),
                    "languages": region_validation["languages"],
                })

        def work() -> None:
            try:
                validator = get_validator()
                cfg = PipelineCfg(cicerone_mode=LayerMode.MANUAL)
                cfg.qa_threshold = threshold
                pipeline = TofuPipeline(
                    cfg,
                    font_registry=validator.font_registry,
                    on_event=observe,
                )
                pipeline.set_manual_manifest(manifest)

                # Preserve the render endpoint's shared Cleanse/patch
                # composition while letting Pipeline own when Cleanse runs.
                pipeline._run_cleanse = lambda asset, value: _composite_patches(
                    asset_id, _cleansed_base(asset_id, value)
                )
                canonical_scribe = pipeline._run_scribe

                def render_with_matched_faces(
                    cleansed_asset, value, target, render_params=None,
                ):
                    with _matched_faces_applied(value):
                        return canonical_scribe(
                            cleansed_asset, value, target, render_params
                        )

                pipeline._run_scribe = render_with_matched_faces
                canonical_scene = pipeline._run_scene

                def enrich_and_validate(asset, value):
                    enriched = canonical_scene(asset, value)
                    issues, languages, contexts = _revalidate_regions(
                        validator, path, enriched, targ_lang, font
                    )
                    region_validation["issues"] = issues
                    region_validation["languages"] = languages
                    region_validation["contexts"] = contexts
                    return enriched

                pipeline._run_scene = enrich_and_validate
                result = pipeline.process(str(path), targ_lang, font=font)

                if result.validation_report is not None:
                    result.validation_report.issues = (
                        list(result.validation_report.issues)
                        + list(region_validation["issues"])
                    )
                for issue in region_validation["issues"]:
                    if issue.severity.value == "error":
                        result.errors.append(
                            f"tofu {issue.code} ({issue.region_id}): {issue.message}"
                        )

                output_url = None
                localized = result.output_asset
                if localized is not None and hasattr(localized, "save"):
                    messages.put({"stage": "save", "status": "running"})
                    try:
                        out_name = f"{asset_id}-{targ_lang}.png"
                        localized.save(OUTPUT_DIR / out_name)
                        output_url = f"/outputs/{out_name}"
                        messages.put({"stage": "save", "status": "complete"})
                    except Exception as exc:
                        result.errors.append(
                            f"save failed: {type(exc).__name__}: {exc}"
                        )

                manifest2 = result.text_manifest or manifest
                save_manifest(UPLOAD_DIR, asset_id, manifest2)
                pid = db.project_for_asset(asset_id)
                tm_saved_count = 0
                if pid:
                    db.log_event(
                        pid, "render",
                        f"rendered {asset_id} â†’ {targ_lang} "
                        f"({'ok' if output_url else 'failed'})",
                    )
                    if result.memory_updates:
                        tm_saved_count = len(
                            _persist_tm_updates(pid, result.memory_updates)
                        )

                messages.put({
                    "stage": "complete",
                    "output_url": output_url,
                    "verification_report": (
                        jsonable(result.verification_report)
                        if result.verification_report else None
                    ),
                    "qa_report": (
                        jsonable(result.qa_report) if result.qa_report else None
                    ),
                    "qa_passed": result.success and output_url is not None,
                    "qa_threshold": threshold,
                    "validation_report": (
                        jsonable(result.validation_report)
                        if result.validation_report else None
                    ),
                    "text_manifest": jsonable(manifest2),
                    "tm_saved": tm_saved_count,
                    "logs": result.logs,
                    "errors": result.errors,
                })
            except Exception as exc:
                messages.put({
                    "stage": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                })
            finally:
                messages.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            payload = messages.get()
            if payload is None:
                return
            yield event(payload)

    if target_ids is None:
        return StreamingResponse(
            canonical_gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def gen():
        logs: List[Dict[str, Any]] = []
        errors: List[str] = []

        def log(stage: str, message: str, level: str = "info", t0: Optional[float] = None) -> None:
            entry = {"ts": datetime.now().strftime("%H:%M:%S"), "stage": stage, "level": level, "message": message}
            if t0 is not None:
                entry["duration_ms"] = int((time.time() - t0) * 1000)
            logs.append(entry)

        try:
            validator = get_validator()

            # -- tofu pre-flight (primary target language) --
            yield event({"stage": "tofu", "status": "running"})
            t0 = time.time()
            context = {"font": font} if font else None
            validation_report = validator.validate(str(path), targ_lang, context)
            log("tofu", f"pre-flight for '{targ_lang}': "
                        f"{'passed' if validation_report.passed else 'failed'}", t0=t0)
            yield event({"stage": "tofu", "status": "complete", "passed": validation_report.passed})
            if not validation_report.passed:
                yield event({
                    "stage": "complete", "output_url": None, "qa_report": None,
                    "qa_passed": False, "qa_threshold": threshold,
                    "validation_report": jsonable(validation_report),
                    "text_manifest": jsonable(manifest), "logs": logs,
                    "errors": [f"tofu {i.code}: {i.message}" for i in validation_report.issues
                              if i.severity.value == "error"],
                })
                return

            # -- scene enrichment --
            yield event({"stage": "scene", "status": "running"})
            t0 = time.time()
            try:
                manifest2 = scene.analyze(str(path), manifest)
                log("scene", f"enriched {len(manifest2.instances)} region(s)", t0=t0)
            except Exception as exc:
                manifest2 = manifest
                log("scene", f"enrichment failed ({type(exc).__name__}); "
                             "continuing with unenriched profiles", "warning", t0=t0)
            yield event({"stage": "scene", "status": "complete"})

            # -- per-region language re-validation (closes the Phase-4 --
            # deferred item: every effective target language, with real
            # per-region font_px/effects context)
            yield event({"stage": "tofu_regions", "status": "running"})
            t0 = time.time()
            region_issues, langs_seen, ctx_count = _revalidate_regions(
                validator, path, manifest2, targ_lang, font
            )
            log("tofu", f"re-validated {len(langs_seen)} distinct target "
                        f"language(s) across {ctx_count} typography context(s)", t0=t0)
            if region_issues:
                validation_report.issues = list(validation_report.issues) + region_issues
                for issue in region_issues:
                    if issue.severity.value == "error":
                        errors.append(f"tofu {issue.code} ({issue.region_id}): {issue.message}")
            yield event({
                "stage": "tofu_regions", "status": "complete",
                "issues": len(region_issues), "languages": langs_seen,
            })

            # per-region font merge (mirrors TofuPipeline._process_static)
            render_params: Optional[Dict[str, RenderParams]] = None
            if font:
                render_params = {}
                for inst in manifest2.instances:
                    base = inst.style_profile
                    render_params[inst.id] = RenderParams(
                        position=inst.bounding_box,
                        style=StyleProfil(
                            font_family=(base.font_family if base and base.font_family else font),
                            font_weight=base.font_weight if base else None,
                            color=base.color if base else None,
                            shadow=base.shadow if base else None,
                            effects=base.effects if base else None,
                        ),
                    )

            # partial re-render: only the requested regions get cleansed +
            # re-scribed; everything else in render_path (the prior output,
            # when one exists) is left untouched
            erase_manifest = manifest2
            if target_ids is not None:
                erase_manifest = _dc.replace(
                    manifest2,
                    instances=[i for i in manifest2.instances if i.id in target_ids],
                )

            # -- cleanse --
            yield event({"stage": "cleanse", "status": "running"})
            t0 = time.time()
            try:
                # A full render shares the exact cached cleanse base used by
                # preview.  Partial re-renders intentionally start from the
                # prior localized output, so they retain their old behavior.
                cleansed_asset = (
                    _cleansed_base(asset_id, erase_manifest)
                    if target_ids is None else cleanse.erase(str(render_path), erase_manifest)
                )
                cleansed_asset = _composite_patches(asset_id, cleansed_asset)
                log("cleanse", "erased region(s)", t0=t0)
            except Exception as exc:
                log("cleanse", f"failed: {type(exc).__name__}: {exc}", "error", t0=t0)
                errors.append(f"cleanse failed: {exc}")
                yield event({
                    "stage": "complete", "output_url": None, "qa_report": None,
                    "qa_passed": False, "qa_threshold": threshold,
                    "validation_report": jsonable(validation_report),
                    "text_manifest": jsonable(manifest2), "logs": logs, "errors": errors,
                })
                return
            yield event({"stage": "cleanse", "status": "complete"})

            # -- scribe --
            yield event({"stage": "scribe", "status": "running"})
            t0 = time.time()
            try:
                with _matched_faces_applied(erase_manifest) as matched:
                    localized = scribe.render(
                        cleansed_asset, erase_manifest, targ_lang, render_params,
                        font_registry=validator.font_registry,
                    )
                verification_report = verify.build_verification_report(
                    localized, erase_manifest, validator.font_registry
                )
                verification_report.run_metadata["scope"] = (
                    "partial" if target_ids is not None else "project"
                )
                if target_ids is not None:
                    verification_report.run_metadata["region_ids"] = sorted(target_ids)
                localized = garnish.apply(localized, manifest2, cleansed_asset, validator.font_registry)
                log("scribe", f"rendered target text for '{targ_lang}'", t0=t0)
                if matched:
                    log("scribe", f"{len(matched)} region(s) drawn with the matched face "
                                  f"({', '.join(matched)})")
            except Exception as exc:
                log("scribe", f"failed: {type(exc).__name__}: {exc}", "error", t0=t0)
                errors.append(f"scribe failed: {exc}")
                yield event({
                    "stage": "complete", "output_url": None, "qa_report": None,
                    "qa_passed": False, "qa_threshold": threshold,
                    "validation_report": jsonable(validation_report),
                    "text_manifest": jsonable(manifest2), "logs": logs, "errors": errors,
                })
                return
            fallback_ids = [i.id for i in manifest2.instances if i.glyph_fallback]
            if fallback_ids:
                log("tofu", f"glyph fallback applied for region(s) {', '.join(fallback_ids)}: "
                            "the requested font lacked codepoints for the target text; "
                            "scribe swapped to the best-covering font it found", "warning")
            yield event({"stage": "scribe", "status": "complete"})

            # -- verify --
            yield event({"stage": "verify", "status": "running"})
            t0 = time.time()
            qa_report = None
            try:
                qa_report = verify.assess(localized, manifest2, str(path), cleansed_asset)
                score = qa_report.overall_score
                log("verify", "overall QA " + (f"{score:.2f}" if score is not None else "n/a"), t0=t0)
            except Exception as exc:
                log("verify", f"failed: {type(exc).__name__}: {exc}", "warning", t0=t0)
            qa_passed = (
                qa_report is not None and qa_report.overall_score is not None
                and qa_report.overall_score >= threshold
            )
            yield event({
                "stage": "verify", "status": "complete",
                "score": qa_report.overall_score if qa_report else None,
                "verification_status": verification_report.project.overall_status,
            })

            # -- memory: only QA-approved regions are remembered --
            pid = db.project_for_asset(asset_id)
            tm_saved_count = 0
            if qa_passed and pid:
                t0 = time.time()
                try:
                    drafts = memory.update(
                        manifest2, targ_lang, localized, str(path),
                        qa_report, threshold,
                    )
                    tm_saved = _persist_tm_updates(pid, drafts)
                    tm_saved_count = len(tm_saved)
                    log("memory", f"{tm_saved_count} TM record(s) stored", t0=t0)
                except Exception as exc:
                    log("memory", f"failed: {type(exc).__name__}: {exc}", "warning", t0=t0)

            # -- save output --
            output_url = None
            t0 = time.time()
            if localized is not None and hasattr(localized, "save"):
                try:
                    out_name = f"{asset_id}-{targ_lang}.png"
                    save_kwargs = {}
                    if hasattr(localized, "info"):
                        if localized.info.get("dpi"): save_kwargs["dpi"] = localized.info["dpi"]
                        if localized.info.get("exif"): save_kwargs["exif"] = localized.info["exif"]
                        if localized.info.get("icc_profile"): save_kwargs["icc_profile"] = localized.info["icc_profile"]
                    localized.save(OUTPUT_DIR / out_name, **save_kwargs)
                    output_url = f"/outputs/{out_name}"
                    log("save", f"wrote {out_name}", t0=t0)
                except Exception as exc:
                    msg = f"save failed: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    log("save", msg, "error")
            elif not errors:
                errors.append("render produced no output image (non-image asset or Pillow missing)")
                log("save", "no renderable output produced", "error")

            save_manifest(UPLOAD_DIR, asset_id, manifest2)
            if pid:
                db.log_event(pid, "render",
                             f"rendered {asset_id} → {targ_lang} ({'ok' if output_url else 'failed'})")

            yield event({
                "stage": "complete",
                "output_url": output_url,
                "verification_report": jsonable(verification_report),
                "qa_report": jsonable(qa_report) if qa_report else None,
                "qa_passed": qa_passed and output_url is not None,
                "qa_threshold": threshold,
                "validation_report": jsonable(validation_report),
                "text_manifest": jsonable(manifest2),
                "tm_saved": tm_saved_count,
                "logs": logs,
                "errors": errors,
            })
        except Exception as exc:
            yield event({"stage": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/render/approve")
def approve_render(req: ApproveRequest):
    """QA Inspector sign-off: records the coverage/score snapshot the user
    approved into project history. A human decision, not a derived fact —
    logged as its own event kind so it's distinguishable from an ordinary
    render in History."""
    _asset_path(req.asset_id)  # 404s on a missing asset
    pid = db.project_for_asset(req.asset_id)
    if not pid:
        raise HTTPException(404, f"asset '{req.asset_id}' is not attached to a project")
    score_str = f"{req.overall_score:.2f}" if req.overall_score is not None else "n/a"
    coverage_str = (
        f"{req.rendered}/{req.regions_total} rendered"
        + (f", {req.dnt} DNT" if req.dnt else "")
        if req.regions_total is not None else "coverage n/a"
    )
    db.log_event(
        pid, "qa-approved",
        f"approved {req.asset_id} → {req.targ_lang}: QA {score_str} ({coverage_str})",
    )
    return {"ok": True}


# --- legacy process (backwards compat) ---

@app.post("/api/process")
def process(req: ProcessRequest):
    path = _asset_path(req.asset_id)
    cfg = PipelineCfg()
    if req.qa_threshold is not None:
        cfg.qa_threshold = req.qa_threshold
    def apply_translations(name: str, payload: Any) -> Any:
        if req.translations and hasattr(payload, "instances"):
            for inst in payload.instances:
                if inst.id in req.translations:
                    inst.target_text = req.translations[inst.id]
        return payload
    pipeline = TofuPipeline(
        cfg, on_checkpoint=apply_translations, font_registry=get_validator().font_registry
    )
    result = pipeline.process(str(path), req.targ_lang)
    output_url = None
    output = result.output_asset
    if output is not None and hasattr(output, "save"):
        out_name = f"{req.asset_id}-{req.targ_lang}.png"
        output.save(OUTPUT_DIR / out_name)
        output_url = f"/outputs/{out_name}"
    # memory_updates carries raw PIL thumb_crop images (jsonable() has no
    # handler for them, same bug class as Phase 5's numpy-scalar leak) --
    # persist_tm_updates strips it and does the actual db/thumbnail write
    pid = db.project_for_asset(req.asset_id)
    result.memory_updates = _persist_tm_updates(pid, result.memory_updates)
    payload = jsonable(result)
    payload.pop("output_asset", None)
    payload["output_url"] = output_url
    return payload
