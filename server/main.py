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
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tofu.core.pipeline import TofuPipeline
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
}


app = FastAPI(title="ToFU", version="0.2.0")
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


# --- upload + languages + fonts ---

@app.post("/api/assets")
async def upload_asset(file: UploadFile = File(...), project_id: Optional[str] = None):
    suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
    asset_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{asset_id}{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    content_hash = hashlib.sha256(data).hexdigest()
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
        if db.get_project(project_id) is None:
            raise HTTPException(404, f"project '{project_id}' not found")
        db.link_asset(project_id, asset_id, file.filename, content_hash)
        db.log_event(project_id, "asset-uploaded",
                     f"uploaded '{file.filename}' ({asset_id})")
    return {
        "asset_id": asset_id, "filename": file.filename,
        "asset_info": jsonable(info),
        "asset_url": f"/uploads/{asset_id}{suffix}",
    }


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
    return db.create_project(name, req.target_lang, req.asset_kind)


@app.get("/api/projects")
def list_projects():
    return {"projects": db.list_projects()}


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
        pid, name=req.name, target_lang=req.target_lang, source_lang=req.source_lang
    )
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
            start = time.time()
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

            detections = []
            if isinstance(backend, cicerone.EasyOCRBackend):
                for n, (tt, lt) in enumerate(cicerone.PASS_THRESHOLDS, 1):
                    yield event({"stage": "cicerone", "pass": n, "status": "running"})
                    passed = backend.detect(str(path), text_threshold=tt, low_text=lt)
                    cicerone.tag_detection_pass(
                        passed, engine=backend, pass_number=n,
                        text_threshold=tt, low_text=lt,
                    )
                    detections = cicerone.merge_detections(detections, passed)
                    yield event({
                        "stage": "cicerone", "pass": n, "status": "complete",
                        "regions": len(detections),
                    })
            else:
                # PaddleOCR / single-pass backend
                passed = backend.detect(str(path))
                cicerone.tag_detection_pass(
                    passed, engine=backend, pass_number=1,
                )
                detections = cicerone.merge_detections(detections, passed)
                yield event({
                    "stage": "cicerone", "pass": 1, "status": "complete",
                    "regions": len(detections),
                })

            yield event({"stage": "finalize", "status": "running"})
            manifest = cicerone.build_manifest(
                str(path), detections,
                asset_info=info, engine=backend,
                scene_regions=regions, start=start,
            )

            # language-adaptive stage 2 (mirrors cicerone.detect): when the
            # unhinted pass identified a language the stage-1 charset could
            # not express, re-detect with a tuned reader — this is where
            # wrong-charset garbage regions become real recall
            if lang_hints is None and isinstance(backend, cicerone.EasyOCRBackend):
                target = cicerone.refine_langset(manifest.instances, backend)
                # scene-surface probe: rescues vertical CJK signage whose
                # fragments carry no usable instance evidence
                surface_dets = []
                if target is None and regions:
                    target, surface_dets = cicerone.probe_uncovered_surfaces(
                        str(path), regions, manifest.instances, backend
                    )
                if target:
                    yield event({
                        "stage": "refine", "status": "running",
                        "langset": list(target),
                    })
                    tuned = cicerone.EasyOCRBackend(languages=target, gpu=gpu)
                    second = cicerone.run_multipass(tuned, str(path))
                    if surface_dets:
                        second = cicerone.union_prefer_primary(surface_dets, second)
                    merged = cicerone.union_prefer_primary(second, detections)
                    manifest = cicerone.build_manifest(
                        str(path), merged,
                        asset_info=info, engine=tuned,
                        scene_regions=regions, start=start,
                    )
                    backend = tuned
                    yield event({
                        "stage": "refine", "status": "complete",
                        "regions": len(manifest.instances),
                    })

            # coarse-to-fine zoom pass: re-detect scene surfaces at 2x —
            # fine boxes replace coarse multi-sign boxes they overlap
            if not isinstance(backend, cicerone.NullBackend) and regions:
                yield event({"stage": "zoom", "status": "running"})
                fine = cicerone.zoom_detect(backend, str(path), regions)
                if fine:
                    detections = cicerone.union_prefer_primary(fine, detections)
                    manifest = cicerone.build_manifest(
                        str(path), detections,
                        asset_info=info, engine=backend,
                        scene_regions=regions, start=start,
                    )
                yield event({
                    "stage": "zoom", "status": "complete",
                    "regions": len(manifest.instances),
                })

            # vertical-stack re-split: a detection box far taller than
            # wide is likely CRAFT over-merging several stacked
            # vertical-CJK characters into one box (see
            # cicerone._split_tall_detections) -- this endpoint calls
            # build_manifest() directly (not cicerone.detect(), which
            # already runs this as its own step) so it needs its own
            # explicit stage here for parity, same as savor below
            if isinstance(backend, cicerone.EasyOCRBackend):
                yield event({"stage": "vertical_split", "status": "running"})
                split = cicerone._split_tall_detections(str(path), backend, detections)
                if split is not None:
                    detections = split
                    manifest = cicerone.build_manifest(
                        str(path), detections,
                        asset_info=info, engine=backend,
                        scene_regions=regions, start=start,
                    )
                yield event({
                    "stage": "vertical_split", "status": "complete",
                    "regions": len(manifest.instances),
                })

            # PaddleOCR rescue: a second, differently-architected engine
            # for whatever EasyOCR's own passes above still leave weak or
            # entirely undetected on a CJK-dominant scene -- self-gating
            # (should_paddle_rescue) and best-effort, same as savor/menu
            # below. backend is already the CJK-tuned reader by this
            # point if the refine stage above fired, so no separate
            # "avoid re-triggering the expensive langset probe" handling
            # is needed here the way cicerone.detect() needs it.
            if (isinstance(backend, cicerone.EasyOCRBackend)
                    and cicerone.PaddleOCRBackend.is_available()):
                should_rescue, dominant = cicerone.should_paddle_rescue(
                    manifest.instances, regions
                )
                if should_rescue:
                    yield event({"stage": "paddle_rescue", "status": "running"})
                    try:
                        rescued = cicerone.run_paddle_rescue(
                            str(path), detections, regions, dominant, gpu=gpu,
                        )
                    except Exception:
                        rescued = None
                    if rescued is not None:
                        detections = rescued
                        manifest = cicerone.build_manifest(
                            str(path), detections,
                            asset_info=info, engine=backend,
                            scene_regions=regions, start=start,
                        )
                    yield event({
                        "stage": "paddle_rescue", "status": "complete",
                        "regions": len(manifest.instances),
                    })

            # second-look recognition on surviving weak regions
            if not isinstance(backend, cicerone.NullBackend) and manifest.instances:
                yield event({"stage": "polish", "status": "running"})
                improved = cicerone.second_look(
                    str(path), manifest.instances, backend
                )
                yield event({
                    "stage": "polish", "status": "complete",
                    "regions": improved,
                })

            if (cicerone._engine_from_env() in {"auto", "hybrid"}
                    and isinstance(backend, cicerone.EasyOCRBackend)
                    and manifest.instances):
                yield event({"stage": "hybrid_audit", "status": "running"})
                corrected = cicerone.hybrid_audit(str(path), manifest.instances)
                yield event({"stage": "hybrid_audit", "status": "complete", "corrected": corrected})

            # Savor's taste test on the FINAL recognized text -- this
            # endpoint calls build_manifest()/second_look() directly
            # (not cicerone.detect(), which already runs Savor as its
            # own last step) to interleave progress events per pass, so
            # Savor needs its own explicit stage here for parity
            if manifest.instances:
                yield event({"stage": "savor", "status": "running"})
                from tofu.layers.savor import taste
                swallowed = taste(str(path), manifest.instances, font_registry=get_validator().font_registry)
                yield event({"stage": "savor", "status": "complete", "corrected": swallowed})

            # gazetteer correction on whatever text Savor left behind --
            # Japanese/simplified-Chinese glyph normalization -- this
            # endpoint calls build_manifest()/taste() directly (not
            # cicerone.detect(), which already runs wasabi.season() as
            # its own step), so wasabi needs its own explicit stage here
            # for parity, same as savor above. runs before menu so its
            # gazetteer fuzzy-match sees corrected characters.
            if manifest.instances:
                yield event({"stage": "wasabi", "status": "running"})
                from tofu.layers.wasabi import season
                normalized = season(manifest.instances)
                yield event({"stage": "wasabi", "status": "complete", "corrected": normalized})

            # this endpoint calls build_manifest()/taste() directly (not
            # cicerone.detect(), which already runs menu.browse() as its
            # own last step), so menu needs its own explicit stage here
            # for parity, same as savor above
            if manifest.instances:
                yield event({"stage": "menu", "status": "running"})
                from tofu.layers.menu import browse
                matched = browse(
                    manifest.instances, asset=str(path),
                    font_registry=get_validator().font_registry,
                )
                yield event({"stage": "menu", "status": "complete", "corrected": matched})

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
            cicerone.label_latin_languages(manifest.instances)
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


@app.post("/api/manifest/{asset_id}/regions")
def add_region(asset_id: str, req: RegionCreate):
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
        bounding_box=BBox(x=req.x, y=req.y, width=req.width, height=req.height),
        text=req.text, target_text=req.target_text,
        reading_order=len(manifest.instances),
    )
    manifest.instances.append(new_inst)
    manifest.total_regions = len(manifest.instances)
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    return jsonable(new_inst)


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
    if req.x is not None: inst.bounding_box.x = req.x
    if req.y is not None: inst.bounding_box.y = req.y
    if req.width is not None: inst.bounding_box.width = req.width
    if req.height is not None: inst.bounding_box.height = req.height
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
    bbox = req.bbox
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise HTTPException(500, "Pillow/numpy not available")
    try:
        img = Image.open(str(path)).convert("RGB")
    except Exception as exc:
        raise HTTPException(500, f"cannot open asset image: {exc}")
    crop = img.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]))
    if crop.width == 0 or crop.height == 0:
        return {"text": "", "confidence": 0.0, "detected_language": None}
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
    bbox = req.bbox
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    if w <= 0 or h <= 0:
        raise HTTPException(400, "bbox must have positive width and height")

    import os
    if req.engine:
        os.environ["OCR_ENGINE"] = req.engine.lower()

    from tofu.core.types import BBox as TofuBBox
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

    region = TofuBBox(x=x, y=y, width=w, height=h)
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
            req.blur_strength,
        )
        patch_id = uuid.uuid4().hex[:12]
        filename = f"{req.asset_id}.patch-{patch_id}.png"
        crop.save(OUTPUT_DIR / filename)
        patches = _load_patches(req.asset_id)
        patches.append({"id": patch_id, "file": filename, "bbox": bbox,
                        "polygon": req.polygon, "points": req.points,
                        "mode": req.mode, "strategy": strategy,
                        "radius": req.radius, "hardness": req.hardness,
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
        with _matched_faces_applied(manifest):
            localized = scribe.render(patched, manifest, req.targ_lang, font_registry=get_validator().font_registry)
        localized = garnish.apply(localized, manifest, patched, get_validator().font_registry)
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
    EventSource can consume it, mirroring /api/detect/stream. Runs the
    layers directly rather than through TofuPipeline (same reason
    /api/detect/stream doesn't use the pipeline either: the pipeline is
    one synchronous call with no per-stage yield points).

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
