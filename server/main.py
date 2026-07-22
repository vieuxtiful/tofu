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
  POST /api/export                          export to XLIFF/TMX/TSV/CSV/TXT
  POST /api/import                          import translated file
  POST /api/render                          run scene→cleanse→scribe→verify
  POST /api/process                         legacy full pipeline

run:  uvicorn main:app --reload --port 8000   (from server/)
"""

import dataclasses
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    RenderParams, StyleProfil,
)
from tofu.layers.tofu import ToFU, lang_to_script
from tofu.layers import cicerone, memory, scribe, cleanse, scene, verify, inpaint_providers
from tofu.layers.cicerone import _to_easyocr_lang
from tofu.utils.manifest_store import save_manifest, load_manifest, _dict_to_manifest
from tofu.utils import interchange

import db

UPLOAD_DIR = ROOT / "server" / "uploads"
OUTPUT_DIR = ROOT / "server" / "outputs"
TM_THUMB_DIR = ROOT / "server" / "tm_thumbs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TM_THUMB_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()


def _font_dir() -> Optional[str]:
    for cand in (
        os.environ.get("TOFU_FONT_DIR"),
        str(ROOT / "server" / "fonts"),
        "C:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts",
    ):
        if cand and Path(cand).exists():
            return cand
    return None


_validator: Optional[ToFU] = None


def get_validator() -> ToFU:
    global _validator
    if _validator is None:
        _validator = ToFU(font_library_path=_font_dir())
    return _validator


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
        if inst.style_profile and inst.style_profile.font_family:
            continue  # explicit pick -- nothing to resolve
        text = inst.target_text or inst.text or ""
        lang = inst.target_language or default_targ_lang or manifest.targ_lang
        weight = inst.style_profile.font_weight if inst.style_profile else None
        italic = bool(inst.style_profile.italic) if inst.style_profile else False
        path = scribe.resolve_auto_font(registry, lang, text, weight=weight, italic=italic)
        if path != inst.resolved_font_family:
            inst.resolved_font_family = path
        if path:
            resolved += 1
    return resolved


def _infer_src_lang(manifest: TextManifest) -> str:
    """dominant detected language, weighted by region area — a storefront
    sign outvotes a handful of small incidental latin fragments."""
    votes: Dict[str, float] = Counter()
    for i in manifest.instances:
        if not i.detected_language:
            continue
        area = (
            i.bounding_box.width * i.bounding_box.height
            if i.bounding_box is not None else 1
        )
        votes[i.detected_language] += max(1, area)
    if not votes:
        return "en"
    return max(votes, key=lambda k: votes[k])


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

class InpaintRequest(BaseModel):
    asset_id: str
    polygon: List[List[int]] = []
    points: List[List[int]] = []
    mode: str = "auto"
    radius: int = 18
    hardness: float = 0.85
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


@app.get("/api/fonts")
def fonts(lang: str, limit: int = 8):
    script = lang_to_script.get(lang)
    if script is None:
        raise HTTPException(400, f"unknown language '{lang}'")
    validator = get_validator()
    if validator.font_registry is None:
        return {"script": script, "fonts": [], "families": []}
    ranked = validator.font_registry.recommend(script, lang, limit=limit)
    families = validator.font_registry.families_with_weights(script, lang, limit=limit)
    return {
        "script": script,
        "fonts": [{"path": p, "coverage": round(c, 4)} for p, c in ranked],
        "families": families,
    }


@app.get("/api/font-file")
def serve_font_file(path: str):
    """serve a font file for @font-face preview in the frontend."""
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "font not found")
    ext = p.suffix.lower()
    media = {
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".ttc": "font/collection",
        ".otc": "font/collection",
    }.get(ext, "application/octet-stream")
    return Response(content=p.read_bytes(), media_type=media)


@app.post("/api/validate")
def validate(req: ValidateRequest):
    path = _asset_path(req.asset_id)
    context = {"font": req.font} if req.font else None
    report = get_validator().validate(str(path), req.targ_lang, context)
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
                    detections = cicerone.merge_detections(detections, passed)
                    yield event({
                        "stage": "cicerone", "pass": n, "status": "complete",
                        "regions": len(detections),
                    })
            else:
                # PaddleOCR / single-pass backend
                passed = backend.detect(str(path))
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
    return jsonable(manifest)


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
    content = (await file.read()).decode("utf-8-sig")
    filename = file.filename or "import.txt"
    translations = interchange.import_file(filename, content)
    manifest_ids = {i.id for i in manifest.instances}
    imported_ids = set(translations.keys())
    missing = list(manifest_ids - imported_ids)
    extra = list(imported_ids - manifest_ids)
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
    return {"imported": len(translations), "missing": missing, "extra": extra, "translations": translations}


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
        manifest = scene.analyze(str(path), manifest)
        cleansed = _cleansed_base(req.asset_id, manifest)
        cleansed = _composite_patches(req.asset_id, cleansed)
        localized = scribe.render(cleansed, manifest, req.targ_lang, font_registry=get_validator().font_registry)
        if localized is None or not hasattr(localized, "save"):
            raise RuntimeError("preview produced no image")
        name = f"{req.asset_id}-{req.targ_lang}.preview.png"
        localized.save(OUTPUT_DIR / name)
        return {
            "output_url": f"/outputs/{name}?v={int(time.time() * 1000)}",
            "text_manifest": jsonable(manifest),
            "cleanse_cache_key": _cleanse_cache_key(req.asset_id, manifest),
        }
    except Exception as exc:
        raise HTTPException(500, f"preview render failed: {type(exc).__name__}: {exc}")


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
            result.output_asset = scribe.render(
                shared_base, result.text_manifest, req.targ_lang,
                font_registry=get_validator().font_registry,
            )
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
            output_url = f"/outputs/{out_name}"
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
            lang_groups: Dict[str, List[InstText]] = {}
            for inst in manifest2.instances:
                if inst.dnt or not inst.target_text:
                    continue
                lang_groups.setdefault(inst.target_language or targ_lang, []).append(inst)
            region_issues = []
            for lang, insts in lang_groups.items():
                for inst in insts:
                    ctx: Dict[str, Any] = {"font": font} if font else {}
                    if inst.characteristics and inst.characteristics.size:
                        ctx["font_px"] = inst.characteristics.size
                    sp = inst.style_profile
                    effects = []
                    if sp:
                        if sp.shadow: effects.append("shadow")
                        if sp.stroke_width: effects.append("stroke")
                        if sp.italic: effects.append("italic")
                    if effects:
                        ctx["effects"] = effects
                    report = validator.validate(str(path), lang, ctx or None)
                    for issue in report.issues:
                        if issue.region_id is None:
                            issue.region_id = inst.id
                        region_issues.append(issue)
            log("tofu", f"re-validated {len(lang_groups)} distinct target "
                        f"language(s) across regions", t0=t0)
            if region_issues:
                validation_report.issues = list(validation_report.issues) + region_issues
                for issue in region_issues:
                    if issue.severity.value == "error":
                        errors.append(f"tofu {issue.code} ({issue.region_id}): {issue.message}")
            yield event({
                "stage": "tofu_regions", "status": "complete",
                "issues": len(region_issues), "languages": list(lang_groups.keys()),
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
                localized = scribe.render(
                    cleansed_asset, erase_manifest, targ_lang, render_params,
                    font_registry=validator.font_registry,
                )
                log("scribe", f"rendered target text for '{targ_lang}'", t0=t0)
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
