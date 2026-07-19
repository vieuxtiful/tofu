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
)
from tofu.layers.tofu import ToFU, lang_to_script
from tofu.layers import cicerone
from tofu.layers.cicerone import _to_easyocr_lang
from tofu.utils.manifest_store import save_manifest, load_manifest
from tofu.utils import interchange

import db

UPLOAD_DIR = ROOT / "server" / "uploads"
OUTPUT_DIR = ROOT / "server" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


# --- upload + languages + fonts ---

@app.post("/api/assets")
async def upload_asset(file: UploadFile = File(...), project_id: Optional[str] = None):
    suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
    asset_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{asset_id}{suffix}"
    dest.write_bytes(await file.read())
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
        db.link_asset(project_id, asset_id, file.filename)
        db.log_event(project_id, "asset-uploaded",
                     f"uploaded '{file.filename}' ({asset_id})")
    return {
        "asset_id": asset_id, "filename": file.filename,
        "asset_info": jsonable(info),
        "asset_url": f"/uploads/{asset_id}{suffix}",
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
        manifest = cicerone.detect(str(path), info, adaptive=False)
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
        str(path), info, backend=backend, scene_regions=scene_regions
    )
    manifest.src_lang = _infer_src_lang(manifest)
    # scene enrichment at CAPTURE time (not just render): style/background
    # profiles + typography power the capture tooltips and region table
    try:
        manifest = scene.analyze(str(path), manifest)
    except Exception:
        pass  # enrichment is best-effort; detection results stand alone
    save_manifest(UPLOAD_DIR, req.asset_id, manifest)
    _record_detection(req.asset_id, manifest)
    payload = jsonable(manifest)
    payload["engine"] = _engine_name(backend)
    return payload


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
            save_manifest(UPLOAD_DIR, asset_id, manifest)
            _record_detection(asset_id, manifest)
            yield event({
                "stage": "complete",
                "manifest": jsonable(manifest),
                "engine": _engine_name(backend),
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
    return jsonable(manifest)


@app.put("/api/manifest/{asset_id}")
def put_manifest(asset_id: str, manifest_data: Dict[str, Any]):
    from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict
    manifest = _dict_to_manifest(manifest_data)
    manifest.asset_id = asset_id
    manifest.total_regions = len(manifest.instances)
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    # autosave ledger: every accepted write is snapshotted (deduped) so a
    # session can always be rolled back
    pid = db.project_for_asset(asset_id)
    if pid:
        db.add_snapshot(pid, asset_id, _manifest_to_dict(manifest), reason="autosave")
    return {"ok": True, "total_regions": manifest.total_regions}


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
    manifest = load_manifest(UPLOAD_DIR, asset_id)
    if manifest is None:
        raise HTTPException(404, f"no manifest for asset '{asset_id}'")
    manifest.instances = [i for i in manifest.instances if i.id != rid]
    for idx, inst in enumerate(manifest.instances):
        inst.reading_order = idx
    manifest.total_regions = len(manifest.instances)
    save_manifest(UPLOAD_DIR, asset_id, manifest)
    return {"ok": True, "total_regions": manifest.total_regions}


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
    if req.target_language is not None: inst.target_language = req.target_language
    if req.language is not None: inst.language = req.language
    if req.font is not None:
        from tofu.core.types import StyleProfil
        if inst.style_profile is None:
            inst.style_profile = StyleProfil()
        inst.style_profile.font_family = req.font
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
    pipeline = TofuPipeline(cfg)
    pipeline.set_manual_manifest(manifest)
    result = pipeline.process(str(path), req.targ_lang, font=req.font)

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
    pipeline = TofuPipeline(cfg, on_checkpoint=apply_translations)
    result = pipeline.process(str(path), req.targ_lang)
    output_url = None
    output = result.output_asset
    if output is not None and hasattr(output, "save"):
        out_name = f"{req.asset_id}-{req.targ_lang}.png"
        output.save(OUTPUT_DIR / out_name)
        output_url = f"/outputs/{out_name}"
    payload = jsonable(result)
    payload.pop("output_asset", None)
    payload["output_url"] = output_url
    return payload
