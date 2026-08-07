"""Read-only runtime capability inventory for the API and diagnostics UI.

This module deliberately performs no model construction, downloads, or
remote probes.  A capability request must be cheap and safe even on a host
where every optional backend is absent.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


CAPABILITIES_SCHEMA_VERSION = "1.0"


def _video_status() -> Dict[str, Any]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    cv = _module_available("cv2")
    ready = bool(ffmpeg and ffprobe and cv)
    missing = [name for name, value in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe), ("OpenCV", cv)) if not value]
    return _entry("video_pipeline", ready, ready=ready,
                  reason=None if ready else "missing required runtime: " + ", ".join(missing),
                  project_creation_enabled=ready, ffmpeg=bool(ffmpeg), ffprobe=bool(ffprobe),
                  decoding=bool(ffprobe and cv), tracking=cv, encoding=bool(ffmpeg),
                  adaptive_ocr=True, resumable_jobs=True, export_container="mp4")


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _entry(
    capability_id: str,
    available: bool,
    *,
    ready: Optional[bool] = None,
    version: Optional[str] = None,
    reason: Optional[str] = None,
    **details: Any,
) -> Dict[str, Any]:
    return {
        "id": capability_id,
        "available": bool(available),
        "ready": bool(available if ready is None else ready),
        "version": version,
        "reason": reason,
        **details,
    }


def _gpu_status() -> Dict[str, Any]:
    installed = _module_available("torch")
    if not installed:
        return _entry(
            "cuda", False, ready=False, reason="PyTorch is not installed",
            device="cpu", device_count=0,
        )
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        return _entry(
            "cuda", available, ready=available, version=_version("torch"),
            reason=None if available else "CUDA is unavailable; CPU fallback is active",
            device="cuda" if available else "cpu", device_count=count,
            devices=names,
        )
    except Exception as exc:
        return _entry(
            "cuda", False, ready=False, version=_version("torch"),
            reason=f"GPU probe failed: {type(exc).__name__}", device="cpu",
            device_count=0,
        )


def _sam_status() -> Dict[str, Any]:
    installed = _module_available("segment_anything")
    checkpoint = os.environ.get("TOFU_SAM_CHECKPOINT")
    enabled = os.environ.get("TOFU_SAM_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    checkpoint_ready = bool(checkpoint and Path(checkpoint).is_file())
    if not enabled:
        state, reason = "disabled", "SAM is disabled; set TOFU_SAM_ENABLED=1 after provisioning a checkpoint"
    elif not installed:
        state, reason = "missing", "segment-anything is not installed"
    elif not checkpoint_ready:
        state, reason = "missing", "no readable checkpoint is configured in TOFU_SAM_CHECKPOINT"
    else:
        state, reason = "ready", None
    return _entry(
        "sam", installed and enabled, ready=installed and checkpoint_ready and enabled,
        version=_version("segment-anything"), reason=reason,
        configured=bool(checkpoint), checkpoint_ready=checkpoint_ready, state=state,
        lifecycle="on_demand",
    )


def _font_status(validator_factory: Callable[[], Any]) -> Dict[str, Any]:
    try:
        registry = validator_factory().font_registry
        count = len(registry.faces) if registry is not None else 0
        return _entry(
            "font_library", registry is not None, ready=count > 0,
            version=_version("fonttools"),
            reason=None if count else "no usable font faces were discovered",
            face_count=count,
        )
    except Exception as exc:
        return _entry(
            "font_library", False, ready=False, version=_version("fonttools"),
            reason=f"font discovery failed: {type(exc).__name__}", face_count=0,
        )


def _normalize_provider(
    provider: Dict[str, Any], *, available_key: str = "available",
) -> Dict[str, Any]:
    available = bool(provider.get(available_key, provider.get("active", False)))
    ready = bool(provider.get("promoted", available)) and available
    reason = provider.get("detail") or (None if ready else provider.get("purpose"))
    return _entry(
        str(provider.get("id", "unknown")), available, ready=ready,
        version=provider.get("revision"), reason=reason,
        **{
            key: value for key, value in provider.items()
            if key not in {
                "id", "available", "active", "ready", "promoted",
                "detail", "reason", "revision", "version",
            }
        },
    )


def build_capabilities(
    validator_factory: Callable[[], Any],
    *,
    app_version: str,
    inpaint_statuses: Iterable[Dict[str, Any]],
    semantic_statuses: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the complete, JSON-safe runtime contract."""
    from tofu.layers import cicerone, knead, scene
    from tofu.layers.language_models import (
        DiacriticRestorationProvider, get_language_provider, get_scoring_provider,
    )
    from tofu.utils import translate

    easy_available = _module_available("easyocr")
    paddle_available = bool(cicerone.PaddleOCRBackend.is_available())
    scene_backend = scene.get_backend()
    hb = _module_available("uharfbuzz")
    ft = _module_available("freetype")
    bidi = _module_available("bidi")
    reshaper = _module_available("arabic_reshaper")
    translator = translate.get_backend()
    mt_available = translator.name != "null"

    return {
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        "build": {
            "app": "ToFU",
            "version": app_version,
            "python": platform.python_version(),
            "platform": sys.platform,
        },
        "compute": {"gpu": _gpu_status()},
        "ocr": {
            "providers": [
                _entry(
                    "easyocr", easy_available, version=_version("easyocr"),
                    reason=None if easy_available else "EasyOCR is not installed",
                    role="primary",
                ),
                _entry(
                    "paddleocr", paddle_available,
                    version=_version("paddleocr") if paddle_available else None,
                    reason=None if paddle_available else "isolated PaddleOCR worker is unavailable",
                    role="independent_verifier",
                ),
            ],
            "language_models": [
                get_language_provider().status(),
                DiacriticRestorationProvider().status(),
                get_scoring_provider().status(),
            ],
        },
        "scene": {
            "active_backend": getattr(scene_backend, "name", type(scene_backend).__name__),
            "providers": [
                _entry(
                    "classical_cv", _module_available("cv2"), version=_version("opencv-python"),
                    reason=None if _module_available("cv2") else "OpenCV is not installed",
                ),
                _sam_status(),
            ],
        },
        "inpainting": {
            "providers": [_normalize_provider(item) for item in inpaint_statuses],
        },
        "shaping": {
            "advanced_available": bool(knead.available()),
            "providers": [
                _entry("harfbuzz", hb, version=_version("uharfbuzz"),
                       reason=None if hb else "uharfbuzz is not installed"),
                _entry("freetype", ft, version=_version("freetype-py"),
                       reason=None if ft else "freetype-py is not installed"),
                _entry("bidi", bidi, version=_version("python-bidi"),
                       reason=None if bidi else "python-bidi is not installed"),
                _entry("arabic_reshaper", reshaper, version=_version("arabic-reshaper"),
                       reason=None if reshaper else "arabic-reshaper is not installed"),
            ],
        },
        "semantics": {
            "providers": [
                _normalize_provider(item, available_key="active")
                for item in semantic_statuses
            ],
        },
        "translation": {
            "active_provider": translator.name,
            "providers": [
                _entry(
                    translator.name, mt_available, ready=mt_available,
                    reason=None if mt_available else "no machine-translation provider is configured",
                ),
            ],
        },
        "fonts": _font_status(validator_factory),
        "video": _video_status(),
    }
