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
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

CAPABILITIES_SCHEMA_VERSION = "1.0"


# End-user display aliases for capability ids.  The `id` field is a stable
# machine contract (cache identity, provenance, routing-adjacent names) and
# must never be renamed just to look nicer in the UI; `label` is the
# presentation string the diagnostics panel shows instead of `id`.
#
# Each line below is annotated with the original raw label that used to be
# shown verbatim to end-users, so the mapping is auditable in one place.
# Groups not rendered by SystemCapabilitiesPanel (e.g. ocr.language_models)
# are intentionally absent here -- add them when they gain a UI surface.
ALIASES: dict[str, str] = {
    # Capture / OCR -- formerly shown as the library names "easyocr" / "paddleocr"
    "easyocr": "Primary OCR",
    "paddleocr": "Verifier OCR (CJK)",
    # Capture / Scene -- formerly "classical_cv" / "sam"
    "classical_cv": "Contour vision",
    "sam": "Segmentation model",
    # Render / Shaping -- formerly the raw python package names
    "harfbuzz": "Complex-script shaping",
    "freetype": "Font rasterizer",
    "bidi": "Bidirectional layout",
    "arabic_reshaper": "Arabic ligature shaping",
    # Render / Inpaint -- formerly the internal provider ids
    "analytic": "Analytic inpaint",
    "lama": "Neural background repair",
    "diffstr_experimental": "Diffusion inpaint (experimental)",
    "brushnet_experimental": "BrushNet inpaint (experimental)",
    "adobe_fill": "Cloud fill (disabled)",
    "manual": "Manual patch",
    # Translate / Semantics -- formerly the basil provider ids
    "deterministic_layout": "Deterministic layout",
    "stanza": "Syntactic parser",
    "nllb": "Neural aligner",
    "awesome_align": "Word aligner",
    "uploaded_glossary": "Uploaded glossary",
    # Translation -- formerly the translator backend name, e.g. "null"
    "null": "None configured",
}


def _video_status() -> dict[str, Any]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    cv = _module_available("cv2")
    ready = bool(ffmpeg and ffprobe and cv)
    missing = [
        name
        for name, value in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe), ("OpenCV", cv))
        if not value
    ]
    return _entry(
        "video_pipeline",
        ready,
        ready=ready,
        reason=None if ready else "missing required runtime: " + ", ".join(missing),
        project_creation_enabled=ready,
        ffmpeg=bool(ffmpeg),
        ffprobe=bool(ffprobe),
        decoding=bool(ffprobe and cv),
        tracking=cv,
        encoding=bool(ffmpeg),
        adaptive_ocr=True,
        resumable_jobs=True,
        export_container="mp4",
    )


def _version(distribution: str) -> str | None:
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
    ready: bool | None = None,
    version: str | None = None,
    reason: str | None = None,
    label: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    # `label` is the end-user display string; `id` stays the stable contract.
    # An explicit label wins, then the ALIASES map, then we fall back to id so
    # the panel never shows a blank where a name should be.
    resolved_label = label or ALIASES.get(capability_id) or capability_id
    return {
        "id": capability_id,
        "label": resolved_label,
        "available": bool(available),
        "ready": bool(available if ready is None else ready),
        "version": version,
        "reason": reason,
        **details,
    }


def _gpu_status() -> dict[str, Any]:
    installed = _module_available("torch")
    if not installed:
        return _entry(
            "cuda",
            False,
            ready=False,
            reason="PyTorch is not installed",
            device="cpu",
            device_count=0,
        )
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        return _entry(
            "cuda",
            available,
            ready=available,
            version=_version("torch"),
            reason=None if available else "CUDA is unavailable; CPU fallback is active",
            device="cuda" if available else "cpu",
            device_count=count,
            devices=names,
        )
    except Exception as exc:
        return _entry(
            "cuda",
            False,
            ready=False,
            version=_version("torch"),
            reason=f"GPU probe failed: {type(exc).__name__}",
            device="cpu",
            device_count=0,
        )


def _sam_status() -> dict[str, Any]:
    installed = _module_available("segment_anything")
    checkpoint = os.environ.get("TOFU_SAM_CHECKPOINT")
    enabled = os.environ.get("TOFU_SAM_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    checkpoint_ready = bool(checkpoint and Path(checkpoint).is_file())
    if not enabled:
        state, reason = (
            "disabled",
            "SAM is disabled; set TOFU_SAM_ENABLED=1 after provisioning a checkpoint",
        )
    elif not installed:
        state, reason = "missing", "segment-anything is not installed"
    elif not checkpoint_ready:
        state, reason = "missing", "no readable checkpoint is configured in TOFU_SAM_CHECKPOINT"
    else:
        state, reason = "ready", None
    return _entry(
        "sam",
        installed and enabled,
        ready=installed and checkpoint_ready and enabled,
        version=_version("segment-anything"),
        reason=reason,
        configured=bool(checkpoint),
        checkpoint_ready=checkpoint_ready,
        state=state,
        lifecycle="on_demand",
    )


def _font_status(validator_factory: Callable[[], Any]) -> dict[str, Any]:
    try:
        registry = validator_factory().font_registry
        count = len(registry.faces) if registry is not None else 0
        return _entry(
            "font_library",
            registry is not None,
            ready=count > 0,
            version=_version("fonttools"),
            reason=None if count else "no usable font faces were discovered",
            face_count=count,
        )
    except Exception as exc:
        return _entry(
            "font_library",
            False,
            ready=False,
            version=_version("fonttools"),
            reason=f"font discovery failed: {type(exc).__name__}",
            face_count=0,
        )


def _normalize_provider(
    provider: dict[str, Any],
    *,
    available_key: str = "available",
) -> dict[str, Any]:
    available = bool(provider.get(available_key, provider.get("active", False)))
    ready = bool(provider.get("promoted", available)) and available
    reason = provider.get("detail") or (None if ready else provider.get("purpose"))
    provider_id = str(provider.get("id", "unknown"))
    # A provider may carry its own `label`; otherwise resolve from ALIASES.
    label = provider.get("label") or ALIASES.get(provider_id)
    return _entry(
        provider_id,
        available,
        ready=ready,
        version=provider.get("revision"),
        reason=reason,
        label=label,
        **{
            key: value
            for key, value in provider.items()
            if key
            not in {
                "id",
                "available",
                "active",
                "ready",
                "promoted",
                "detail",
                "reason",
                "revision",
                "version",
                "label",
            }
        },
    )


def build_capabilities(
    validator_factory: Callable[[], Any],
    *,
    app_version: str,
    inpaint_statuses: Iterable[dict[str, Any]],
    semantic_statuses: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return the complete, JSON-safe runtime contract."""
    from tofu.layers import cicerone, knead, scene
    from tofu.layers.language_models import (
        DiacriticRestorationProvider,
        get_language_provider,
        get_scoring_provider,
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
                    "easyocr",
                    easy_available,
                    version=_version("easyocr"),
                    reason=None if easy_available else "EasyOCR is not installed",
                    role="primary",
                ),
                _entry(
                    "paddleocr",
                    paddle_available,
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
                    "classical_cv",
                    _module_available("cv2"),
                    version=_version("opencv-python"),
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
                _entry(
                    "harfbuzz",
                    hb,
                    version=_version("uharfbuzz"),
                    reason=None if hb else "uharfbuzz is not installed",
                ),
                _entry(
                    "freetype",
                    ft,
                    version=_version("freetype-py"),
                    reason=None if ft else "freetype-py is not installed",
                ),
                _entry(
                    "bidi",
                    bidi,
                    version=_version("python-bidi"),
                    reason=None if bidi else "python-bidi is not installed",
                ),
                _entry(
                    "arabic_reshaper",
                    reshaper,
                    version=_version("arabic-reshaper"),
                    reason=None if reshaper else "arabic-reshaper is not installed",
                ),
            ],
        },
        "semantics": {
            "providers": [
                _normalize_provider(item, available_key="active") for item in semantic_statuses
            ],
        },
        "translation": {
            "active_provider": translator.name,
            "providers": [
                _entry(
                    translator.name,
                    mt_available,
                    ready=mt_available,
                    reason=None
                    if mt_available
                    else "no machine-translation provider is configured",
                ),
            ],
        },
        "fonts": _font_status(validator_factory),
        "video": _video_status(),
    }
