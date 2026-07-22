"""Evidence-gated visual font identification and licensing-aware advice.

Typography's weight/slant estimator answers *what kind* of face is present.
This layer answers the different question, *which available face is closest to
the actual glyph outlines?*  It is deliberately conservative: a local visual
match becomes a recommendation, never an implicit replacement for a user's
font selection.  Optional commercial-catalog lookup is separately gated so a
source crop is never sent to a third party without an explicit editor action.

The local course is a lightweight, deterministic analogue of a visual-font
retrieval index: render the recognised source string in known installed faces,
normalise at measured cap height, then combine silhouette Dice, symmetric
Chamfer, projection-profile, and natural-width evidence.  It is robust to
colour/background because Scene's text mask supplies the observed glyph
silhouette.  A future DeepFont-style embedding can be added as another ranked
provider without changing the persisted evidence contract.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tofu.core.types import InstText, TextManifest
from tofu.utils.imaging import text_mask


MAX_LOCAL_FACES = 72
TOP_CANDIDATES = 5
MIN_GLYPH_PIXELS = 45
EXACT_CONFIDENCE = 0.82
EXACT_MARGIN = 0.075


def _split_face(path: str) -> Tuple[str, int]:
    base, marker, index = path.rpartition("#")
    if marker and index.isdigit():
        return base, int(index)
    return path, 0


def _load_font(path: str, size: int):
    from PIL import ImageFont

    base, index = _split_face(path)
    return ImageFont.truetype(base, size, index=index)


def _tight(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _render_mask(path: str, text: str, height: int):
    """Render text at a stable high resolution and return its tight mask."""
    from PIL import Image, ImageDraw
    import numpy as np

    # Rendering large then reducing to the measured source height gives all
    # candidate faces the same antialiasing budget and avoids DPI-dependent
    # font rasterisation artifacts dominating the comparison.
    font = _load_font(path, max(48, int(height * 5)))
    probe = Image.new("L", (8, 8), 0)
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font, stroke_width=0)
    w = max(1, box[2] - box[0])
    h = max(1, box[3] - box[1])
    canvas = Image.new("L", (w + 16, h + 16), 0)
    ImageDraw.Draw(canvas).text((8 - box[0], 8 - box[1]), text, font=font, fill=255)
    mask = _tight(np.asarray(canvas) > 96)
    if mask is None:
        return None
    import cv2
    scale = height / max(1, mask.shape[0])
    width = max(1, int(round(mask.shape[1] * scale)))
    return cv2.resize(mask.astype("uint8"), (width, height), interpolation=cv2.INTER_AREA) > 0


def _aligned_masks(source, candidate):
    """Center source/candidate in one envelope without distorting width."""
    import numpy as np

    height = max(source.shape[0], candidate.shape[0])
    width = max(source.shape[1], candidate.shape[1]) + 8
    a = np.zeros((height + 8, width), dtype=bool)
    b = np.zeros_like(a)
    ay = (a.shape[0] - source.shape[0]) // 2
    ax = (a.shape[1] - source.shape[1]) // 2
    by = (b.shape[0] - candidate.shape[0]) // 2
    bx = (b.shape[1] - candidate.shape[1]) // 2
    a[ay:ay + source.shape[0], ax:ax + source.shape[1]] = source
    b[by:by + candidate.shape[0], bx:bx + candidate.shape[1]] = candidate
    return a, b


def _visual_score(source, candidate) -> Tuple[float, Dict[str, float]]:
    """Return an interpretable [0, 1] glyph-shape score."""
    import cv2
    import numpy as np

    a, b = _aligned_masks(source, candidate)
    intersection = float(np.logical_and(a, b).sum())
    dice = (2 * intersection) / max(1.0, float(a.sum() + b.sum()))

    # Chamfer is forgiving of the one- or two-pixel boundary movement caused
    # by a photographed sign, while strongly penalising a different R bowl,
    # leg, aperture, or terminal shape.
    da = cv2.distanceTransform((~a).astype("uint8"), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~b).astype("uint8"), cv2.DIST_L2, 3)
    chamfer = (float(da[b].mean()) if b.any() else 99.0) + (float(db[a].mean()) if a.any() else 99.0)
    chamfer /= max(1.0, math.hypot(*a.shape))
    chamfer_score = max(0.0, 1.0 - chamfer * 2.6)

    def projection_similarity(x, y) -> float:
        px, py = x.sum(axis=0).astype(float), y.sum(axis=0).astype(float)
        if px.std() < 1e-6 or py.std() < 1e-6:
            return 0.0
        return max(0.0, min(1.0, (float(np.corrcoef(px, py)[0, 1]) + 1.0) / 2.0))

    projection = (projection_similarity(a, b) + projection_similarity(a.T, b.T)) / 2.0
    aspect = 1.0 - min(1.0, abs(source.shape[1] / max(1, source.shape[0]) - candidate.shape[1] / max(1, candidate.shape[0])) / 1.25)
    score = 0.50 * dice + 0.30 * chamfer_score + 0.12 * projection + 0.08 * aspect
    return round(float(score), 4), {
        "dice": round(float(dice), 4),
        "chamfer": round(float(chamfer_score), 4),
        "projection": round(float(projection), 4),
        "aspect": round(float(aspect), 4),
    }


def _weight_target(inst: InstText) -> Optional[int]:
    value = (inst.style_profile.font_weight if inst.style_profile else None) or ""
    value = str(value).lower()
    if any(x in value for x in ("heavy", "black", "ultra")):
        return 900
    if "bold" in value:
        return 700
    if "semi" in value or "demi" in value:
        return 600
    if "medium" in value:
        return 500
    if "light" in value:
        return 300
    if value in {"regular", "normal", "book"}:
        return 400
    return None


def _eligible_faces(registry, text: str, inst: InstText) -> Iterable[Any]:
    """Narrow an installed library without treating coverage as visual rank."""
    target = _weight_target(inst)
    faces = []
    for face in getattr(registry, "_fonts", {}).values():
        if not all(ord(ch) in face.codepoints for ch in text if not ch.isspace()):
            continue
        weight_distance = abs(int(getattr(face, "weight_class", 400) or 400) - target) if target else 0
        italic = "italic" in (face.subfamily or "").lower() or "oblique" in (face.subfamily or "").lower()
        expected_italic = bool(inst.style_profile.italic) if inst.style_profile else False
        style_penalty = 120 if italic != expected_italic else 0
        faces.append((weight_distance + style_penalty, face.family.lower(), face))
    faces.sort(key=lambda item: (item[0], item[1], item[2].font_path.lower()))
    # Do not compare every face of a 300-font family collection.  The best
    # weight/style face for a family is the meaningful first-pass candidate.
    chosen, seen = [], set()
    for _, family, face in faces:
        if family in seen:
            continue
        chosen.append(face)
        seen.add(family)
        if len(chosen) >= MAX_LOCAL_FACES:
            break
    return chosen


def local_match(img: Any, inst: InstText, registry) -> Optional[Dict[str, Any]]:
    """Rank installed faces against a source instance's glyph silhouette."""
    if not registry or not inst.text or len(inst.text.strip()) < 2:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        source = text_mask(img, inst.bounding_box)
        if source is None:
            return None
        source = _tight(np.asarray(source, dtype=bool))
    except Exception:
        return None
    if source is None or int(source.sum()) < MIN_GLYPH_PIXELS or source.shape[0] < 8:
        return None

    ranked: List[Dict[str, Any]] = []
    for face in _eligible_faces(registry, inst.text, inst):
        try:
            candidate = _render_mask(face.font_path, inst.text, source.shape[0])
            if candidate is None:
                continue
            score, evidence = _visual_score(source, candidate)
        except Exception:
            continue
        ranked.append({
            "family": face.family,
            "subfamily": face.subfamily,
            "font_path": face.font_path,
            "license": "installed",
            "available": True,
            "score": score,
            "evidence": evidence,
        })
    ranked.sort(key=lambda candidate: (-candidate["score"], candidate["family"].lower()))
    ranked = ranked[:TOP_CANDIDATES]
    if not ranked:
        return None
    top = ranked[0]
    margin = top["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)
    confidence = max(0.0, min(1.0, 0.62 * top["score"] + 0.38 * min(1.0, margin / .16)))
    return {
        "schema": 1,
        "status": "matched" if confidence >= EXACT_CONFIDENCE and margin >= EXACT_MARGIN else "review",
        "provider": "local_glyph_retrieval",
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "source_text": inst.text,
        "candidates": ranked,
        "recommended_substitute": {
            "font_path": top["font_path"], "family": top["family"],
            "subfamily": top["subfamily"], "score": top["score"],
        },
    }


def _looks_like_french_enamel_sign(img, manifest: TextManifest) -> bool:
    """Return a *style context*, never a geographic/font identification.

    Blue enamel plaques occur across many French municipalities.  They are
    not evidence that the sign is Parisian or that it used any particular
    foundry face, so callers may present only a clearly-labeled research
    reference and must not promote it as a detected font.
    """
    words = " ".join((inst.text or "").lower() for inst in manifest.instances)
    if not any(token in words.split() for token in ("rue", "avenue", "boulevard", "place", "quai")):
        return False
    try:
        import numpy as np
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return False
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        blue = ((b > r * 1.18) & (b > g * 1.08) & (b > 55)).mean()
        red = ((r > g * 1.3) & (r > b * .9) & (r > 85)).mean()
        return bool(blue > .08 and red > .002)
    except Exception:
        return False


def _french_enamel_reference() -> Dict[str, Any]:
    return {
        "family": "Plaak",
        "subfamily": None,
        "font_path": None,
        "license": "commercial",
        "available": False,
        "score": None,
        "source": "contextual_style_reference",
        "url": "https://www.205.tf/Plaak",
        "foundry": "205TF",
        "reason": "This is a licensed French street-sign lettering reference, not a detected match. Use catalog retrieval or human review before licensing.",
    }


def identify_manifest_fonts(asset: Any, manifest: TextManifest, registry) -> int:
    """Attach local glyph-retrieval evidence to every eligible instance."""
    try:
        from PIL import Image
        import numpy as np
        img = np.asarray(Image.open(asset).convert("RGB")) if not hasattr(asset, "shape") else asset
    except Exception:
        return 0
    enamel_context = _looks_like_french_enamel_sign(img, manifest)
    updated = 0
    for inst in manifest.instances:
        result = local_match(img, inst, registry)
        if result is None:
            continue
        if enamel_context:
            result["candidates"] = [_french_enamel_reference(), *result["candidates"]]
            result["contextual_candidates"] = ["Plaak"]
        inst.font_match = result
        updated += 1
    return updated


def external_catalog_match(asset: Any, manifest: TextManifest, registry) -> int:
    """Explicit WhatFontIs adapter for commercial/free catalog recognition.

    This function is intentionally opt-in.  It sends only an individual text
    crop and only after the UI receives an explicit consent action.  The API
    key remains server-side in ``TOFU_WHATFONTIS_API_KEY``; without it this
    records an actionable unavailable-provider state and performs no network
    request.
    """
    key = os.environ.get("TOFU_WHATFONTIS_API_KEY", "").strip()
    if not key:
        for inst in manifest.instances:
            if inst.font_match:
                inst.font_match["external_provider"] = {
                    "name": "WhatFontIs", "enabled": False,
                    "reason": "Set TOFU_WHATFONTIS_API_KEY to enable commercial-catalog matching.",
                }
        return 0
    try:
        from PIL import Image
        image = Image.open(asset).convert("RGB") if not hasattr(asset, "crop") else asset.convert("RGB")
    except Exception:
        return 0
    matched = 0
    for inst in manifest.instances:
        if not inst.text or not inst.font_match:
            continue
        box = inst.bounding_box
        crop = image.crop((max(0, box.x), max(0, box.y), max(0, box.x + box.width), max(0, box.y + box.height)))
        data = io.BytesIO()
        crop.save(data, format="PNG")
        payload = urllib.parse.urlencode({
            "API_KEY": key,
            "IMAGEBASE64": 1,
            "urlimagebase64": base64.b64encode(data.getvalue()).decode("ascii"),
            "NOTTEXTBOXSDETECTION": 0,
            "FREEFONTS": 0,
            "limit": TOP_CANDIDATES,
        }).encode("utf-8")
        try:
            request = urllib.request.Request("https://www.whatfontis.com/api2/", data=payload,
                                             headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
            with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310: fixed HTTPS endpoint
                raw = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            inst.font_match["external_provider"] = {"name": "WhatFontIs", "enabled": True, "error": type(exc).__name__}
            continue
        candidates = []
        for item in raw if isinstance(raw, list) else []:
            candidates.append({
                "family": item.get("title", "Unknown"), "subfamily": None,
                "font_path": None,
                "license": "commercial" if str(item.get("type", "")).lower() == "commercial" else "free",
                "available": False,
                "score": None,
                "source": "whatfontis",
                "url": item.get("url"), "preview_url": item.get("image"),
                "foundry": item.get("site"),
            })
        if candidates:
            inst.font_match["external_candidates"] = candidates
            inst.font_match["external_provider"] = {"name": "WhatFontIs", "enabled": True}
            matched += 1
    return matched
