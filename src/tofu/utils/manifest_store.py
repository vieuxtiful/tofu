## 🍢 manifest_store — persistence for TextManifest
## vieuxtiful
"""
JSON-based persistence for TextManifest objects so the frontend can
auto-save and restore the interactive preview state across sessions.

Each manifest is stored as uploads/{asset_id}.manifest.json.
"""

import json
from pathlib import Path
from typing import Optional

from tofu.core.types import (
    TextManifest, InstText, BBox, Mask, AssetType,
    StyleProfil, BgProfil, CharactText, SceneRegion,
)


def _manifest_path(store_dir: Path, asset_id: str) -> Path:
    return store_dir / f"{asset_id}.manifest.json"


def save_manifest(store_dir: Path, asset_id: str, manifest: TextManifest) -> None:
    """serialize a TextManifest to JSON on disk."""
    store_dir.mkdir(parents=True, exist_ok=True)
    data = _manifest_to_dict(manifest)
    _manifest_path(store_dir, asset_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_manifest(store_dir: Path, asset_id: str) -> Optional[TextManifest]:
    """deserialize a TextManifest from disk, or None if not found."""
    p = _manifest_path(store_dir, asset_id)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    return _dict_to_manifest(data)


def _manifest_to_dict(m: TextManifest) -> dict:
    return {
        "asset_id": m.asset_id,
        "total_regions": m.total_regions,
        "src_lang": m.src_lang,
        "targ_lang": m.targ_lang,
        "img_dim": list(m.img_dim) if m.img_dim else None,
        "scene_regions": [_region_to_dict(r) for r in (m.scene_regions or [])],
        "asset_type": m.asset_type.value if hasattr(m.asset_type, "value") else str(m.asset_type),
        "frame_count": m.frame_count,
        "fps": m.fps,
        "duration": m.duration,
        "prcssng_time": m.prcssng_time,
        "instances": [_inst_to_dict(i) for i in m.instances],
    }


def _region_to_dict(r: SceneRegion) -> dict:
    return {
        "bbox": {"x": r.bbox.x, "y": r.bbox.y,
                 "width": r.bbox.width, "height": r.bbox.height},
        "semantic_label": r.semantic_label,
        "confidence": r.confidence,
        "background_color": r.background_color,
        "border_detected": r.border_detected,
        "polygon": r.polygon,
        "texture": r.texture,
        "material": r.material,
    }


def _dict_to_region(rdict: dict) -> SceneRegion:
    b = rdict["bbox"]
    return SceneRegion(
        bbox=BBox(x=b["x"], y=b["y"], width=b["width"], height=b["height"]),
        semantic_label=rdict.get("semantic_label", "surface"),
        confidence=rdict.get("confidence", 0.0),
        background_color=rdict.get("background_color"),
        border_detected=rdict.get("border_detected", False),
        polygon=(
            [tuple(p) for p in rdict["polygon"]]
            if rdict.get("polygon") else None
        ),
        texture=rdict.get("texture"),
        material=rdict.get("material"),
    )


def _inst_to_dict(inst: InstText) -> dict:
    d = {
        "id": inst.id,
        "bounding_box": {
            "x": inst.bounding_box.x,
            "y": inst.bounding_box.y,
            "width": inst.bounding_box.width,
            "height": inst.bounding_box.height,
        },
        "text": inst.text,
        "target_text": inst.target_text,
        "confidence": inst.confidence,
        "detected_language": inst.detected_language,
        "reading_order": inst.reading_order,
        "dnt": inst.dnt,
        "excluded": inst.excluded,
        "target_language": inst.target_language,
        "glyph_fallback": inst.glyph_fallback,
        "tm_suggestion": inst.tm_suggestion,
        "ocr_correction": inst.ocr_correction,
        "recognition_history": inst.recognition_history,
        "repair_provenance": inst.repair_provenance,
        "resolved_font_family": inst.resolved_font_family,
    }
    if inst.language is not None:
        d["language"] = inst.language
    if inst.segmentation_mask:
        d["segmentation_mask"] = {
            "polygon": inst.segmentation_mask.polygon,
            "confidence": inst.segmentation_mask.confidence,
        }
        if inst.segmentation_mask.holes:
            d["segmentation_mask"]["holes"] = inst.segmentation_mask.holes
    if inst.style_profile:
        s = inst.style_profile
        d["style_profile"] = {
            "font_family": s.font_family,
            "font_weight": s.font_weight,
            "color": s.color,
            "shadow": s.shadow,
            "effects": s.effects,
            "font_size": s.font_size,
            "italic": s.italic,
            "underline": s.underline,
            "underline_offset": s.underline_offset,
            "underline_width": s.underline_width,
            "subscript": s.subscript,
            "superscript": s.superscript,
            "align_h": s.align_h,
            "align_v": s.align_v,
            "justification": s.justification,
            "indent": s.indent,
            "tracking": s.tracking,
            "kerning": s.kerning,
            "leading": s.leading,
            "baseline_shift": s.baseline_shift,
            "tab_width": s.tab_width,
            "tsume": s.tsume,
            "stroke_color": s.stroke_color,
            "stroke_width": s.stroke_width,
            "target_orientation": s.target_orientation,
            "word_order": s.word_order,
            "transform": s.transform,
        }
    if inst.background_profile:
        d["background_profile"] = {
            "semantic_label": inst.background_profile.semantic_label,
            "texture": inst.background_profile.texture,
            "material": inst.background_profile.material,
            "gradients": inst.background_profile.gradients,
            "patterns": inst.background_profile.patterns,
            "dominant_color": inst.background_profile.dominant_color,
            "surface_texture": inst.background_profile.surface_texture,
            "cleanse_strategy": inst.background_profile.cleanse_strategy,
        }
    if inst.characteristics:
        d["characteristics"] = {
            "font_style": inst.characteristics.font_style,
            "color": inst.characteristics.color,
            "size": inst.characteristics.size,
            "positioning": inst.characteristics.positioning,
            "effects": inst.characteristics.effects,
        }
    return d


def _dict_to_manifest(data: dict) -> TextManifest:
    instances = []
    for idict in data.get("instances", []):
        bbox = BBox(
            x=idict["bounding_box"]["x"],
            y=idict["bounding_box"]["y"],
            width=idict["bounding_box"]["width"],
            height=idict["bounding_box"]["height"],
        )
        mask = None
        if idict.get("segmentation_mask"):
            mdict = idict["segmentation_mask"]
            mask = Mask(
                polygon=[tuple(p) for p in mdict["polygon"]],
                confidence=mdict["confidence"],
                holes=(
                    [[tuple(p) for p in hole] for hole in mdict["holes"]]
                    if mdict.get("holes") else None
                ),
            )
        style = None
        if idict.get("style_profile"):
            sdict = idict["style_profile"]
            style = StyleProfil(
                font_family=sdict.get("font_family"),
                font_weight=sdict.get("font_weight"),
                color=sdict.get("color"),
                shadow=sdict.get("shadow"),
                effects=sdict.get("effects"),
                font_size=sdict.get("font_size"),
                italic=sdict.get("italic"),
                underline=sdict.get("underline"),
                underline_offset=sdict.get("underline_offset"),
                underline_width=sdict.get("underline_width"),
                subscript=sdict.get("subscript"),
                superscript=sdict.get("superscript"),
                align_h=sdict.get("align_h"),
                align_v=sdict.get("align_v"),
                justification=sdict.get("justification"),
                indent=sdict.get("indent"),
                tracking=sdict.get("tracking"),
                kerning=sdict.get("kerning"),
                leading=sdict.get("leading"),
                baseline_shift=sdict.get("baseline_shift"),
                tab_width=sdict.get("tab_width"),
                tsume=sdict.get("tsume"),
                stroke_color=sdict.get("stroke_color"),
                stroke_width=sdict.get("stroke_width"),
                target_orientation=sdict.get("target_orientation"),
                word_order=sdict.get("word_order"),
                transform=sdict.get("transform"),
            )
        bg = None
        if idict.get("background_profile"):
            gdict = idict["background_profile"]
            bg = BgProfil(
                semantic_label=gdict.get("semantic_label"),
                texture=gdict.get("texture"),
                material=gdict.get("material"),
                gradients=gdict.get("gradients"),
                patterns=gdict.get("patterns"),
                dominant_color=gdict.get("dominant_color"),
                surface_texture=gdict.get("surface_texture"),
                cleanse_strategy=gdict.get("cleanse_strategy"),
            )
        chars = None
        if idict.get("characteristics"):
            cdict = idict["characteristics"]
            chars = CharactText(
                font_style=cdict.get("font_style"),
                color=cdict.get("color"),
                size=cdict.get("size"),
                positioning=cdict.get("positioning"),
                effects=cdict.get("effects"),
            )
        instances.append(InstText(
            id=idict["id"],
            bounding_box=bbox,
            segmentation_mask=mask,
            text=idict.get("text"),
            target_text=idict.get("target_text"),
            language=idict.get("language"),
            confidence=idict.get("confidence"),
            detected_language=idict.get("detected_language"),
            reading_order=idict.get("reading_order"),
            dnt=idict.get("dnt", False),
            excluded=idict.get("excluded", False),
            target_language=idict.get("target_language"),
            glyph_fallback=idict.get("glyph_fallback"),
            tm_suggestion=idict.get("tm_suggestion"),
            ocr_correction=idict.get("ocr_correction"),
            recognition_history=idict.get("recognition_history"),
            repair_provenance=idict.get("repair_provenance"),
            resolved_font_family=idict.get("resolved_font_family"),
            style_profile=style,
            background_profile=bg,
            characteristics=chars,
        ))
    atype = AssetType(data.get("asset_type", "image"))
    return TextManifest(
        asset_id=data["asset_id"],
        total_regions=len(instances),
        instances=instances,
        src_lang=data.get("src_lang"),
        targ_lang=data.get("targ_lang"),
        img_dim=tuple(data["img_dim"]) if data.get("img_dim") else None,
        scene_regions=[_dict_to_region(r) for r in data.get("scene_regions", [])],
        asset_type=atype,
        frame_count=data.get("frame_count", 1),
        fps=data.get("fps"),
        duration=data.get("duration"),
        prcssng_time=data.get("prcssng_time"),
    )
