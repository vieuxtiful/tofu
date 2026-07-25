## 🍢 vtm — Visual Translation Memory, reference implementation of VTM v1.0
## vieuxtiful
"""Read and write the Open Visual Translation Memory format.

A conventional translation memory stores ``(source_text, target_text)``. That
is enough to reuse a string and useless for reusing how the string LOOKED. A
visual TM has to carry where the text sat, how big it was, what face and
colour it used, and how confident the pipeline was about all of it -- so that
a second asset containing the same source string can be localized the same
way without a human re-deciding every typographic detail.

VTM v1.0 is that record, specified in ``spec/vtm-1.0.md`` with a machine
schema in ``spec/vtm-1.0.schema.json``. This module is the reference
implementation: it converts to and from ToFU's :class:`TextManifest` and
validates documents from anyone else.

Three design commitments the format makes, each of which is a decision a
consumer would otherwise have to guess:

**Geometry is meaningless without the asset it came from.** A bbox of
``(10, 20, 100, 30)`` describes a completely different region on a 640px
thumbnail than on a 4096px source. ``asset.width``/``asset.height`` are
therefore REQUIRED, and coordinates are defined in one place: pixels, origin
top-left, x rightward, y downward. Consumers that need normalized
coordinates divide; nobody has to reverse-engineer an origin.

**Languages are BCP 47.** ToFU's internal codes (``zh-cn``, ``ja``) are
convenient and non-standard. They are converted on the way out via the
existing :func:`tofu.utils.interchange._lang_to_full` table rather than a
second mapping that could drift from it.

**Vendor data lives behind ``x-`` keys.** ``tsume``, ``cleanse_strategy`` and
``garnish_profile`` are real and useful, and no competing implementation can
be expected to honour them. They round-trip under ``x-tofu`` so ToFU loses
nothing, while a conforming reader that ignores every ``x-`` key still gets a
complete, useful record.

Validation is hand-written rather than delegated to ``jsonschema`` so that
this stays a zero-dependency capability -- the published JSON Schema is for
third-party tooling, and :func:`validate` exists so ToFU can check documents
without asking anyone to install anything.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tofu.core.types import BBox, InstText, StyleProfil, TextManifest

#: Format version this module reads and writes.
VTM_VERSION = "1.0"

#: XML namespace for the XLIFF 1.2 binding (see spec/vtm-1.0.md §7).
VTM_NAMESPACE = "urn:vieuxtiful:vtm:1.0"

#: Keys a conforming document MUST carry at the top level.
REQUIRED_TOP_LEVEL = ("vtm_version", "source_language", "target_language", "asset", "entries")

#: Keys every entry MUST carry (conformance level "core").
REQUIRED_ENTRY = ("id", "source", "target", "geometry")

#: StyleProfil fields promoted into the portable `style` object. Everything
#: else on StyleProfil is ToFU-specific and goes to `x-tofu`.
_PORTABLE_STYLE = (
    "font_family", "font_size", "color", "italic", "underline",
    "tracking", "leading",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _bcp47(lang: Optional[str]) -> Optional[str]:
    """ToFU language code -> BCP 47, reusing interchange's existing table."""
    if not lang:
        return None
    from tofu.utils.interchange import _lang_to_full

    return _lang_to_full(lang)


def _style_to_vtm(style: Optional[StyleProfil]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Split a StyleProfil into (portable style, ToFU-only extension)."""
    if style is None:
        return None, None

    portable: Dict[str, Any] = {}
    for field in _PORTABLE_STYLE:
        value = getattr(style, field, None)
        if value is not None:
            portable[field] = value

    if style.font_weight is not None:
        portable["weight"] = style.font_weight
    if style.align_h is not None:
        portable["align"] = style.align_h
    if style.align_v is not None:
        portable["valign"] = style.align_v
    if style.target_orientation is not None:
        portable["orientation"] = style.target_orientation
    if style.word_order is not None:
        portable["direction"] = style.word_order
    if style.stroke_color is not None or style.stroke_width is not None:
        portable["stroke"] = {
            k: v for k, v in
            (("color", style.stroke_color), ("width", style.stroke_width))
            if v is not None
        }
    if style.shadow:
        portable["shadow"] = dict(style.shadow)

    # everything else on StyleProfil that is set, verbatim, for round-tripping
    vendor: Dict[str, Any] = {}
    handled = set(_PORTABLE_STYLE) | {
        "font_weight", "align_h", "align_v", "target_orientation",
        "word_order", "stroke_color", "stroke_width", "shadow",
    }
    for field, value in vars(style).items():
        if field not in handled and value is not None:
            vendor[field] = value

    return (portable or None), (vendor or None)


def _style_from_vtm(style: Optional[Dict], vendor: Optional[Dict]) -> Optional[StyleProfil]:
    """Rebuild a StyleProfil from a portable style plus ToFU's extension."""
    if not style and not vendor:
        return None
    kwargs: Dict[str, Any] = {}
    style = style or {}
    for field in _PORTABLE_STYLE:
        if field in style:
            kwargs[field] = style[field]
    if "weight" in style:
        kwargs["font_weight"] = style["weight"]
    if "align" in style:
        kwargs["align_h"] = style["align"]
    if "valign" in style:
        kwargs["align_v"] = style["valign"]
    if "orientation" in style:
        kwargs["target_orientation"] = style["orientation"]
    if "direction" in style:
        kwargs["word_order"] = style["direction"]
    stroke = style.get("stroke") or {}
    if "color" in stroke:
        kwargs["stroke_color"] = stroke["color"]
    if "width" in stroke:
        kwargs["stroke_width"] = stroke["width"]
    if "shadow" in style:
        kwargs["shadow"] = style["shadow"]
    # vendor keys are trusted only when they name real StyleProfil fields --
    # a foreign x-tofu block must never inject arbitrary attributes
    valid = set(vars(StyleProfil()).keys())
    for field, value in (vendor or {}).items():
        if field in valid:
            kwargs[field] = value
    return StyleProfil(**kwargs)


def export_vtm(
    manifest: TextManifest,
    src_lang: Optional[str] = None,
    targ_lang: Optional[str] = None,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    generator: str = "ToFU",
    include_untranslated: bool = False,
) -> Dict[str, Any]:
    """Serialize a manifest into a VTM v1.0 document.

    `image_size` overrides ``manifest.img_dim``; one of the two MUST resolve,
    because the format refuses to emit geometry it cannot anchor to a
    resolution (see the module docstring).

    Regions flagged ``dnt`` or ``excluded`` are omitted -- do-not-translate
    text is not a translation pair, and an excluded region was removed from
    the workspace by a human. Untranslated regions are omitted too unless
    `include_untranslated` is set, which is what you want when exporting a
    job FOR translation rather than a memory OF one.
    """
    size = image_size or manifest.img_dim
    if not size or len(size) != 2 or not all(size):
        raise ValueError(
            "VTM requires the asset's pixel dimensions: pass image_size=(w, h) "
            "or set manifest.img_dim. A bbox without a resolution cannot be "
            "interpreted by a consumer."
        )
    width, height = int(size[0]), int(size[1])

    doc_src = _bcp47(src_lang or manifest.src_lang)
    doc_tgt = _bcp47(targ_lang or manifest.targ_lang)

    entries: List[Dict[str, Any]] = []
    for inst in manifest.instances:
        if inst.dnt or inst.excluded:
            continue
        if not include_untranslated and not inst.target_text:
            continue
        entries.append(_entry_from_inst(inst, doc_src, doc_tgt))

    document: Dict[str, Any] = {
        "vtm_version": VTM_VERSION,
        "generator": {"name": generator, "version": _generator_version()},
        "created": _now_iso(),
        "source_language": doc_src,
        "target_language": doc_tgt,
        "asset": {"id": manifest.asset_id, "width": width, "height": height},
        "entries": entries,
    }
    return document


def _generator_version() -> str:
    from tofu import __version__

    return __version__


def _entry_from_inst(inst: InstText, doc_src: Optional[str], doc_tgt: Optional[str]) -> Dict[str, Any]:
    bbox = inst.adjusted_bbox or inst.bounding_box
    geometry: Dict[str, Any] = {
        "bbox": {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height},
    }
    if inst.segmentation_mask and inst.segmentation_mask.polygon:
        geometry["polygon"] = [[int(x), int(y)] for x, y in inst.segmentation_mask.polygon]
    if inst.reading_order is not None:
        geometry["reading_order"] = inst.reading_order

    entry: Dict[str, Any] = {
        "id": inst.id,
        "source": inst.text or "",
        "target": inst.target_text or "",
        "geometry": geometry,
    }

    # per-entry language only when it differs from the document default,
    # so the common case stays uncluttered
    entry_src = _bcp47(inst.language or inst.detected_language)
    if entry_src and entry_src != doc_src:
        entry["source_language"] = entry_src
    entry_tgt = _bcp47(inst.target_language)
    if entry_tgt and entry_tgt != doc_tgt:
        entry["target_language"] = entry_tgt

    style, vendor_style = _style_to_vtm(inst.style_profile)
    if style:
        entry["style"] = style

    if inst.background_profile is not None:
        background = {
            k: v for k, v in (
                ("color", inst.background_profile.dominant_color),
                ("texture", inst.background_profile.texture),
            ) if v is not None
        }
        if background:
            entry["background"] = background

    provenance = {
        k: v for k, v in (
            ("confidence", inst.confidence),
            ("glyph_fallback", inst.glyph_fallback),
        ) if v is not None
    }
    if provenance:
        entry["provenance"] = provenance

    vendor: Dict[str, Any] = {}
    if vendor_style:
        vendor["style"] = vendor_style
    if inst.background_profile and inst.background_profile.cleanse_strategy:
        vendor["cleanse_strategy"] = inst.background_profile.cleanse_strategy
    if vendor:
        entry["x-tofu"] = vendor

    return entry


def import_vtm(document: Dict[str, Any]) -> List[InstText]:
    """Rebuild instances from a VTM document.

    Returns :class:`InstText` objects rather than a whole manifest: a VTM
    document is a memory of translation pairs, not a description of an asset,
    so the caller decides what to attach them to.

    Raises ValueError when the document does not validate -- importing a
    malformed memory silently is how a TM gets poisoned.
    """
    problems = validate(document)
    if problems:
        raise ValueError("not a valid VTM document: " + "; ".join(problems[:5]))

    instances: List[InstText] = []
    for entry in document["entries"]:
        box = entry["geometry"]["bbox"]
        vendor = entry.get("x-tofu") or {}
        instances.append(InstText(
            id=entry["id"],
            bounding_box=BBox(
                x=int(box["x"]), y=int(box["y"]),
                width=int(box["width"]), height=int(box["height"]),
            ),
            text=entry.get("source") or None,
            target_text=entry.get("target") or None,
            language=entry.get("source_language") or document.get("source_language"),
            target_language=entry.get("target_language"),
            reading_order=entry["geometry"].get("reading_order"),
            confidence=(entry.get("provenance") or {}).get("confidence"),
            style_profile=_style_from_vtm(entry.get("style"), vendor.get("style")),
        ))
    return instances


def validate(document: Any) -> List[str]:
    """Check a document against VTM v1.0. Returns problems; empty means valid.

    Returns a LIST rather than raising on the first error, because a human
    fixing a hand-edited memory wants to see everything wrong with it at once.
    """
    problems: List[str] = []
    if not isinstance(document, dict):
        return ["document is not a JSON object"]

    for key in REQUIRED_TOP_LEVEL:
        if key not in document:
            problems.append(f"missing required top-level key {key!r}")

    version = document.get("vtm_version")
    if version is not None and not str(version).startswith("1."):
        problems.append(f"unsupported vtm_version {version!r} (this reader implements 1.x)")

    asset = document.get("asset")
    if isinstance(asset, dict):
        for dimension in ("width", "height"):
            value = asset.get(dimension)
            if not isinstance(value, int) or value <= 0:
                problems.append(f"asset.{dimension} must be a positive integer, got {value!r}")
    elif "asset" in document:
        problems.append("asset must be an object with width and height")

    entries = document.get("entries")
    if not isinstance(entries, list):
        if "entries" in document:
            problems.append("entries must be an array")
        return problems

    seen_ids = set()
    for index, entry in enumerate(entries):
        where = f"entries[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not an object")
            continue
        for key in REQUIRED_ENTRY:
            if key not in entry:
                problems.append(f"{where} missing required key {key!r}")

        entry_id = entry.get("id")
        if entry_id in seen_ids:
            problems.append(f"{where} duplicate id {entry_id!r}")
        seen_ids.add(entry_id)

        geometry = entry.get("geometry")
        if isinstance(geometry, dict):
            problems.extend(_validate_bbox(geometry.get("bbox"), asset, where))
        elif "geometry" in entry:
            problems.append(f"{where}.geometry must be an object")

    return problems


def _validate_bbox(bbox: Any, asset: Any, where: str) -> List[str]:
    """Geometry checks, including the containment one that actually bites.

    A box that runs off the asset is the classic symptom of coordinates
    recorded against a different resolution -- the exact failure the required
    asset dimensions exist to make detectable.
    """
    if not isinstance(bbox, dict):
        return [f"{where}.geometry.bbox must be an object with x, y, width, height"]

    problems = []
    for key in ("x", "y", "width", "height"):
        value = bbox.get(key)
        if not isinstance(value, (int, float)):
            problems.append(f"{where}.geometry.bbox.{key} must be a number, got {value!r}")
        elif key in ("width", "height") and value <= 0:
            problems.append(f"{where}.geometry.bbox.{key} must be positive, got {value!r}")
        elif key in ("x", "y") and value < 0:
            problems.append(f"{where}.geometry.bbox.{key} must be non-negative, got {value!r}")
    if problems or not isinstance(asset, dict):
        return problems

    aw, ah = asset.get("width"), asset.get("height")
    if isinstance(aw, int) and isinstance(ah, int):
        if bbox["x"] + bbox["width"] > aw or bbox["y"] + bbox["height"] > ah:
            problems.append(
                f"{where}.geometry.bbox extends past the asset "
                f"({bbox['x']}+{bbox['width']}x{bbox['y']}+{bbox['height']} vs {aw}x{ah}) "
                "-- coordinates were probably recorded at a different resolution"
            )
    return problems
