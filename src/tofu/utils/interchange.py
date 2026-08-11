## 🍢 interchange — XLIFF/TMX/TSV/CSV/TXT export and import
## vieuxtiful
"""
Translation file format interchange for TMS/CAT interoperability.

Export formats:
  - XLIFF 1.2 (standard, SDL, Crowdin, Smartling variants)
  - TMX (Translation Memory eXchange)
  - TSV (tab-separated values)
  - CSV (comma-separated values)
  - TXT (plain text segment list)

Import formats:
  - XLIFF 1.2 (any variant — parsed by trans-unit ID)
  - TMX
  - TSV
  - CSV
  - TXT

All formats use the manifest's region IDs (r1, r2, ...) as the
immutable segment identifier. Round-trip integrity is enforced:
import validates that every ID in the file matches a manifest region.
"""

import csv
import io
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

from tofu.core.types import TextManifest


def _version() -> str:
    """The package version, for the tool-provenance headers below.

    Read from tofu.__version__ rather than written out here. These two
    headers used to carry literals, and they had already drifted into
    disagreeing with each other -- XLIFF announced 0.2 while TMX, forty
    lines away in the same file, announced 0.1. Provenance that a CAT tool
    records against a translation unit is exactly the wrong place for a
    number nobody remembers to update.

    Imported lazily: tofu/__init__.py re-exports from core.types, and
    core.types is what imports this module's sibling -- a module-level
    import here would close that circle.
    """
    from tofu import __version__

    return __version__


# ---------------------------------------------------------------------------
# XLIFF 1.2 export
# ---------------------------------------------------------------------------

def export_xliff(
    manifest: TextManifest,
    src_lang: str = "en",
    targ_lang: str = "",
    variant: str = "standard",
) -> str:
    """Generate XLIFF 1.2 XML from a TextManifest.

    TMS-import guarantees (Trados, Crowdin, Smartling, MemoQ):
      - target-language is ALWAYS present on <file> (Trados/MemoQ
        reject files without it)
      - every non-DNT unit carries a <target> element — empty targets
        get state="needs-translation", filled ones state="translated" —
        so the file drops into an empty TMS project and comes back with
        the same unit IDs for round-trip import
      - a <header><tool/></header> identifies the producer

    Variants:
      - standard: base XLIFF 1.2 (also correct for MemoQ)
      - memoq: alias of standard
      - sdl: adds SDL Trados namespace extensions
      - crowdin: adds context attributes
      - smartling: adds Smartling namespace
    """
    src_lang_full = _lang_to_full(src_lang)
    # target-language must never be empty: fall back to source so the
    # attribute is always valid for strict TMS importers
    targ_lang_full = _lang_to_full(targ_lang) if targ_lang else src_lang_full

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2"',
    ]
    if variant == "sdl":
        lines.append('  xmlns:sdl="http://sdl.com/FileFormats/SdlXliff/1.0"')
    elif variant == "crowdin":
        lines.append('  xmlns:cr="http://crowdin.com/ns/xliff"')
    elif variant == "smartling":
        lines.append('  xmlns:sl="http://smartling.com/ns/xliff"')
    lines.append('>')
    lines.append(f'  <file source-language="{src_lang_full}" '
                 f'target-language="{targ_lang_full}" '
                 f'datatype="plaintext" original="{manifest.asset_id}">')
    lines.append('    <header>')
    lines.append(f'      <tool tool-id="tofu" tool-name="ToFU" tool-version="{_version()}" />')
    lines.append('    </header>')
    lines.append('    <body>')

    # Bounding-box detection and translation order are deliberately separate:
    # `reading_order` is the user-managed Text Manifest order.  Preserve the
    # original list sequence as a stable fallback for older manifests.
    ordered_instances = sorted(
        enumerate(manifest.instances),
        key=lambda pair: (
            pair[1].reading_order is None,
            pair[1].reading_order if pair[1].reading_order is not None else pair[0],
            pair[0],
        ),
    )
    for _, inst in ordered_instances:
        if getattr(inst, "dnt", False):
            continue
        source = inst.text or ""
        target = inst.target_text or ""
        # per-region language overrides beat the manifest-level defaults
        unit_src = _lang_to_full(inst.language) if getattr(inst, "language", None) else src_lang_full
        unit_targ = (
            _lang_to_full(inst.target_language)
            if getattr(inst, "target_language", None) else targ_lang_full
        )
        state = "translated" if target else "needs-translation"
        unit_attrs = f'id="{inst.id}"'
        if variant == "crowdin":
            unit_attrs += f' cr:context="bbox:{inst.bounding_box.x},{inst.bounding_box.y}"'

        lines.append(f'      <trans-unit {unit_attrs}>')
        lines.append(f'        <source xml:lang="{unit_src}">{xml_escape(source)}</source>')
        lines.append(f'        <target xml:lang="{unit_targ}" state="{state}">{xml_escape(target)}</target>')
        lines.append(f'        <note>bbox: {inst.bounding_box.x},{inst.bounding_box.y},'
                     f'{inst.bounding_box.width},{inst.bounding_box.height}</note>')
        if variant == "sdl":
            lines.append('        <sdl:seg-defs>')
            lines.append(f'          <sdl:seg id="{inst.id}" conf="Draft" />')
            lines.append('        </sdl:seg-defs>')
        if variant == "smartling":
            lines.append(f'        <sl:variant variant="text" />')
        lines.append('      </trans-unit>')

    lines.append('    </body>')
    lines.append('  </file>')
    lines.append('</xliff>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# XLIFF 1.2 import
# ---------------------------------------------------------------------------

def _local_name(name: str) -> str:
    """Namespace-agnostic XML local name (XLIFF CAT tools vary wildly)."""
    return name.rsplit("}", 1)[-1].split(":")[-1].lower()


def _local_attr(element: ET.Element, *names: str) -> Optional[str]:
    wanted = {name.lower() for name in names}
    for key, value in element.attrib.items():
        if _local_name(key) in wanted and value:
            return value
    return None


def _first_descendant(element: ET.Element, *names: str) -> Optional[ET.Element]:
    wanted = {name.lower() for name in names}
    for child in element.iter():
        if child is element:
            continue
        if _local_name(child.tag) in wanted:
            return child
    return None


def _segment_text(element: Optional[ET.Element]) -> str:
    """Preserve text inside CAT markup such as <mrk>, <g>, and <ph>."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _xliff_units(xml_text: str) -> List[Dict[str, Any]]:
    """Extract target-bearing XLIFF 1.2 and 2.x/CAT units.

    SDLXLIFF, MemoQ, Smartling, and Crowdin wrap targets in their own
    namespaces and often place inline <mrk> tags inside target.  ElementTree's
    namespace-exact ``find('x:target')`` silently dropped those translations;
    local-name traversal deliberately accepts standards-compliant extensions
    while leaving the immutable ToFU-ID mapping policy to the caller.
    """
    root = ET.fromstring(xml_text)
    units: List[Dict[str, Any]] = []
    for element in root.iter():
        if _local_name(element.tag) not in {"trans-unit", "unit"}:
            continue
        unit_id = _local_attr(element, "id")
        keys = [key for key in (
            unit_id, _local_attr(element, "resname", "name"), _local_attr(element, "id")
        ) if key]
        source = _segment_text(_first_descendant(element, "source"))
        target = _segment_text(_first_descendant(element, "target", "seg-target"))
        notes = [
            _segment_text(child) for child in element.iter()
            if _local_name(child.tag) in {"note", "context"}
        ]
        # XLIFF 2.0 may put one or more <segment> children under a unit.  A
        # segment id is a useful extra identity key, but unit id remains too.
        segments = [child for child in element.iter() if _local_name(child.tag) == "segment"]
        if segments:
            for segment in segments:
                segment_keys = [key for key in [_local_attr(segment, "id"), *keys] if key]
                seg_source = _segment_text(_first_descendant(segment, "source")) or source
                seg_target = _segment_text(_first_descendant(segment, "target", "seg-target"))
                units.append({"keys": segment_keys, "source": seg_source, "target": seg_target, "notes": notes})
        else:
            units.append({"keys": keys, "source": source, "target": target, "notes": notes})
    return units


def import_xliff(xml_text: str) -> Dict[str, str]:
    """Parse XLIFF and return the direct-ID subset for legacy callers."""
    translations: Dict[str, str] = {}
    for unit in _xliff_units(xml_text):
        if unit["target"] and unit["keys"]:
            translations[unit["keys"][0]] = unit["target"]
    return translations


def _norm_segment(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _bbox_from_notes(notes: List[str]) -> Optional[Tuple[int, int, int, int]]:
    joined = " ".join(notes)
    match = re.search(r"\bbbox\s*:\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)", joined, re.I)
    if not match:
        return None
    return tuple(int(value) for value in match.groups())


def import_xliff_for_manifest(xml_text: str, manifest: TextManifest) -> Dict[str, Any]:
    """Map CAT/XLIFF segments back to ToFU regions with auditable fallbacks.

    Direct ``rN``/``resname`` identity wins.  When a CAT tool regenerates
    IDs, we accept a unique exported bbox note, then a unique normalized source
    segment.  Ambiguous source text fails closed and appears in ``unresolved``
    rather than overwriting a different region.
    """
    ids = {inst.id: inst.id for inst in manifest.instances}
    bboxes = {
        (inst.bounding_box.x, inst.bounding_box.y, inst.bounding_box.width, inst.bounding_box.height): inst.id
        for inst in manifest.instances
    }
    sources: Dict[str, List[str]] = {}
    for inst in manifest.instances:
        sources.setdefault(_norm_segment(inst.text or ""), []).append(inst.id)

    translations: Dict[str, str] = {}
    matched_by = {"id": 0, "bbox": 0, "source": 0}
    unresolved, empty = [], 0
    for unit in _xliff_units(xml_text):
        target = unit["target"]
        if not target:
            empty += 1
            continue
        rid = next((ids[key] for key in unit["keys"] if key in ids), None)
        method = "id" if rid else None
        if rid is None:
            bbox = _bbox_from_notes(unit["notes"])
            if bbox in bboxes:
                rid, method = bboxes[bbox], "bbox"
        if rid is None:
            normalized_source = _norm_segment(unit["source"])
            source_matches = sources.get(normalized_source, []) if normalized_source else []
            if len(source_matches) == 1:
                rid, method = source_matches[0], "source"
        if rid is None:
            unresolved.append({"id": unit["keys"][0] if unit["keys"] else None, "source": unit["source"]})
            continue
        translations[rid] = target
        matched_by[method] += 1
    return {
        "translations": translations, "matched_by": matched_by,
        "unresolved": unresolved, "empty_targets": empty,
    }


# ---------------------------------------------------------------------------
# TMX export
# ---------------------------------------------------------------------------

def export_tmx(
    pairs: List[Tuple],
    src_lang: str = "en",
    targ_lang: str = "",
) -> str:
    """Generate TMX XML from (id, source, target, note[, src_lang, tgt_lang])
    tuples. Per-pair languages (elements 5-6, tofu codes) override the
    manifest-level defaults on the tuv elements."""
    src_full = _lang_to_full(src_lang)
    targ_full = _lang_to_full(targ_lang) if targ_lang else ""

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<tmx version="1.4">',
        f'  <header creationtool="ToFU" creationtoolversion="{_version()}"',
        f'    segtype="sentence" o-tmf="plain text"',
        f'    adminlang="en" srclang="{src_full}"',
        '    datatype="plaintext" />',
        '  <body>',
    ]
    for pair in pairs:
        tu_id, source, target, note = pair[0], pair[1], pair[2], pair[3]
        pair_src = _lang_to_full(pair[4]) if len(pair) > 4 and pair[4] else src_full
        pair_targ = _lang_to_full(pair[5]) if len(pair) > 5 and pair[5] else targ_full
        lines.append(f'    <tu tuid="{tu_id}">')
        if note:
            lines.append(f'      <note>{xml_escape(note)}</note>')
        lines.append(f'      <tuv xml:lang="{pair_src}"><seg>{xml_escape(source)}</seg></tuv>')
        lines.append(f'      <tuv xml:lang="{pair_targ}"><seg>{xml_escape(target)}</seg></tuv>')
        lines.append('    </tu>')
    lines.append('  </body>')
    lines.append('</tmx>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TMX import
# ---------------------------------------------------------------------------

def import_tmx(xml_text: str) -> Dict[str, str]:
    """Parse TMX and return {tu_id: target_segment}."""
    tree = ET.fromstring(xml_text)
    translations: Dict[str, str] = {}

    for tu in tree.findall(".//tu"):
        tuid = tu.get("tuid")
        if tuid is None:
            continue
        tuvs = tu.findall("tuv")
        if len(tuvs) >= 2:
            seg = tuvs[-1].find("seg")
            if seg is not None and seg.text:
                translations[tuid] = seg.text

    return translations


# ---------------------------------------------------------------------------
# TSV export/import
# ---------------------------------------------------------------------------

def export_tsv(manifest: TextManifest) -> str:
    lines = ["id\tsource\tsrc_lang\ttarget\ttgt_lang\tfont\tbbox"]
    for inst in manifest.instances:
        if getattr(inst, "dnt", False):
            continue
        bbox_str = f"{inst.bounding_box.x},{inst.bounding_box.y},{inst.bounding_box.width},{inst.bounding_box.height}"
        src_lang = inst.language or inst.detected_language or manifest.src_lang or ""
        tgt_lang = inst.target_language or manifest.targ_lang or ""
        font = (inst.style_profile.font_family if inst.style_profile else None) or ""
        lines.append(
            f"{inst.id}\t{inst.text or ''}\t{src_lang}\t"
            f"{inst.target_text or ''}\t{tgt_lang}\t{font}\t{bbox_str}"
        )
    return "\n".join(lines)


def import_tsv(text: str) -> Dict[str, str]:
    """header-aware: finds the 'target' column, so both the old 4-column
    and new 7-column layouts round-trip."""
    translations: Dict[str, str] = {}
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(reader, None)
    if not header:
        return translations
    try:
        target_idx = [h.strip().lower() for h in header].index("target")
    except ValueError:
        target_idx = 2
    for row in reader:
        if len(row) > target_idx:
            translations[row[0]] = row[target_idx]
    return translations


# ---------------------------------------------------------------------------
# CSV export/import
# ---------------------------------------------------------------------------

def export_csv(manifest: TextManifest) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "source", "src_lang", "target", "tgt_lang", "font", "bbox"])
    for inst in manifest.instances:
        if getattr(inst, "dnt", False):
            continue
        bbox_str = f"{inst.bounding_box.x},{inst.bounding_box.y},{inst.bounding_box.width},{inst.bounding_box.height}"
        src_lang = inst.language or inst.detected_language or manifest.src_lang or ""
        tgt_lang = inst.target_language or manifest.targ_lang or ""
        font = (inst.style_profile.font_family if inst.style_profile else None) or ""
        writer.writerow([
            inst.id, inst.text or "", src_lang,
            inst.target_text or "", tgt_lang, font, bbox_str,
        ])
    return buf.getvalue()


def import_csv(text: str) -> Dict[str, str]:
    translations: Dict[str, str] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        rid = row.get("id")
        target = row.get("target")
        if rid and target:
            translations[rid] = target
    return translations


# ---------------------------------------------------------------------------
# TXT export/import
# ---------------------------------------------------------------------------

def export_txt(manifest: TextManifest) -> str:
    lines = []
    for inst in manifest.instances:
        if getattr(inst, "dnt", False):
            continue
        lines.append(f"[{inst.id}] {inst.text or ''}")
    return "\n".join(lines)


def import_txt(text: str) -> Dict[str, str]:
    """Parse TXT format: [r1] translated text"""
    translations: Dict[str, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("["):
            close = line.find("]")
            if close > 0:
                rid = line[1:close]
                target = line[close + 1:].strip()
                translations[rid] = target
    return translations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LANG_FULL = {
    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
    "it": "it-IT", "pt": "pt-PT", "ru": "ru-RU", "ja": "ja-JP",
    "ko": "ko-KR", "zh-cn": "zh-CN", "zh-tw": "zh-TW",
    "ar": "ar-SA", "he": "he-IL", "hi": "hi-IN", "th": "th-TH",
    "vi": "vi-VN", "tr": "tr-TR", "nl": "nl-NL", "sv": "sv-SE",
    "pl": "pl-PL", "el": "el-GR",
}


def _lang_to_full(lang: str) -> str:
    return _LANG_FULL.get(lang, lang)


def detect_format(filename: str) -> str:
    """Infer format from file extension."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return {
        "xliff": "xliff", "xlf": "xliff", "sdlxliff": "xliff",
        "mxliff": "xliff", "mqxliff": "xliff", "txlf": "xliff",
        "tmx": "tmx",
        "tsv": "tsv",
        "csv": "csv",
        "txt": "txt",
        "vtm": "vtm",   # Visual Translation Memory, spec/vtm-1.0.md
    }.get(ext, "txt")


def decode_translation_bytes(raw: bytes) -> str:
    """Decode common CAT/TMS text exports without UTF-32-as-UTF-16 mojibake.

    SDL Trados exports UTF-16 often enough that this belongs in interchange
    rather than in one HTTP endpoint.  Four-byte BOMs must be tested first:
    their leading two bytes also resemble UTF-16 and Python can decode that
    wrong choice without raising, yielding nul-padded XML.
    """
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encodings = ("utf-32", "utf-16", "utf-8-sig")
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16", "utf-8-sig", "utf-32")
    else:
        encodings = ("utf-8-sig", "utf-16", "utf-32")
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("translation", raw, 0, len(raw), "not valid UTF-8, UTF-16, or UTF-32")


def import_file(filename: str, content: str) -> Dict[str, str]:
    """Auto-detect format and import."""
    fmt = detect_format(filename)
    if fmt == "xliff":
        return import_xliff(content)
    elif fmt == "tmx":
        return import_tmx(content)
    elif fmt == "tsv":
        return import_tsv(content)
    elif fmt == "csv":
        return import_csv(content)
    else:
        return import_txt(content)


def extract_ground_truth(filename: str, content: str, source_lang: Optional[str] = None) -> List[str]:
    """Extract source-side text from common localization interchange files.

    This intentionally returns segments, not a translation mapping. The API's
    Ground Truth normalizer remains responsible for whitespace tokenization,
    Unicode preservation, and deduplication.
    """
    fmt = detect_format(filename)
    if fmt == "xliff":
        return [unit["source"].strip() for unit in _xliff_units(content) if unit["source"].strip()]
    if fmt == "tmx":
        root = ET.fromstring(content)
        wanted = (source_lang or "").lower().split("-")[0]
        values: List[str] = []
        for tu in root.iter():
            if _local_name(tu.tag) != "tu":
                continue
            tuvs = [child for child in tu if _local_name(child.tag) == "tuv"]
            selected = None
            if wanted:
                selected = next((tuv for tuv in tuvs if (
                    (_local_attr(tuv, "lang") or "").lower().split("-")[0] == wanted
                )), None)
            if selected is None:
                selected = tuvs[0] if tuvs else None
            segment = _first_descendant(selected, "seg") if selected is not None else None
            text = _segment_text(segment).strip()
            if text:
                values.append(text)
        return values
    if fmt in {"tsv", "csv"}:
        delimiter = "\t" if fmt == "tsv" else ","
        rows = csv.reader(io.StringIO(content), delimiter=delimiter)
        header = next(rows, [])
        source_index = next((i for i, value in enumerate(header) if value.strip().lower() in {
            "source", "source_text", "src", "term", "text",
        }), None)
        if source_index is None:
            source_index = 1 if len(header) > 1 and header[0].strip().lower() in {"id", "key"} else 0
            rows = iter([header, *rows])
        return [row[source_index].strip() for row in rows if len(row) > source_index and row[source_index].strip()]
    return [line.strip() for line in content.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Semantic (Plate-oriented) projections
#
# The region-based writers above stay exactly as they are.  A file exported
# before Basil owned interchange must keep importing, so the two schemes
# coexist and the IMPORT side decides which it is looking at rather than the
# caller having to declare it.
#
# The durable `plate_uid` is the identity; the positional "u2" travels only as
# a human-readable resname.  A returned file is matched on uid + revision:
# same uid with a different revision means the plate moved on since it was
# sent out, which must stop the import rather than overwrite the newer plate.
# ---------------------------------------------------------------------------

TOFU_NOTE_PREFIX = "tofu"


def _plate_units(manifest: TextManifest) -> List[Any]:
    """Exportable plates in source reading order."""
    units = list(getattr(manifest, "semantic_units", None) or [])
    units.sort(key=lambda unit: (unit.display_number is None, unit.display_number or 0))
    return [unit for unit in units if (unit.source_text or "").strip()]


def _plate_is_dnt(unit: Any, by_id: Dict[str, Any]) -> bool:
    """A plate every one of whose regions is withheld from translation."""
    members = [by_id[rid] for rid in unit.region_ids if rid in by_id]
    if not members:
        return False
    return all(getattr(inst, "dnt", False) for inst in members)


def _plate_target(unit: Any, by_id: Dict[str, Any]) -> str:
    """The plate's current target: an applied arrangement, else its members."""
    substitution = unit.substitution if isinstance(unit.substitution, dict) else None
    if substitution and substitution.get("applied") and substitution.get("target_text"):
        return str(substitution["target_text"])
    members = [by_id[rid] for rid in unit.region_ids if rid in by_id]
    parts = [(inst.target_text or "").strip() for inst in members]
    return " ".join(part for part in parts if part)


def export_xliff_semantic(
    manifest: TextManifest,
    src_lang: str = "en",
    targ_lang: str = "",
    variant: str = "standard",
) -> str:
    """One trans-unit per Plate, carrying durable identity and revision.

    A translator sees a whole reading unit -- "Rue des Martyrs", not "Rue" /
    "des" / "Martyrs" in three separate boxes -- which is the entire reason
    plates exist.  Region membership rides along in a note so the file maps
    back without ToFU having to guess.
    """
    src_lang_full = _lang_to_full(src_lang)
    targ_lang_full = _lang_to_full(targ_lang) if targ_lang else src_lang_full
    by_id = {inst.id: inst for inst in manifest.instances}

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2"',
    ]
    if variant == "sdl":
        lines.append('  xmlns:sdl="http://sdl.com/FileFormats/SdlXliff/1.0"')
    elif variant == "crowdin":
        lines.append('  xmlns:cr="http://crowdin.com/ns/xliff"')
    elif variant == "smartling":
        lines.append('  xmlns:sl="http://smartling.com/ns/xliff"')
    lines.append('>')
    lines.append(f'  <file source-language="{src_lang_full}" '
                 f'target-language="{targ_lang_full}" '
                 f'datatype="plaintext" original="{manifest.asset_id}">')
    lines.append('    <header>')
    lines.append(f'      <tool tool-id="tofu" tool-name="ToFU" tool-version="{_version()}" />')
    lines.append('    </header>')
    lines.append('    <body>')

    for unit in _plate_units(manifest):
        if _plate_is_dnt(unit, by_id):
            continue
        source = unit.source_text or ""
        target = _plate_target(unit, by_id)
        state = "translated" if target else "needs-translation"
        label = "tofu:" + unit.id
        unit_attrs = 'id="%s" resname="%s"' % (
            xml_escape(unit.plate_uid or unit.id), xml_escape(label),
        )
        if variant == "crowdin":
            unit_attrs += ' cr:context="plate:%s"' % xml_escape(unit.id)
        lines.append(f'      <trans-unit {unit_attrs}>')
        lines.append(f'        <source xml:lang="{src_lang_full}">{xml_escape(source)}</source>')
        lines.append(f'        <target xml:lang="{targ_lang_full}" state="{state}">{xml_escape(target)}</target>')
        lines.append(f'        <note from="{TOFU_NOTE_PREFIX}">plate: {xml_escape(unit.plate_uid or "")}</note>')
        lines.append(f'        <note from="{TOFU_NOTE_PREFIX}">revision: {xml_escape(unit.membership_hash or "")}</note>')
        lines.append(f'        <note from="{TOFU_NOTE_PREFIX}">regions: {xml_escape(",".join(unit.region_ids))}</note>')
        if variant == "sdl":
            lines.append('        <sdl:seg-defs>')
            lines.append(f'          <sdl:seg id="{xml_escape(unit.id)}" conf="Draft" />')
            lines.append('        </sdl:seg-defs>')
        if variant == "smartling":
            lines.append('        <sl:variant variant="text" />')
        lines.append('      </trans-unit>')

    lines.append('    </body>')
    lines.append('  </file>')
    lines.append('</xliff>')
    return "\n".join(lines)


SEMANTIC_COLUMNS = [
    "unit_id", "plate_uid", "revision", "source", "target",
    "source_language", "target_language", "region_ids", "status", "context",
]


def _semantic_rows(manifest: TextManifest, targ_lang: str = "") -> List[List[str]]:
    by_id = {inst.id: inst for inst in manifest.instances}
    resolved_target_lang = targ_lang or manifest.targ_lang or ""
    rows: List[List[str]] = []
    for unit in _plate_units(manifest):
        if _plate_is_dnt(unit, by_id):
            continue
        target = _plate_target(unit, by_id)
        rows.append([
            unit.id,
            unit.plate_uid or "",
            unit.membership_hash or "",
            unit.source_text or "",
            target,
            manifest.src_lang or "",
            resolved_target_lang,
            ",".join(unit.region_ids),
            "translated" if target else "needs-translation",
            unit.entity_type or "",
        ])
    return rows


def export_csv_semantic(manifest: TextManifest, targ_lang: str = "") -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(SEMANTIC_COLUMNS)
    writer.writerows(_semantic_rows(manifest, targ_lang))
    return buf.getvalue()


def export_tsv_semantic(manifest: TextManifest, targ_lang: str = "") -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(SEMANTIC_COLUMNS)
    writer.writerows(_semantic_rows(manifest, targ_lang))
    return buf.getvalue()


def export_txt_semantic(manifest: TextManifest) -> str:
    """Plate source text, one per line, in source reading order."""
    by_id = {inst.id: inst for inst in manifest.instances}
    return "\n".join(
        unit.source_text for unit in _plate_units(manifest)
        if not _plate_is_dnt(unit, by_id)
    )


def export_tmx_profiles(
    manifest: TextManifest,
    src_lang: str = "en",
    targ_lang: str = "",
    profile: str = "semantic",
) -> str:
    """Translation memory at plate level, region level, or both.

    A memory entry is only worth having when both sides are real and aligned.
    Plates give phrase memory; regions give the atomic pairs.  Anything with
    an empty side is skipped rather than written as a half-entry: a source
    with no target teaches a TM nothing and pollutes later matches.
    """
    by_id = {inst.id: inst for inst in manifest.instances}
    pairs: List[Tuple] = []
    if profile in {"semantic", "both"}:
        for unit in _plate_units(manifest):
            if _plate_is_dnt(unit, by_id):
                continue
            target = _plate_target(unit, by_id)
            if not target.strip():
                continue
            pairs.append((
                unit.plate_uid or unit.id, unit.source_text, target,
                "plate:" + unit.id,
                manifest.src_lang or src_lang,
                targ_lang or manifest.targ_lang or "",
            ))
    if profile in {"atomic", "both"}:
        for inst in manifest.instances:
            if getattr(inst, "dnt", False):
                continue
            source = (inst.text or "").strip()
            target = (inst.target_text or "").strip()
            if not source or not target:
                continue
            pairs.append((
                inst.id, source, target,
                f"bbox:{inst.bounding_box.x},{inst.bounding_box.y}",
                inst.language or inst.detected_language or src_lang,
                inst.target_language or targ_lang or "",
            ))
    return export_tmx(pairs, src_lang, targ_lang)


# ---------------------------------------------------------------------------
# Semantic import
# ---------------------------------------------------------------------------

def _note_value(notes: List[str], key: str) -> Optional[str]:
    for note in notes:
        match = re.search(rf"\b{key}\s*:\s*([^\s,]+)", note or "", re.I)
        if match:
            return match.group(1)
    return None


def import_semantic_for_manifest(xml_text: str, manifest: TextManifest) -> Dict[str, Any]:
    """Map a returned file back to Plates, failing closed on drift.

    Resolution order is fixed and deliberately narrow:

      1. durable plate_uid whose revision still matches  -> applied
      2. durable plate_uid whose revision has MOVED      -> stale, refused
      3. a plate source text that is unique in the asset -> applied
      4. anything else                                   -> unresolved

    A positional "u2" arriving from outside is never trusted: after any
    region insert or reorder it names a different plate.

    Single-region plates are applied directly -- there is only one place the
    text can go.  Multi-region plates come back as PROPOSALS: distributing a
    target across regions needs alignment the user has to approve, so this
    step records the translation without touching region targets.
    """
    units = list(getattr(manifest, "semantic_units", None) or [])
    by_uid = {unit.plate_uid: unit for unit in units if unit.plate_uid}
    by_source: Dict[str, List[Any]] = {}
    for unit in units:
        by_source.setdefault(_norm_segment(unit.source_text or ""), []).append(unit)

    translations: Dict[str, str] = {}
    proposals: List[Dict[str, Any]] = []
    resolution = {"plate_uid": 0, "plate_source": 0}
    stale: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    empty = 0

    for parsed in _xliff_units(xml_text):
        target = parsed["target"]
        if not target:
            empty += 1
            continue
        notes = parsed["notes"]
        uid = _note_value(notes, "plate") or next(
            (key for key in parsed["keys"] if key in by_uid), None
        )
        revision = _note_value(notes, "revision")

        unit = by_uid.get(uid) if uid else None
        method = None
        if unit is not None:
            if revision and unit.membership_hash and revision != unit.membership_hash:
                stale.append({
                    "plate_uid": unit.plate_uid, "plate_id": unit.id,
                    "source": parsed["source"],
                    "reason": "the plate changed after this file was exported",
                })
                continue
            method = "plate_uid"
        else:
            matches = by_source.get(_norm_segment(parsed["source"]), [])
            if len(matches) == 1:
                unit, method = matches[0], "plate_source"
            elif len(matches) > 1:
                ambiguous.append({"source": parsed["source"],
                                  "candidates": [item.id for item in matches]})
                continue

        if unit is None:
            unresolved.append({"id": parsed["keys"][0] if parsed["keys"] else None,
                               "source": parsed["source"]})
            continue

        resolution[method] += 1
        present = [rid for rid in unit.region_ids
                   if rid in {inst.id for inst in manifest.instances}]
        if len(present) == 1:
            translations[present[0]] = target
        else:
            proposals.append({
                "plate_uid": unit.plate_uid, "plate_id": unit.id,
                "display_number": unit.display_number,
                "source": unit.source_text, "target": target,
                "region_ids": list(unit.region_ids),
                "reason": "needs region distribution before it can be applied",
            })

    return {
        "translations": translations,
        "proposals": proposals,
        "resolution": resolution,
        "stale": stale,
        "ambiguous": ambiguous,
        "unresolved": unresolved,
        "empty_targets": empty,
    }


def looks_semantic(xml_text: str) -> bool:
    """Does this file carry ToFU plate identity?

    Checked rather than declared: a user drops a file in, and which scheme it
    was cut from is a property of the file, not something they should have to
    tell us.
    """
    try:
        for parsed in _xliff_units(xml_text):
            if _note_value(parsed["notes"], "plate"):
                return True
    except ET.ParseError:
        return False
    return False


def _tabular_rows(text: str, delimiter: str) -> List[Dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [row for row in reader]


def looks_semantic_tabular(text: str, delimiter: str) -> bool:
    """Does this table carry plate identity columns?

    Same principle as the XLIFF side: the file says which scheme it came
    from, so the user never has to declare it.
    """
    try:
        header = next(csv.reader(io.StringIO(text), delimiter=delimiter))
    except StopIteration:
        return False
    return "plate_uid" in {column.strip() for column in header}


def import_semantic_tabular_for_manifest(
    text: str, manifest: TextManifest, delimiter: str = ",",
) -> Dict[str, Any]:
    """CSV/TSV counterpart of `import_semantic_for_manifest`.

    Same resolution order and the same refusals: durable uid with a matching
    revision, else a unique plate source, else unresolved -- and a revision
    that has moved is stale, never merged.
    """
    units = list(getattr(manifest, "semantic_units", None) or [])
    by_uid = {unit.plate_uid: unit for unit in units if unit.plate_uid}
    by_source: Dict[str, List[Any]] = {}
    for unit in units:
        by_source.setdefault(_norm_segment(unit.source_text or ""), []).append(unit)
    known_regions = {inst.id for inst in manifest.instances}

    translations: Dict[str, str] = {}
    proposals: List[Dict[str, Any]] = []
    resolution = {"plate_uid": 0, "plate_source": 0}
    stale: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    empty = 0

    for row in _tabular_rows(text, delimiter):
        target = (row.get("target") or "").strip()
        if not target:
            empty += 1
            continue
        uid = (row.get("plate_uid") or "").strip()
        revision = (row.get("revision") or "").strip()
        source = row.get("source") or ""

        unit = by_uid.get(uid) if uid else None
        method = None
        if unit is not None:
            if revision and unit.membership_hash and revision != unit.membership_hash:
                stale.append({
                    "plate_uid": unit.plate_uid, "plate_id": unit.id, "source": source,
                    "reason": "the plate changed after this file was exported",
                })
                continue
            method = "plate_uid"
        else:
            matches = by_source.get(_norm_segment(source), [])
            if len(matches) == 1:
                unit, method = matches[0], "plate_source"
            elif len(matches) > 1:
                ambiguous.append({"source": source,
                                  "candidates": [item.id for item in matches]})
                continue

        if unit is None:
            unresolved.append({"id": row.get("unit_id"), "source": source})
            continue

        resolution[method] += 1
        present = [rid for rid in unit.region_ids if rid in known_regions]
        if len(present) == 1:
            translations[present[0]] = target
        else:
            proposals.append({
                "plate_uid": unit.plate_uid, "plate_id": unit.id,
                "display_number": unit.display_number,
                "source": unit.source_text, "target": target,
                "region_ids": list(unit.region_ids),
                "reason": "needs region distribution before it can be applied",
            })

    return {
        "translations": translations, "proposals": proposals,
        "resolution": resolution, "stale": stale, "ambiguous": ambiguous,
        "unresolved": unresolved, "empty_targets": empty,
    }
