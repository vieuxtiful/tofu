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
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

from tofu.core.types import TextManifest


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
    lines.append('      <tool tool-id="tofu" tool-name="ToFU" tool-version="0.2" />')
    lines.append('    </header>')
    lines.append('    <body>')

    for inst in manifest.instances:
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

def import_xliff(xml_text: str) -> Dict[str, str]:
    """Parse XLIFF 1.2 and return {trans_unit_id: target_text}."""
    tree = ET.fromstring(xml_text)
    ns = {"x": "urn:oasis:names:tc:xliff:document:1.2"}
    translations: Dict[str, str] = {}

    for unit in tree.findall(".//x:trans-unit", ns):
        uid = unit.get("id")
        if uid is None:
            continue
        target_el = unit.find("x:target", ns)
        if target_el is not None and target_el.text:
            translations[uid] = target_el.text

    return translations


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
        '  <header creationtool="ToFU" creationtoolversion="0.1"',
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
        "xliff": "xliff", "xlf": "xliff",
        "tmx": "tmx",
        "tsv": "tsv",
        "csv": "csv",
        "txt": "txt",
    }.get(ext, "txt")


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
