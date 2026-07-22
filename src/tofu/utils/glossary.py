"""Glossary ingestion and cascading resolution for Basil.

The parser is deliberately conservative: malformed or ambiguous files fail
with :class:`GlossaryParseError`, while a valid file only contributes explicit
source/target term pairs.  Locale tags are retained (``en-US`` and ``en-EU``
are different keys); Basil may choose a base-language fallback only when an
exact locale pair is unavailable.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tofu.utils import interchange


class GlossaryParseError(Exception):
    """Raised when a glossary cannot be parsed into usable term pairs."""


@dataclass
class GlossaryEntry:
    source_term: str
    target_term: str
    src_lang: str
    targ_lang: str


@dataclass
class GlossaryFile:
    entries: List[GlossaryEntry]
    source_format: str
    entry_count: int
    language_pairs: List[Tuple[str, str]]


def _locale(value: Optional[str]) -> str:
    value = (value or "").strip().replace("_", "-")
    if not value:
        return ""
    parts = value.split("-")
    return "-".join([parts[0].lower(), *[part.upper() if len(part) in (2, 3) else part for part in parts[1:]]])


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _header_index(row: List[str], *names: str) -> Optional[int]:
    wanted = {name.replace("_", "").replace("-", "").lower() for name in names}
    for index, value in enumerate(row):
        key = _clean(value).replace(" ", "").replace("_", "").replace("-", "").lower()
        if key in wanted:
            return index
    return None


def _infer_pair(filename: str, default_src_lang: Optional[str], default_targ_lang: Optional[str]) -> Tuple[str, str]:
    src, targ = _locale(default_src_lang), _locale(default_targ_lang)
    if src and targ:
        return src, targ
    stem = Path(filename).stem.lower()
    match = re.search(r"(?:^|[_-])([a-z]{2,3}(?:[-_][a-z0-9]{2,})?)[-_](to[-_])?([a-z]{2,3}(?:[-_][a-z0-9]{2,})?)(?:$|[_-])", stem)
    if match:
        return _locale(match.group(1)), _locale(match.group(3))
    return src, targ


def _rows_to_entries(rows: Iterable[List[str]], filename: str, src: Optional[str], targ: Optional[str]) -> List[GlossaryEntry]:
    rows = [list(row) for row in rows if any(_clean(cell) for cell in row)]
    if not rows:
        return []
    first = rows[0]
    source_i = _header_index(first, "source", "src", "source_term", "source term", "term")
    target_i = _header_index(first, "target", "targ", "translation", "target_term", "target term")
    src_i = _header_index(first, "src_lang", "source_language", "source language", "source_locale")
    targ_i = _header_index(first, "targ_lang", "target_language", "target language", "target_locale")
    start = 1 if source_i is not None or target_i is not None else 0
    if source_i is None:
        source_i, target_i = 0, 1
    if target_i is None:
        target_i = 1
    inferred_src, inferred_targ = _infer_pair(filename, src, targ)
    entries: List[GlossaryEntry] = []
    for row in rows[start:]:
        if len(row) <= max(source_i, target_i):
            continue
        source_term, target_term = _clean(row[source_i]), _clean(row[target_i])
        row_src = _locale(row[src_i]) if src_i is not None and len(row) > src_i else inferred_src
        row_targ = _locale(row[targ_i]) if targ_i is not None and len(row) > targ_i else inferred_targ
        if source_term and target_term and row_src and row_targ:
            entries.append(GlossaryEntry(source_term, target_term, row_src, row_targ))
    return entries


def _parse_delimited(text: str, delimiter: str, filename: str, src: Optional[str], targ: Optional[str]) -> List[GlossaryEntry]:
    return _rows_to_entries(csv.reader(io.StringIO(text), delimiter=delimiter), filename, src, targ)


def _parse_txt(text: str, filename: str, src: Optional[str], targ: Optional[str]) -> List[GlossaryEntry]:
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and all(line.lstrip().startswith("[") for line in lines):
        return []  # ToFU translation export, not a glossary.
    if any("\t" in line for line in lines):
        return _parse_delimited(text, "\t", filename, src, targ)
    return []


def _parse_utx(text: str, filename: str, src: Optional[str], targ: Optional[str]) -> List[GlossaryEntry]:
    rows = [row for row in csv.reader(io.StringIO(text), delimiter="\t") if row and not row[0].startswith("#")]
    if rows and rows[0][0].strip().lower() in {"src_lang", "source"}:
        rows = rows[1:]
    entries: List[GlossaryEntry] = []
    for row in rows:
        if len(row) < 4:
            continue
        row_src, row_targ = _locale(row[0] or src), _locale(row[1] or targ)
        if _clean(row[2]) and _clean(row[3]) and row_src and row_targ:
            entries.append(GlossaryEntry(_clean(row[2]), _clean(row[3]), row_src, row_targ))
    return entries


def _parse_tbx(text: str, filename: str, src: Optional[str], targ: Optional[str]) -> List[GlossaryEntry]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise GlossaryParseError(f"TBX XML is invalid: {exc}") from exc
    entries: List[GlossaryEntry] = []
    for term_entry in root.iter():
        if interchange._local_name(term_entry.tag) != "termentry":
            continue
        sets: List[Tuple[str, str]] = []
        for lang_set in term_entry.iter():
            if interchange._local_name(lang_set.tag) != "langset":
                continue
            language = _locale(interchange._local_attr(lang_set, "lang", "xml:lang") or "")
            term = next((_clean(node.text) for node in lang_set.iter() if interchange._local_name(node.tag) == "term" and _clean(node.text)), "")
            if language and term:
                sets.append((language, term))
        if len(sets) >= 2:
            entries.append(GlossaryEntry(sets[0][1], sets[1][1], sets[0][0], sets[1][0]))
    return entries


def _parse_xliff(text: str, filename: str, src: Optional[str], targ: Optional[str]) -> List[GlossaryEntry]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise GlossaryParseError(f"XLIFF XML is invalid: {exc}") from exc
    entries: List[GlossaryEntry] = []
    for file_node in root.iter():
        if interchange._local_name(file_node.tag) != "file":
            continue
        file_src = _locale(interchange._local_attr(file_node, "source-language") or src)
        file_targ = _locale(interchange._local_attr(file_node, "target-language") or targ)
        if not file_src or not file_targ:
            continue
        for unit in interchange._xliff_units(ET.tostring(file_node, encoding="unicode")):
            source_term, target_term = _clean(unit.get("source")), _clean(unit.get("target"))
            if source_term and target_term:
                entries.append(GlossaryEntry(source_term, target_term, file_src, file_targ))
    if not entries:
        # XLIFF 2.x may omit a file wrapper; use request-provided locales.
        src_locale, targ_locale = _infer_pair(filename, src, targ)
        if src_locale and targ_locale:
            for unit in interchange._xliff_units(text):
                source_term, target_term = _clean(unit.get("source")), _clean(unit.get("target"))
                if source_term and target_term:
                    entries.append(GlossaryEntry(source_term, target_term, src_locale, targ_locale))
    return entries


def _parse_workbook(raw: bytes, filename: str, src: Optional[str], targ: Optional[str], xlsx: bool) -> List[GlossaryEntry]:
    try:
        if xlsx:
            import openpyxl  # type: ignore
            workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
            rows = next(workbook.active.iter_rows(values_only=True), ())
            all_rows = [list(rows)] + [list(row) for row in workbook.active.iter_rows(min_row=2, values_only=True)]
        else:
            import xlrd  # type: ignore
            workbook = xlrd.open_workbook(file_contents=raw, on_demand=True)
            sheet = workbook.sheet_by_index(0)
            all_rows = [sheet.row_values(index) for index in range(sheet.nrows)]
    except ImportError as exc:
        package = "openpyxl" if xlsx else "xlrd"
        raise GlossaryParseError(f"XLS support requires {package}. Install: pip install {package}") from exc
    except Exception as exc:
        raise GlossaryParseError(f"Unable to read {Path(filename).suffix.upper()} glossary: {exc}") from exc
    return _rows_to_entries(all_rows, filename, src, targ)


def parse_glossary(filename: str, raw: bytes, default_src_lang: Optional[str] = None, default_targ_lang: Optional[str] = None) -> GlossaryFile:
    fmt = interchange.detect_format(filename)
    if fmt == "txt" and filename.lower().endswith(".tab"):
        fmt = "tsv"
    if fmt == "txt" and filename.lower().endswith(".utx"):
        fmt = "utx"
    if fmt == "txt" and filename.lower().endswith(".tbx"):
        fmt = "tbx"
    if fmt == "txt" and filename.lower().endswith((".xls", ".xlsx")):
        fmt = "xlsx" if filename.lower().endswith(".xlsx") else "xls"
    try:
        text = interchange.decode_translation_bytes(raw)
        if fmt == "csv":
            entries = _parse_delimited(text, ",", filename, default_src_lang, default_targ_lang)
        elif fmt == "tsv":
            entries = _parse_delimited(text, "\t", filename, default_src_lang, default_targ_lang)
        elif fmt == "utx":
            entries = _parse_utx(text, filename, default_src_lang, default_targ_lang)
        elif fmt == "tbx":
            entries = _parse_tbx(text, filename, default_src_lang, default_targ_lang)
        elif fmt == "xliff":
            entries = _parse_xliff(text, filename, default_src_lang, default_targ_lang)
        elif fmt in {"xls", "xlsx"}:
            entries = _parse_workbook(raw, filename, default_src_lang, default_targ_lang, fmt == "xlsx")
        else:
            entries = _parse_txt(text, filename, default_src_lang, default_targ_lang)
    except GlossaryParseError:
        raise
    except Exception as exc:
        raise GlossaryParseError(f"Unable to parse {filename}: {exc}") from exc
    entries = [GlossaryEntry(_clean(item.source_term), _clean(item.target_term), _locale(item.src_lang), _locale(item.targ_lang)) for item in entries if _clean(item.source_term) and _clean(item.target_term) and _locale(item.src_lang) and _locale(item.targ_lang)]
    pairs = sorted({(item.src_lang, item.targ_lang) for item in entries})
    return GlossaryFile(entries=entries, source_format=fmt, entry_count=len(entries), language_pairs=pairs)


def to_basil_lexicon(glossary: GlossaryFile) -> Dict[Tuple[str, str], Dict[str, str]]:
    result: Dict[Tuple[str, str], Dict[str, str]] = {}
    for entry in glossary.entries:
        key = (entry.src_lang, entry.targ_lang)
        result.setdefault(key, {})[entry.source_term.casefold()] = entry.target_term
    return result


def merge_glossaries(built_in: Dict[Tuple[str, str], Dict[str, str]], uploaded: Dict[Tuple[str, str], Dict[str, str]], mode: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    if mode == "replace":
        return copy.deepcopy(uploaded)
    result = copy.deepcopy(built_in)
    if mode == "auxiliary":
        for pair, terms in uploaded.items():
            if pair not in result:
                result[pair] = copy.deepcopy(terms)
    else:  # merge
        for pair, terms in uploaded.items():
            result.setdefault(pair, {}).update(copy.deepcopy(terms))
    return result


def _load_file(path: Path) -> Tuple[Optional[Dict[Tuple[str, str], Dict[str, str]]], Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_lexicon = data.get("lexicon", {})
        lexicon = {tuple(key.split("|", 1)): dict(value) for key, value in raw_lexicon.items() if "|" in key and isinstance(value, dict)}
        meta = {key: value for key, value in data.items() if key != "lexicon"}
        meta["entry_count"] = sum(len(value) for value in lexicon.values())
        meta["language_pairs"] = [list(pair) for pair in sorted(lexicon)]
        return (lexicon or None), meta
    except Exception:
        return None, {}


def resolve_active_glossary(glossary_dir: Path, project_id: Optional[str]) -> Tuple[Optional[Dict[Tuple[str, str], Dict[str, str]]], Dict[str, Any]]:
    """Resolve global then project glossary; project mode controls overrides."""
    global_lex, global_meta = _load_file(glossary_dir / "global.json")
    project_lex, project_meta = _load_file(glossary_dir / f"project_{project_id}.json") if project_id else (None, {})
    if not global_lex and not project_lex:
        return None, {}
    from tofu.layers.basil import _GLOSSARY
    effective = merge_glossaries(_GLOSSARY, global_lex or {}, global_meta.get("mode", "auxiliary"))
    if project_lex:
        effective = merge_glossaries(effective, project_lex, project_meta.get("mode", "merge"))
    metadata = {
        "mode": project_meta.get("mode") or global_meta.get("mode", "auxiliary"),
        "entry_count": sum(len(value) for value in effective.values()),
        "language_pairs": [list(pair) for pair in sorted(effective)],
        "global": global_meta or None,
        "project": project_meta or None,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    return effective, metadata


def serializable_lexicon(lexicon: Dict[Tuple[str, str], Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {f"{src}|{targ}": terms for (src, targ), terms in lexicon.items()}
