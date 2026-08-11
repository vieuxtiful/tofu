## 🍢 Mise -- mise en place: the ingredients laid out, in order, before cooking.
## vieuxtiful
"""Turn text the user typed into ordered Blocks and atoms.

Whitespace keeps its visible significance, but it stops being the only
tokenizer.  Splitting on spaces alone loses four things this layer has to
keep, every one of which breaks localisation downstream:

* **duplicates** -- "PARIS PARIS" is two things to find, not one seen twice;
* **numbers** -- "5.2" is one atom, not "5", ".", "2";
* **languages written without spaces** -- 東京都渋谷区 has no delimiters at
  all, so a space-splitter returns one undifferentiated lump;
* **mixed scripts** -- "5.2 FRI アベンジャーズ" changes script three times,
  and each run may need different typography at render time.

Nothing here decides what a Block MEANS.  It only lays it out in order so
that later stages have something stable to align against.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from tofu.core.types import GuidedAtom, GuidedBlock
from tofu.layers.palate import direction_for, normalize_script, scripts_in

## A decimal, a thousands-separated figure, a time, a version -- one atom.
_NUMERIC_RE = re.compile(r"\d+(?:[.,:]\d+)*")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’‐-―-][^\W\d_]+)*", re.UNICODE)

## Scripts that do not delimit words with spaces.  Runs are split on script
## CHANGE rather than guessed at word boundaries: a segmenter would need a
## dictionary per language, and the plan rules out mandatory model downloads.
_UNSPACED = {"Hani", "Hira", "Kana", "Hang", "Thai"}


def _script_of(char: str) -> str:
    if char.isspace():
        return "Zyyy"
    found = scripts_in(char)
    return found[0] if found else "Zyyy"


def _classify(text: str) -> str:
    if _NUMERIC_RE.fullmatch(text):
        return "numeric"
    if _WORD_RE.fullmatch(text):
        return "word"
    if any(ch.isalnum() for ch in text):
        return "phrase"
    return "punctuation"


def atomize(text: str, block_id: str = "b1") -> List[GuidedAtom]:
    """Split a Block into ordered atoms, preserving order and duplicates."""
    atoms: List[GuidedAtom] = []
    if not (text or "").strip():
        return atoms

    position = 0
    for chunk in re.split(r"(\s+)", text):
        if not chunk or chunk.isspace():
            continue
        # Within a whitespace-delimited chunk, break on script change so an
        # unspaced run is not swallowed into its neighbour.
        runs: List[str] = []
        current, current_script = "", None
        for char in chunk:
            script = _script_of(char)
            # Digits and punctuation (Zyyy) attach to the run in progress
            # rather than starting one: "5.2" and "N°" stay whole.
            if current and (script == current_script or script == "Zyyy" or current_script == "Zyyy"):
                current += char
                if current_script == "Zyyy" and script != "Zyyy":
                    current_script = script
                continue
            if current:
                runs.append(current)
            current, current_script = char, script
        if current:
            runs.append(current)

        for run in runs:
            # An unspaced script gets one atom per run, not per character:
            # 東京都 is a place, and per-character atoms would ask the matcher
            # to locate three unrelated glyphs.
            pieces = [run]
            if any(_script_of(ch) in _UNSPACED for ch in run) and _NUMERIC_RE.search(run):
                pieces = [p for p in re.split(r"(\d+(?:[.,:]\d+)*)", run) if p]
            for piece in pieces:
                scripts = scripts_in(piece)
                atoms.append(GuidedAtom(
                    id=f"{block_id}a{position + 1}",
                    text=piece,
                    position=position,
                    kind=_classify(piece),
                    script=normalize_script(scripts[0]) if scripts else "Zyyy",
                    direction=direction_for(scripts),
                ))
                position += 1
    return atoms


def normalize_block_text(text: str) -> str:
    """Comparison form: NFC, collapsed runs of whitespace, trimmed.

    Case is deliberately preserved.  Signage capitalisation is evidence --
    "SORTIE" set in caps is a different rendering intent from "Sortie" --
    and the matcher folds case itself where it wants to.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def make_block(
    raw_text: str,
    position: int,
    *,
    block_id: Optional[str] = None,
    source_language: Optional[str] = None,
    scope: str = "asset",
    find_all: bool = False,
) -> GuidedBlock:
    """Build one ordered Block, with its language agreement already recorded."""
    from tofu.layers.palate import assess

    identifier = block_id or f"g{position + 1}"
    block = GuidedBlock(
        id=identifier,
        raw_text=raw_text,
        normalized_text=normalize_block_text(raw_text),
        position=position,
        source_language=source_language,
        scope=scope,
        find_all=find_all,
        atoms=atomize(raw_text, identifier),
    )
    assessment = assess(raw_text, source_language)
    block.language_assessment = {
        "state": assessment.state,
        "detected_language": assessment.detected_language,
        "scripts": assessment.scripts,
        "direction": assessment.direction,
        "reasons": assessment.reasons,
        "policy_revision": assessment.policy_revision,
    }
    return block


def make_blocks(
    entries: List[str],
    source_language: Optional[str] = None,
    scope: str = "asset",
) -> List[GuidedBlock]:
    """Ordered Blocks from ordered entries.

    Blank entries are dropped; everything else keeps its position, including
    two entries that happen to be identical.  A user who typed the same term
    twice is asking for two occurrences.
    """
    blocks: List[GuidedBlock] = []
    for entry in entries or []:
        if not (entry or "").strip():
            continue
        blocks.append(make_block(entry, len(blocks),
                                 source_language=source_language, scope=scope))
    return blocks


def from_legacy_ground_truth(
    terms: List[str], source_language: Optional[str] = None
) -> List[GuidedBlock]:
    """Read an existing `ground_truth: string[]` as compatibility Blocks.

    Legacy Ground Truth stays usable exactly as it is: it becomes ordered
    Blocks with `scope="project"` so nothing has to be re-entered, and so a
    manifest saved before Guided existed still opens.
    """
    return make_blocks(terms, source_language=source_language, scope="project")
