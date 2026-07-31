"""Versioned, integrity-checked correction dictionaries.

Correction data is deliberately kept outside correction algorithms so it can
be reviewed, upgraded, and rolled back without changing Python code.  A
resource checksum covers its canonical ``entries`` payload; metadata remains
human-readable and can evolve independently under ``schema_version``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED_SCHEMA_VERSION = "1"
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


class CorrectionResourceError(ValueError):
    """A correction resource is malformed, unsupported, or corrupted."""


@dataclass(frozen=True)
class CorrectionResource:
    resource_id: str
    kind: str
    locale: str
    schema_version: str
    data_version: str
    checksum: str
    provenance: Mapping[str, Any]
    license: str
    entries: tuple[Mapping[str, Any], ...]

    def audit_identity(self) -> dict[str, str]:
        """Stable identity recorded alongside an applied correction."""
        return {
            "id": self.resource_id,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "checksum": self.checksum,
        }


def _payload_checksum(entries: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _parse_resource(data: Any, *, source: str) -> CorrectionResource:
    if not isinstance(data, dict):
        raise CorrectionResourceError(f"{source}: resource root must be an object")
    required = {
        "id",
        "kind",
        "locale",
        "schema_version",
        "data_version",
        "checksum",
        "provenance",
        "license",
        "entries",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise CorrectionResourceError(f"{source}: missing fields: {', '.join(missing)}")
    if data["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise CorrectionResourceError(
            f"{source}: unsupported schema_version {data['schema_version']!r}"
        )
    if not isinstance(data["data_version"], str) or not _SEMVER.fullmatch(data["data_version"]):
        raise CorrectionResourceError(f"{source}: data_version must be semantic versioning")
    if not isinstance(data["entries"], list) or not all(
        isinstance(entry, dict) for entry in data["entries"]
    ):
        raise CorrectionResourceError(f"{source}: entries must be an array of objects")
    if not isinstance(data["provenance"], dict):
        raise CorrectionResourceError(f"{source}: provenance must be an object")
    actual = _payload_checksum(data["entries"])
    if data["checksum"] != actual:
        raise CorrectionResourceError(
            f"{source}: checksum mismatch (declared {data['checksum']!r}, actual {actual!r})"
        )
    return CorrectionResource(
        resource_id=str(data["id"]),
        kind=str(data["kind"]),
        locale=str(data["locale"]),
        schema_version=data["schema_version"],
        data_version=data["data_version"],
        checksum=data["checksum"],
        provenance=dict(data["provenance"]),
        license=str(data["license"]),
        entries=tuple(dict(entry) for entry in data["entries"]),
    )


def load_correction_file(path: Path) -> CorrectionResource:
    """Load and validate a correction resource from an explicit path."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectionResourceError(f"{path}: unable to read correction resource") from exc
    return _parse_resource(data, source=str(path))


@lru_cache(maxsize=None)
def load_correction_resource(relative_path: str) -> CorrectionResource:
    """Load a packaged resource relative to ``tofu/resources/corrections``."""
    if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise CorrectionResourceError("resource path must be a safe relative path")
    resource = files("tofu").joinpath("resources", "corrections", *Path(relative_path).parts)
    try:
        data = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectionResourceError(
            f"{relative_path}: unable to read packaged correction resource"
        ) from exc
    return _parse_resource(data, source=relative_path)


def gazetteer_entries(resource: CorrectionResource) -> list[tuple[str, str]]:
    if resource.kind != "gazetteer":
        raise CorrectionResourceError(f"{resource.resource_id}: expected gazetteer resource")
    try:
        return [(str(entry["text"]), str(entry["language"])) for entry in resource.entries]
    except KeyError as exc:
        raise CorrectionResourceError(
            f"{resource.resource_id}: gazetteer entry lacks {exc.args[0]!r}"
        ) from exc


def variant_pairs(resource: CorrectionResource) -> dict[str, str]:
    if resource.kind != "variant-pairs":
        raise CorrectionResourceError(f"{resource.resource_id}: expected variant-pairs resource")
    try:
        return {str(entry["source"]): str(entry["target"]) for entry in resource.entries}
    except KeyError as exc:
        raise CorrectionResourceError(
            f"{resource.resource_id}: variant entry lacks {exc.args[0]!r}"
        ) from exc


def fold_text(text: str) -> str:
    """Comparison-only case/accent fold.

    Used to COMPARE two strings, never to rewrite one: a folded form is
    lossy, so it may decide whether an anchor matches but must never become
    the text a correction emits.
    """
    return "".join(
        char for char in unicodedata.normalize("NFD", text).casefold()
        if not unicodedata.combining(char)
    )


def phrase_entries(resource: CorrectionResource) -> list[dict[str, Any]]:
    """Validated multi-token phrases for segmentation-gated token recovery.

    A phrase needs at least two tokens because the course that consumes it
    recovers exactly ONE token and requires every other token in the window
    to match the OCR exactly.  A single-token phrase would carry no anchor
    and would degenerate into an unconditional find-and-replace.
    """
    if resource.kind != "phrase-forms":
        raise CorrectionResourceError(f"{resource.resource_id}: expected phrase-forms resource")
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in resource.entries:
        try:
            phrase, language = str(entry["phrase"]), str(entry["language"])
        except KeyError as exc:
            raise CorrectionResourceError(
                f"{resource.resource_id}: phrase entry lacks {exc.args[0]!r}"
            ) from exc
        if not phrase.strip() or not language:
            raise CorrectionResourceError(f"{resource.resource_id}: phrase entries cannot be empty")
        tokens = tuple(phrase.split())
        if len(tokens) < 2:
            raise CorrectionResourceError(
                f"{resource.resource_id}: phrase {phrase!r} needs at least two tokens to anchor"
            )
        if " ".join(tokens) != phrase:
            raise CorrectionResourceError(
                f"{resource.resource_id}: phrase {phrase!r} must be single-space separated"
            )
        folded = tuple(fold_text(token) for token in tokens)
        if any(not token for token in folded):
            raise CorrectionResourceError(
                f"{resource.resource_id}: phrase {phrase!r} has a token with no folded form"
            )
        key = (" ".join(folded), language)
        if key in seen:
            raise CorrectionResourceError(
                f"{resource.resource_id}: duplicate folded phrase/language entry {key[0]!r}/{language!r}"
            )
        seen.add(key)
        parsed.append({"phrase": phrase, "language": language, "tokens": tokens, "folded": folded})
    return parsed


def diacritic_entries(resource: CorrectionResource) -> list[dict[str, str]]:
    """Validated stored-form entries for pixel-gated Latin mark repair.

    ``folded`` is curated data rather than a runtime accent-stripping
    transform.  That keeps lookup auditable and prevents this correction path
    from becoming a broad, lossy normalization pass.
    """
    if resource.kind != "diacritic-forms":
        raise CorrectionResourceError(f"{resource.resource_id}: expected diacritic-forms resource")
    parsed: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in resource.entries:
        try:
            folded, text, language = (str(entry["folded"]), str(entry["text"]), str(entry["language"]))
        except KeyError as exc:
            raise CorrectionResourceError(
                f"{resource.resource_id}: diacritic entry lacks {exc.args[0]!r}"
            ) from exc
        if not folded or not text or not language:
            raise CorrectionResourceError(f"{resource.resource_id}: diacritic entries cannot be empty")
        if folded != fold_text(text):
            raise CorrectionResourceError(
                f"{resource.resource_id}: folded form does not match canonical text for {text!r}"
            )
        if len(folded) != len(text):
            raise CorrectionResourceError(
                f"{resource.resource_id}: diacritic forms must preserve character length"
            )
        key = (folded, language)
        if key in seen:
            raise CorrectionResourceError(
                f"{resource.resource_id}: duplicate folded/language entry {folded!r}/{language!r}"
            )
        seen.add(key)
        parsed.append({"folded": folded, "text": text, "language": language})
    return parsed
