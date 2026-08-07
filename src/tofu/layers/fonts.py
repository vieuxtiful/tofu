## 🍢 Fonts - ToFU font registry
## vieuxtiful
"""
fonts as a modulatable entity: discovery, cached cmap coverage scoring,
and per-script ranking.

the frontend flow this enables:
  1. user uploads an asset and (optionally) selects a font
  2. the registry scores that font against the exact target script +
     language-required characters (diacritics, conjuncts) in the background
  3. tofu.validate() either confirms the selection or surfaces a ranked
     list of pre-verified alternatives — minimizing end-user work
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from tofu.core.types import ScrptSpprt
from tofu.layers.script_samples import SCRIPT_SAMPLES, LANG_REQUIRED

try:
    from fontTools.ttLib import TTFont, TTCollection
    HAVE_FONTTOOLS = True
except ImportError:  # registry degrades gracefully; tofu falls back to static map
    HAVE_FONTTOOLS = False


class _BrokenVendorTimestampFilter(logging.Filter):
    """Hide harmless malformed ``head`` dates without masking font errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.startswith("'created' timestamp")
            or message.startswith("'modified' timestamp")
        )


logging.getLogger("fontTools.ttLib.tables._h_e_a_d").addFilter(
    _BrokenVendorTimestampFilter()
)

FONT_EXTS = {".ttf", ".otf", ".ttc", ".otc"}
FULL_THRESHOLD = 0.995    # tolerate trivial gaps in range-derived samples
PARTIAL_THRESHOLD = 0.80

# repo root: fonts.py -> layers -> tofu -> src -> root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


#: Where a deployment may BUNDLE fonts. Read-only as far as ToFU cares.
_BUNDLED_ROOT = _PROJECT_ROOT / "server" / "fonts"

#: Where ToFU INSTALLS fonts at runtime -- uploads under ``user/``, packs
#: under ``packs/<name>/``. Deliberately a SIBLING of the bundled root
#: rather than a child of it; see the warning in ``pantry()``.
_INSTALLED_ROOT = _PROJECT_ROOT / "server" / "font-library"
USER_SUBDIR = "user"
PACKS_SUBDIR = "packs"


def _system_font_dir() -> str:
    return "C:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts"


def _static_roots() -> List[str]:
    """The candidates that constitute "the font library" for anything
    wanting one answer. Runtime installs are deliberately absent."""
    return [
        candidate for candidate in (
            os.environ.get("TOFU_FONT_DIR"),
            str(_BUNDLED_ROOT),
            _system_font_dir(),
        ) if candidate
    ]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def pantries() -> List[str]:
    """every font directory that exists, in precedence order.

    Nested roots are dropped: ``discover()`` walks with ``rglob``, so a root
    already contained by an earlier one would be scanned twice and every
    face loaded twice. Same keys, same result, wasted seconds -- and a
    misleading answer to "where do fonts come from".

    ``user/`` and ``packs/`` are not listed individually; they live under
    the installed root, which ``discover()`` already walks recursively.
    """
    roots: List[str] = []
    for candidate in _static_roots() + [str(_INSTALLED_ROOT)]:
        path = Path(candidate)
        if not path.exists():
            continue
        if any(_is_within(path, Path(existing)) for existing in roots):
            continue
        roots.append(candidate)
    return roots


def pantry() -> Optional[str]:
    """where the font ingredients are kept: the first font directory that
    actually exists, in preference order.

    every entry point that needs a FontRegistry (the server's validator,
    the eval harnesses) resolved this same chain independently, which
    meant a fix to one never reached the others. one pantry, one answer.

    returns None when nothing is found — callers build no registry and
    tofu falls back to its static script map.

    Considers only the STATIC roots, never the runtime-installed one, and
    that is load-bearing. The eval harnesses call this to name the single
    library they measure against, then join paths against it. If the
    installed root could win here, the first font a user ever uploaded
    would silently repoint every harness from the system library to a
    near-empty directory -- measured during development: 409 faces down to
    4 -- and every eval number afterwards would be quietly incomparable to
    the ones before. It is also why the installed root is a SIBLING of
    ``server/fonts`` rather than a child: creating ``server/fonts/user``
    would bring ``server/fonts`` into existence and shadow the system
    directory by exactly the same mechanism.
    """
    for candidate in _static_roots():
        if Path(candidate).exists():
            return candidate
    return None


def faces_of(registry: Any) -> Dict[str, "FontCoverage"]:
    """The faces a registry has discovered, tolerating a missing registry.

    Every consumer of the font library accepts ``font_registry=None`` -- a
    ToFU built without one falls back to its static script map -- which is
    why a dozen call sites wrote ``getattr(registry, "_fonts", {})``. That
    idiom was doing two jobs at once: reaching past an underscore, and
    absorbing None. This keeps the second and drops the first.
    """
    return getattr(registry, "faces", None) or {}


def writable_pantry(subdir: str) -> Path:
    """A font directory ToFU may write into, created on demand.

    Never under ``TOFU_FONT_DIR``: that points at whatever a deployment
    chose, which may be a read-only mount or the OS font directory, and
    writing uploads into ``C:/Windows/Fonts`` is not something this should
    ever do.
    """
    target = _INSTALLED_ROOT / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target

import re
_WEIGHT_TOKENS = re.compile(
    r"\s+(?:black|heavy|ultra(?:\s*bold)?|extra\s*bold|extrabold|semi\s*bold|semibold|"
    r"demi\s*bold|demibold|bold|medium|regular|book|light|thin|narrow|condensed|"
    r"expanded|oblique|italic)$",
    re.IGNORECASE,
)


_ITALIC_TOKENS = re.compile(r"\b(?:italic|oblique|cursive|kursiv)\b", re.IGNORECASE)


def _is_italic(subfamily: str) -> bool:
    """fallback slant test for faces whose OS/2 fsSelection is unset.

    Three separate frontend call sites were each sniffing the subfamily
    string for 'italic' on their own; the registry is the right place to
    answer it once.
    """
    return bool(_ITALIC_TOKENS.search(subfamily or ""))


def _normalize_family(raw: str) -> str:
    """Strip trailing weight/width qualifiers from a family name so that
    legacy name-ID-1 families like 'Arial Black' or 'Arial Narrow Bold'
    collapse into 'Arial' when no typographic family (ID 16) is present."""
    return _WEIGHT_TOKENS.sub("", raw).strip() or raw


@dataclass
class FontCoverage:
    font_path: str
    family: str
    subfamily: str = "Regular"       # name ID 2 (e.g. "Bold", "Light")
    weight_class: int = 400           # OS/2.usWeightClass
    codepoints: Set[int] = field(repr=False, default_factory=set)
    per_script: Dict[str, float] = field(default_factory=dict)  # script -> coverage %
    units_per_em: int = 1000
    advances: Dict[int, int] = field(repr=False, default_factory=dict)  # codepoint -> advance (font units)
    italic: bool = False              # OS/2.fsSelection bit 0, name fallback


class FontRegistry:
    """discovers fonts and caches per-font, per-script glyph coverage."""

    def __init__(
        self, font_library_path: Union[str, Sequence[str], None] = None,
    ):
        """One directory, several, or none.

        ``discover()`` was always additive -- calling it twice merges into
        the same dict -- the constructor simply never exposed that. Several
        roots is what lets bundled, uploaded and system fonts coexist
        without any of them knowing about the others.
        """
        self._fonts: Dict[str, FontCoverage] = {}
        if not font_library_path:
            return
        roots = (
            [font_library_path] if isinstance(font_library_path, (str, Path))
            else list(font_library_path)
        )
        for root in roots:
            if root:
                self.discover(str(root))

    # -- discovery ---------------------------------------------------------

    def discover(self, root: str) -> int:
        """recursively load fonts under root. returns count loaded."""
        if not HAVE_FONTTOOLS:
            return 0
        loaded = 0
        root_path = Path(root)
        candidates = [root_path] if root_path.is_file() else root_path.rglob("*")
        for p in candidates:
            if p.suffix.lower() in FONT_EXTS:
                loaded += self.load_font(str(p))
        return loaded

    def load_font(self, path: str) -> int:
        """load a single font file (or collection). returns faces loaded.

        Everything this needs is read eagerly below, so the file handle is
        released before returning.  It used to be left open: ``lazy=True``
        defers table parsing and keeps the file mapped, which meant
        discovering a system library held one OS handle per face for the
        life of the process -- 409 of them on this machine -- and Windows
        then refuses to delete or replace any font ToFU has ever looked at.
        That is invisible while fonts only arrive at startup and fatal the
        moment they can be installed and removed at runtime.
        """
        if not HAVE_FONTTOOLS:
            return 0
        p = Path(path)
        collection = None
        try:
            if p.suffix.lower() in {".ttc", ".otc"}:
                collection = TTCollection(str(p), lazy=True)
                faces = collection.fonts
            else:
                faces = [TTFont(str(p), lazy=True)]
        except Exception:
            return 0
        try:
            return self._absorb_faces(p, faces)
        finally:
            # A collection shares one reader across its faces, so close the
            # collection; standalone faces own theirs.
            if collection is not None:
                try:
                    collection.close()
                except Exception:
                    pass
            else:
                for font in faces:
                    try:
                        font.close()
                    except Exception:
                        pass

    def _absorb_faces(self, p: Path, faces: List[Any]) -> int:
        loaded = 0
        for i, font in enumerate(faces):
            key = str(p) if len(faces) == 1 else f"{p}#{i}"
            try:
                cmap = font.getBestCmap() or {}  # symbol fonts have no unicode cmap
                # Prefer typographic family (name ID 16) over the legacy
                # family (ID 1) so weight-specific families like "Arial Black"
                # or "Arial Narrow Bold" are grouped under their parent
                # family ("Arial") as weight variants in the dropdown.
                # When ID 16 is absent, strip trailing weight/width qualifiers
                # from ID 1 as a fallback normalization.
                raw_family = font["name"].getDebugName(1) or p.stem
                name = font["name"].getDebugName(16) or _normalize_family(raw_family)
                subfamily = font["name"].getDebugName(17) or font["name"].getDebugName(2) or "Regular"
            except Exception:
                continue
            units_per_em, advances = 1000, {}
            weight_class = 400
            italic = _is_italic(subfamily)
            try:
                units_per_em = font["head"].unitsPerEm
                hmtx = font["hmtx"]
                advances = {cp: hmtx[g][0] for cp, g in cmap.items() if g in hmtx.metrics}
                if "OS/2" in font:
                    weight_class = font["OS/2"].usWeightClass or 400
                    # fsSelection bit 0 is the foundry's own ITALIC flag and
                    # beats the name for faces called "Oblique", "Inclined"
                    # or nothing at all.  Keep the name verdict when the bit
                    # is clear: plenty of faces set neither.
                    italic = italic or bool(font["OS/2"].fsSelection & 0x01)
            except Exception:
                pass  # coverage still works without metrics
            self._fonts[key] = FontCoverage(
                font_path=key,
                family=name,
                subfamily=subfamily,
                weight_class=weight_class,
                codepoints=set(cmap.keys()),
                units_per_em=units_per_em,
                advances=advances,
                italic=italic,
            )
            loaded += 1
        return loaded

    @property
    def fonts(self) -> List[str]:
        return list(self._fonts.keys())

    @property
    def faces(self) -> Dict[str, FontCoverage]:
        """The discovered faces, keyed by path (``file.ttc#n`` for
        collection members).

        The public name for what a dozen call sites reach in and take as
        ``registry._fonts``. Returned live rather than copied: callers read
        it per region inside scoring loops, and a copy per call would be a
        real cost for no protection -- the leading underscore was never
        stopping anyone.
        """
        return self._fonts

    def unload_font(self, path: str) -> int:
        """Forget a face, or every face of a collection. Returns the count.

        Needed the moment fonts can be REMOVED at runtime: a deleted pack
        whose faces stay in the registry keeps offering families whose files
        are gone, and the next render fails at ``ImageFont.truetype`` rather
        than at the choice. Collections expand to ``path#0``, ``path#1`` …,
        so removing one path may remove several entries.
        """
        key = self._key(path)
        removed = 1 if self._fonts.pop(key, None) is not None else 0
        prefix = f"{key}#"
        for face_key in [k for k in self._fonts if k.startswith(prefix)]:
            del self._fonts[face_key]
            removed += 1
        return removed

    @staticmethod
    def _key(font_path: str) -> str:
        """normalize separators so 'C:/x/y.ttf' and 'C:\\x\\y.ttf' match;
        preserves '#n' face suffixes for collections."""
        if "#" in font_path:
            base, _, face = font_path.rpartition("#")
            return f"{Path(base)}#{face}"
        return str(Path(font_path))

    def resolve_key(self, font_path: Optional[str]) -> Optional[str]:
        """Registry key for a path, tolerating a collection named without a face.

        ``load_font`` keys the Nth face of a multi-face .ttc as ``file.ttc#n``,
        so a bare ``file.ttc`` matches no entry -- and every lookup through it
        then reports the font as covering NO glyphs, because the face it would
        have asked simply isn't there.  That is how an explicit ``simsun.ttc``
        pick came back "not covered" and was silently replaced by DFKai-SB: a
        TRADITIONAL Chinese face substituted for a simplified request, with
        coverage still reported as 1.0 afterwards so nothing looked wrong.
        Most CJK system fonts are collections, so this hit zh/ja/ko hardest.

        A bare collection path means "this collection", so resolve it to the
        first face.  Single-face collections are already keyed bare by
        ``load_font`` and are found by the exact-match branch.
        """
        if not font_path:
            return None
        key = self._key(font_path)
        if key in self._fonts:
            return key
        if "#" not in key and Path(key).suffix.lower() in {".ttc", ".otc"}:
            first = f"{key}#0"
            if first in self._fonts:
                return first
        return key

    # -- coverage & classification ------------------------------------------

    def coverage(
        self,
        font_path: str,
        script: str,
        extra_chars: Optional[Set[str]] = None,
    ) -> float:
        """fraction of the script's exhaustive sample (plus any
        language-required extras) present in the font's cmap."""
        fc = self._fonts.get(self._key(font_path))
        if fc is None:
            return 0.0
        if extra_chars is None and script in fc.per_script:
            return fc.per_script[script]
        sample = SCRIPT_SAMPLES.get(script, set()) | (extra_chars or set())
        if not sample:
            return 0.0
        hits = sum(1 for ch in sample if ord(ch) in fc.codepoints)
        score = hits / len(sample)
        if extra_chars is None:
            fc.per_script[script] = score
        return score

    def missing_required(self, font_path: str, lang: str) -> Set[str]:
        """language-required characters (e.g., vietnamese diacritics)
        absent from the font's cmap."""
        fc = self._fonts.get(self._key(font_path))
        required = LANG_REQUIRED.get(lang, set())
        if fc is None:
            return required
        return {ch for ch in required if ord(ch) not in fc.codepoints}

    def classify(
        self, font_path: str, script: str, lang: Optional[str] = None
    ) -> ScrptSpprt:
        """classify support: coverage thresholds + hard language requirements.
        a font missing any language-required character cannot be FULL."""
        extra = LANG_REQUIRED.get(lang or "", set())
        score = self.coverage(font_path, script, extra_chars=extra or None)
        if lang and self.missing_required(font_path, lang):
            return (
                ScrptSpprt.PARTIAL
                if score >= PARTIAL_THRESHOLD
                else ScrptSpprt.UNSUPPORTED
            )
        if score >= FULL_THRESHOLD:
            return ScrptSpprt.FULL
        if score >= PARTIAL_THRESHOLD:
            return ScrptSpprt.PARTIAL
        return ScrptSpprt.UNSUPPORTED

    def avg_advance_em(self, font_path: str, text: str) -> float:
        """mean advance width of text's characters in em units
        (advance / unitsPerEm). returns 0.0 if metrics are unavailable."""
        fc = self._fonts.get(self._key(font_path))
        if fc is None or not fc.advances:
            return 0.0
        widths = [fc.advances[ord(ch)] for ch in text if ord(ch) in fc.advances]
        if not widths:
            return 0.0
        return (sum(widths) / len(widths)) / fc.units_per_em

    def recommend(
        self, script: str, lang: Optional[str] = None, limit: int = 5
    ) -> List[Tuple[str, float]]:
        """ranked (font_path, coverage) for the frontend font picker,
        so users only choose among pre-verified fonts."""
        extra = LANG_REQUIRED.get(lang or "", set())
        scored = [
            (path, self.coverage(path, script, extra_chars=extra or None))
            for path in self._fonts
        ]
        return sorted(scored, key=lambda t: -t[1])[:limit]

    def families_with_weights(
        self, script: str, lang: Optional[str] = None, limit: Optional[int] = 12
    ) -> List[Dict[str, Any]]:
        """ranked font families, each with its available weights.

        returns a list of dicts:
          {family, best_path, best_coverage,
           weights: [{path, subfamily, weight_class, italic, coverage}]}
        families are ranked by their best-coverage face; within a family,
        weights are sorted by weight_class ascending.

        ``limit=None`` returns the entire library, which is what the Font
        Manager asks for.  That is only affordable because every face is
        scored EXACTLY ONCE below: coverage() memoizes into per_script only
        when extra_chars is None, so for any language with LANG_REQUIRED
        entries (vietnamese diacritics, indic conjuncts) a repeated call
        re-walks the whole script sample every time.
        """
        extra = LANG_REQUIRED.get(lang or "", set())
        scores: Dict[str, float] = {}
        by_family: Dict[str, List[FontCoverage]] = {}
        for fc in self._fonts.values():
            scores[fc.font_path] = self.coverage(
                fc.font_path, script, extra_chars=extra or None
            )
            by_family.setdefault(fc.family, []).append(fc)

        # rank families by max coverage among their faces
        ranked = sorted(
            by_family.items(),
            key=lambda kv: max(scores[fc.font_path] for fc in kv[1]),
            reverse=True,
        )
        if limit is not None:
            ranked = ranked[:limit]

        result = []
        for family, faces in ranked:
            face_data = sorted(
                faces,
                key=lambda fc: fc.weight_class,
            )
            weights = []
            best_cov = 0.0
            best_path = None
            for fc in face_data:
                cov = scores[fc.font_path]
                if cov > best_cov:
                    best_cov = cov
                    best_path = fc.font_path
                weights.append({
                    "path": fc.font_path,
                    "subfamily": fc.subfamily,
                    "weight_class": fc.weight_class,
                    # `or` so a FontCoverage built by hand (tests, injected
                    # registries) still reports slant from its subfamily
                    "italic": fc.italic or _is_italic(fc.subfamily),
                    "coverage": round(cov, 4),
                })
            if best_path is None and face_data:
                best_path = face_data[0].font_path
            result.append({
                "family": family,
                "best_path": best_path,
                "best_coverage": round(best_cov, 4),
                "weights": weights,
            })
        return result
