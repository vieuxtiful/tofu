## 🍢 ToFU — font registry
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tofu.core.types import ScrptSpprt
from tofu.layers.script_samples import SCRIPT_SAMPLES, LANG_REQUIRED

try:
    from fontTools.ttLib import TTFont, TTCollection
    HAVE_FONTTOOLS = True
except ImportError:  # registry degrades gracefully; tofu falls back to static map
    HAVE_FONTTOOLS = False

FONT_EXTS = {".ttf", ".otf", ".ttc", ".otc"}
FULL_THRESHOLD = 0.995    # tolerate trivial gaps in range-derived samples
PARTIAL_THRESHOLD = 0.80


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


class FontRegistry:
    """discovers fonts and caches per-font, per-script glyph coverage."""

    def __init__(self, font_library_path: Optional[str] = None):
        self._fonts: Dict[str, FontCoverage] = {}
        if font_library_path:
            self.discover(font_library_path)

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
        """load a single font file (or collection). returns faces loaded."""
        if not HAVE_FONTTOOLS:
            return 0
        p = Path(path)
        try:
            if p.suffix.lower() in {".ttc", ".otc"}:
                faces = TTCollection(str(p), lazy=True).fonts
            else:
                faces = [TTFont(str(p), lazy=True)]
        except Exception:
            return 0
        loaded = 0
        for i, font in enumerate(faces):
            key = str(p) if len(faces) == 1 else f"{p}#{i}"
            try:
                cmap = font.getBestCmap() or {}  # symbol fonts have no unicode cmap
                name = font["name"].getDebugName(1) or p.stem
                subfamily = font["name"].getDebugName(2) or "Regular"
            except Exception:
                continue
            units_per_em, advances = 1000, {}
            weight_class = 400
            try:
                units_per_em = font["head"].unitsPerEm
                hmtx = font["hmtx"]
                advances = {cp: hmtx[g][0] for cp, g in cmap.items() if g in hmtx.metrics}
                if "OS/2" in font:
                    weight_class = font["OS/2"].usWeightClass or 400
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
            )
            loaded += 1
        return loaded

    @property
    def fonts(self) -> List[str]:
        return list(self._fonts.keys())

    @staticmethod
    def _key(font_path: str) -> str:
        """normalize separators so 'C:/x/y.ttf' and 'C:\\x\\y.ttf' match;
        preserves '#n' face suffixes for collections."""
        if "#" in font_path:
            base, _, face = font_path.rpartition("#")
            return f"{Path(base)}#{face}"
        return str(Path(font_path))

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
        self, script: str, lang: Optional[str] = None, limit: int = 12
    ) -> List[Dict[str, Any]]:
        """ranked font families, each with its available weights.

        returns a list of dicts:
          {family, best_path, best_coverage, weights: [{path, subfamily, weight_class, coverage}]}
        families are ranked by their best-coverage face; within a family,
        weights are sorted by weight_class ascending.
        """
        extra = LANG_REQUIRED.get(lang or "", set())
        # group by family
        by_family: Dict[str, List[FontCoverage]] = {}
        for fc in self._fonts.values():
            cov = self.coverage(fc.font_path, script, extra_chars=extra or None)
            by_family.setdefault(fc.family, []).append(fc)

        # rank families by max coverage among their faces
        ranked = sorted(
            by_family.items(),
            key=lambda kv: max(
                self.coverage(fc.font_path, script, extra_chars=extra or None)
                for fc in kv[1]
            ),
            reverse=True,
        )[:limit]

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
                cov = self.coverage(fc.font_path, script, extra_chars=extra or None)
                if cov > best_cov:
                    best_cov = cov
                    best_path = fc.font_path
                weights.append({
                    "path": fc.font_path,
                    "subfamily": fc.subfamily,
                    "weight_class": fc.weight_class,
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
