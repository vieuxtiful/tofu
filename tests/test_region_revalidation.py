## 🍢 per-region preflight re-validation: one finding, one issue
"""Regression cover for the duplicated-issue defect.

Most of what ToFU reports during region re-validation is a fact about a
(language, font, size, effects) combination, not about one box. The old
loop validated once per INSTANCE, so a scene with N regions in one
language produced N identical warnings, each stamped with a different
region id — burying any genuinely region-scoped finding among copies.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

pytest.importorskip("fastapi")

from tofu.core.types import (  # noqa: E402
    BBox, CharactText, InstText, StyleProfil, TextManifest, VldtnClass,
    VldtnSeverity,
)


@pytest.fixture(scope="module")
def revalidate():
    main = pytest.importorskip("main")
    return main._revalidate_regions


class _Validator:
    """Stands in for ToFU: records every call and always returns one
    language-level issue plus one region-scoped issue."""

    def __init__(self):
        self.calls = []

    def validate(self, path, lang, context=None, manifest=None):
        self.calls.append((lang, tuple(sorted((context or {}).items()))))
        from tofu.core.types import VldtnReport
        return VldtnReport(
            passed=True,
            issues=[
                VldtnClass(
                    severity=VldtnSeverity.WARNING, code="ToFU_003",
                    message=f"glyph segmentation may be unreliable for {lang}.",
                ),
                VldtnClass(
                    severity=VldtnSeverity.WARNING, code="ToFU_005",
                    message="tight", region_id="r-anchored",
                ),
            ],
        )


def _inst(rid, size=None, shadow=False, lang=None):
    return InstText(
        id=rid,
        bounding_box=BBox(x=0, y=0, width=100, height=30),
        text="src", target_text="tgt", target_language=lang,
        characteristics=CharactText(size=size) if size else None,
        style_profile=StyleProfil(shadow={"blur": 2}) if shadow else None,
    )


def _manifest(instances):
    return TextManifest(
        asset_id="a", total_regions=len(instances), instances=instances
    )


class TestIssueDeduplication:
    def test_identical_context_validates_once(self, revalidate):
        v = _Validator()
        manifest = _manifest([_inst(f"r{i}", size=20) for i in range(1, 31)])

        issues, langs, ctx_count = revalidate(v, "a.png", manifest, "ja", None)

        # 30 regions, one shared typography context -> one validate call
        assert len(v.calls) == 1
        assert ctx_count == 1
        assert langs == ["ja"]
        # ...and one copy of the language-level finding, not 30
        assert sum(1 for i in issues if i.code == "ToFU_003") == 1

    def test_language_level_issue_carries_every_contributing_region(self, revalidate):
        v = _Validator()
        manifest = _manifest([_inst("r1", size=20), _inst("r2", size=20)])

        issues, _, _ = revalidate(v, "a.png", manifest, "ja", None)

        seg = next(i for i in issues if i.code == "ToFU_003")
        assert seg.region_ids == ["r1", "r2"]
        assert seg.region_id == "r1"  # primary anchor kept for old consumers

    def test_region_scoped_issue_keeps_its_own_anchor(self, revalidate):
        v = _Validator()
        manifest = _manifest([_inst("r1", size=20), _inst("r2", size=20)])

        issues, _, _ = revalidate(v, "a.png", manifest, "ja", None)

        anchored = next(i for i in issues if i.code == "ToFU_005")
        assert anchored.region_id == "r-anchored"
        assert anchored.region_ids == ["r-anchored"]

    def test_distinct_typography_still_validated_separately(self, revalidate):
        # different font_px / effects are genuinely different questions
        v = _Validator()
        manifest = _manifest([
            _inst("r1", size=20),
            _inst("r2", size=8),
            _inst("r3", size=20, shadow=True),
        ])

        _, _, ctx_count = revalidate(v, "a.png", manifest, "ja", None)

        assert ctx_count == 3
        assert len(v.calls) == 3

    def test_per_region_target_language_override_is_honoured(self, revalidate):
        v = _Validator()
        manifest = _manifest([
            _inst("r1", size=20),
            _inst("r2", size=20, lang="ko"),
        ])

        issues, langs, _ = revalidate(v, "a.png", manifest, "ja", None)

        assert sorted(langs) == ["ja", "ko"]
        # the two languages produce genuinely different findings, so both
        # survive the merge rather than collapsing into one
        assert sum(1 for i in issues if i.code == "ToFU_003") == 2

    def test_untranslated_and_dnt_regions_are_skipped(self, revalidate):
        v = _Validator()
        translated = _inst("r1", size=20)
        untranslated = _inst("r2", size=20)
        untranslated.target_text = None
        dnt = _inst("r3", size=20)
        dnt.dnt = True

        _, _, ctx_count = revalidate(
            v, "a.png", _manifest([translated, untranslated, dnt]), "ja", None
        )

        assert ctx_count == 1

    def test_nothing_to_validate_makes_no_calls(self, revalidate):
        v = _Validator()
        assert revalidate(v, "a.png", _manifest([]), "ja", None) == ([], [], 0)
        assert v.calls == []
