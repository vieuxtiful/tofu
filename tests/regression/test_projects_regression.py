"""Regression test over the REAL project corpus in images/.

tests/fixtures holds synthetic, single-defect images. This harness runs the
same metrics over the photographs and posters that real projects were built
from -- street plaques, billboards, protest posters, a lithograph -- where the
failures actually surface: rotated enamel signage, Cyrillic slogans, ornamental
display type, textured masonry backgrounds.

Detection is run language-tuned, the way Capture runs it: the project declares a
source language and cicerone gets it as a reader hint. The hint is taken from
the ground truth's own `lang` fields so the corpus stays self-describing -- add
an asset with its .gt.json and it is measured, with no second table to update.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare  # noqa: E402

CORPUS_DIR = ROOT / "images"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _fixture_names() -> list[str]:
    # NOT gt.stem: Path strips only the last suffix, so "rue-des-martyrs.gt.json"
    # would yield "rue-des-martyrs.gt" and never resolve to an image.
    return sorted(gt.name.removesuffix(".gt.json") for gt in CORPUS_DIR.glob("*.gt.json"))


def _image_for(fixture: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = CORPUS_DIR / f"{fixture}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _declared_language(gt_path: Path) -> str | None:
    """Dominant ground-truth language, used as the reader hint.

    Mixed-script assets are normal here -- the russian-billboard slogans set a
    Latin "Za" against Cyrillic -- so the MAJORITY language is the project
    source, not the only one present.
    """
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    langs = [r.get("lang") for r in data.get("regions", []) if r.get("lang")]
    if not langs:
        return None
    return Counter(langs).most_common(1)[0][0]


def _project_metrics(fixture: str) -> dict[str, float]:
    from tofu.layers import cicerone, scene
    from eval_detect import evaluate as detect_evaluate, _font_registry

    image_path = _image_for(fixture)
    gt_path = CORPUS_DIR / f"{fixture}.gt.json"
    if image_path is None:
        pytest.skip(f"corpus image missing for: {fixture}")

    languages = _declared_language(gt_path)
    try:
        scene_regions = scene.analyze_regions(str(image_path))
    except Exception:
        scene_regions = []
    manifest = cicerone.detect(
        str(image_path),
        scene_regions=scene_regions,
        languages=[languages] if languages else None,
        font_registry=_font_registry(),
    )
    metrics = detect_evaluate(image_path, manifest, gt_path, scene_regions)
    result: dict[str, float] = {}
    # precision/f1 are None for partial ground truth, where un-annotated
    # background text must not count as a false positive.
    for key in ("precision", "recall", "f1"):
        if metrics.get(key) is not None:
            result[key] = float(metrics[key])
    result["mean_norm_ed"] = float(metrics["mean_norm_ed"])
    return result


@pytest.mark.parametrize("fixture", _fixture_names())
def test_projects_regression(fixture: str) -> None:
    """Real-project detection metrics must stay within baseline tolerances."""
    metrics = _project_metrics(fixture)
    violations = compare("projects", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Project-corpus regression detected:\n{msgs}")
