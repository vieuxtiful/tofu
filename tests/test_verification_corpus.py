import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_verification_corpus.py"
SPEC = ROOT / "scripts" / "verification_corpus.json"


def _module():
    spec = importlib.util.spec_from_file_location("eval_verification_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_corpus_declares_required_expectations():
    corpus = _module().load_spec(SPEC)
    # The four image waves are 15 cases; wave 5 varies only the target
    # language over an existing layout, so it adds cases without fixtures.
    assert len(corpus["cases"]) >= 15
    assert {case["wave"] for case in corpus["cases"]} == {
        "wave-1-simple-latin",
        "wave-2-dense-layouts",
        "wave-3-difficult-rendering",
        "wave-4-multilingual-scale",
        "wave-5-language-matrix",
        "wave-6-perspective",
    }
    for case in corpus["cases"]:
        assert case["expected_regions"] > 0
        assert case["target_language"]
        assert case["asset_class"]
        assert case["expected_outcome"] in {"pass", "review", "fail"}


def test_language_matrix_covers_every_supported_target():
    """Wave 5 exists so no supported target language is verified only by unit
    tests.  Before it, the corpus had end-to-end evidence for en/fr/de/ja/ar
    and none at all for the other seven."""
    corpus = _module().load_spec(SPEC)
    covered = {case["target_language"] for case in corpus["cases"]}
    required = {"fr", "it", "es", "de", "pl", "ru", "zh-cn", "zh-hk", "ko", "ja"}
    assert required <= covered, f"no corpus case targets {sorted(required - covered)}"


def test_language_matrix_pins_the_orthography_negatives():
    """The zh-cn/zh-hk swap is invisible to every other check: both fonts
    carry the glyphs, OCR reads it perfectly, and Unicode calls both Hani.
    Only the han_variant comparison can fail it, so it is pinned here."""
    corpus = _module().load_spec(SPEC)
    by_id = {case["id"]: case for case in corpus["cases"]}
    for case_id in ("negative-zh-cn-traditional", "negative-zh-hk-simplified",
                    "negative-untranslated-cyrillic"):
        assert by_id[case_id]["expected_outcome"] == "review"


def test_full_corpus_ground_truth_builds_expected_manifest():
    module = _module()
    corpus = module.load_spec(SPEC)
    for case in corpus["cases"]:
        manifest = module.build_manifest(case)
        assert manifest.total_regions == case["expected_regions"]
        assert len(manifest.instances) == case["expected_regions"]
        assert manifest.targ_lang == case["target_language"]
        assert manifest.asset_class == case["asset_class"]
        assert all(instance.target_text for instance in manifest.instances)


def test_multilingual_cases_preserve_declared_targets():
    module = _module()
    corpus = module.load_spec(SPEC)
    translated = [case for case in corpus["cases"] if case.get("targets")]
    assert translated
    for case in translated:
        manifest = module.build_manifest(case)
        assert [instance.target_text for instance in manifest.instances] == case["targets"]
