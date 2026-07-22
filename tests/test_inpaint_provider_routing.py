import json
import sys

from tofu.core.types import BBox, BgProfil, InstText, SceneRegion
from tofu.layers import inpaint_providers


def _inst(texture):
    return InstText("r1", BBox(0, 0, 20, 10), background_profile=BgProfil(texture=texture))


def _surface(texture, label="panel"):
    return SceneRegion(BBox(0, 0, 20, 10), label, 1.0, texture=texture)


def test_agreed_flat_is_only_high_confidence_auto_accept_path():
    route = inpaint_providers.route(_inst("flat"), _surface("flat"))
    assert route.provider == "analytic"
    assert route.auto_accept is True
    assert route.review_required is False


def test_unknown_or_textured_surface_never_silently_auto_accepts_without_promoted_model(monkeypatch, tmp_path):
    monkeypatch.delenv("TOFU_LAMA_ENABLED", raising=False)
    # This assertion exercises the unprovisioned deployment contract.  A
    # developer's real local provider config must not change its outcome.
    monkeypatch.setenv("TOFU_INPAINT_CONFIG", str(tmp_path / "missing.json"))
    route = inpaint_providers.route(_inst("textured"), _surface("textured", "surface"))
    assert route.auto_accept is False
    assert route.review_required is True
    assert route.provider == "telea_fallback"


def test_cleanse_persists_fallback_review_evidence():
    from PIL import Image
    from tofu.core.types import TextManifest
    from tofu.layers import cleanse
    inst = _inst("textured")
    manifest = TextManifest("a", 1, [inst], scene_regions=[_surface("textured", "surface")])
    cleanse.erase(Image.new("RGB", (30, 20), "white"), manifest)
    assert inst.repair_provenance is not None
    assert inst.repair_provenance["review_required"] is True
    assert inst.repair_provenance["executed_provider"] == "telea_fallback"


def test_provider_contract_exposes_self_hosted_and_experimental_paths():
    providers = {provider["id"]: provider for provider in inpaint_providers.provider_statuses()}
    assert {"analytic", "lama", "diffstr_experimental", "brushnet_experimental", "manual"} <= set(providers)
    assert providers["analytic"]["promoted"] is True
    assert providers["diffstr_experimental"]["review_required"] is True


def test_quality_gate_accepts_context_compatible_candidate_and_rejects_rewrite():
    import numpy as np
    before = np.full((24, 24, 3), 125, dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=bool); mask[8:16, 8:16] = True
    accepted, evidence = inpaint_providers.quality_gate(before, before.copy(), mask)
    assert accepted is True and evidence["score"] >= .88
    rewritten = before.copy(); rewritten[:] = 10
    accepted, evidence = inpaint_providers.quality_gate(before, rewritten, mask)
    assert accepted is False and evidence["outside_delta"] > 1


def test_official_lama_config_activates_the_local_router(tmp_path, monkeypatch):
    repo = tmp_path / "lama"; (repo / "bin").mkdir(parents=True)
    (repo / "bin" / "predict.py").write_text("# fixture", encoding="utf-8")
    model = tmp_path / "big-lama"; model.mkdir()
    config = tmp_path / "providers.json"
    config.write_text(json.dumps({"schema": 1, "providers": {"lama": {
        "enabled": True, "promoted": False, "runtime": "official_lama",
        "python": sys.executable, "repo_dir": str(repo), "model_path": str(model),
    }}}), encoding="utf-8")
    monkeypatch.setenv("TOFU_INPAINT_CONFIG", str(config))
    spec = inpaint_providers.provider_spec("lama")
    route = inpaint_providers.route(_inst("textured"), _surface("textured", "surface"))
    assert spec.available is True
    assert route.provider == "lama" and route.review_required is True


def test_torchscript_lama_config_requires_the_isolated_runtime_and_weight(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"; python.touch()
    model = tmp_path / "big-lama.pt"; model.touch()
    config = tmp_path / "providers.json"
    config.write_text(json.dumps({"schema": 1, "providers": {"lama": {
        "enabled": True, "promoted": False, "runtime": "torchscript_lama",
        "python": str(python), "model_path": str(model),
    }}}), encoding="utf-8")
    monkeypatch.setenv("TOFU_INPAINT_CONFIG", str(config))
    spec = inpaint_providers.provider_spec("lama")
    assert spec.available is True
    assert spec.runtime == "torchscript_lama"


def test_external_provider_uses_the_isolated_file_contract(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("from pathlib import Path\nimport shutil, sys\nshutil.copy2(sys.argv[1], sys.argv[2])\n", encoding="utf-8")
    config = tmp_path / "providers.json"
    config.write_text(json.dumps({"schema": 1, "providers": {"diffstr_experimental": {
        "enabled": True, "promoted": False, "runtime": "external_command",
        "command": [sys.executable, str(worker), "{input}", "{output}"],
        "required_paths": [str(worker)],
    }}}), encoding="utf-8")
    monkeypatch.setenv("TOFU_INPAINT_CONFIG", str(config))
    import numpy as np
    image = np.full((12, 12, 3), 100, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool); mask[3:8, 3:8] = True
    outcome = inpaint_providers.repair("diffstr_experimental", image, mask)
    assert outcome.error is None
    assert np.array_equal(outcome.image, image)
