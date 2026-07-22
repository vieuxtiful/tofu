from tofu.core.types import BBox, InstText, SceneRegion, TextManifest
from tofu.layers import basil
from tofu.utils.manifest_store import load_manifest, save_manifest


def _rue_vieux_manifest() -> TextManifest:
    # IDs deliberately differ from their lower-line visual order: MURS was
    # allocated r2, while VIEUX is physically to its left as r3.
    instances = [
        InstText("r1", BBox(228, 109, 235, 71), text="Rue des", reading_order=0),
        InstText("r2", BBox(281, 206, 175, 72), text="MURS", reading_order=1),
        InstText("r3", BBox(89, 209, 176, 69), text="VIEUX", reading_order=2),
    ]
    return TextManifest(
        asset_id="rue-vieux", total_regions=3, instances=instances,
        src_lang="fr", targ_lang="it",
        scene_regions=[SceneRegion(BBox(36, 69, 493, 253), "bordered_region", 0.9)],
    )


def test_basil_registers_visual_source_order_not_raw_region_ids():
    manifest = _rue_vieux_manifest()

    units = basil.unify_manifest(manifest)

    assert len(units) == 1
    assert units[0].region_ids == ["r1", "r3", "r2"]
    assert units[0].source_text == "Rue des VIEUX MURS"
    assert units[0].entity_type == "street_name"
    assert units[0].semantic_roles == {
        "r1": "street_designator", "r3": "modifier", "r2": "head",
    }


def test_basil_assigns_italian_grammar_to_immutable_spatial_anchors():
    manifest = _rue_vieux_manifest()
    unit = basil.unify_manifest(manifest)[0]
    original_boxes = {inst.id: inst.bounding_box for inst in manifest.instances}

    plan = basil.plan_substitution(manifest, unit.id, "Via dei Muri Vecchi", "it")

    assert plan["source_region_order"] == ["r1", "r3", "r2"]
    assert plan["target_region_order"] == ["r1", "r2", "r3"]
    assert {item["region_id"]: item["text"] for item in plan["assignments"]} == {
        "r1": "Via dei", "r2": "Muri", "r3": "Vecchi",
    }
    assert {item["region_id"]: item["anchor_id"] for item in plan["assignments"]} == {
        "r1": "r1", "r2": "r3", "r3": "r2",
    }
    assert plan["review_required"] is False

    basil.apply_substitution(manifest, plan, "it")

    assert {inst.id: inst.target_text for inst in manifest.instances} == {
        "r1": "Via dei", "r2": "Vecchi", "r3": "Muri",
    }
    assert {inst.id: inst.bounding_box for inst in manifest.instances} == original_boxes
    assert manifest.semantic_units[0].substitution["target_region_order"] == ["r1", "r2", "r3"]
    assert manifest.instances[1].semantic_assignment["geometry_unchanged"] is True
    assert manifest.instances[1].semantic_assignment["semantic_region_id"] == "r3"
    assert basil.plated_texts(manifest) == {
        "r1": "Via dei", "r3": "Muri", "r2": "Vecchi",
    }


def test_basil_fails_open_when_no_verified_alignment_exists():
    manifest = _rue_vieux_manifest()
    unit = basil.unify_manifest(manifest)[0]

    plan = basil.plan_substitution(manifest, unit.id, "Old Walls Street", "en")

    assert plan["assignments"] == []
    assert plan["review_required"] is True
    assert plan["method"] == "manual_required"
    assert all(inst.target_text is None for inst in manifest.instances)


def test_basil_semantic_provenance_round_trips(tmp_path):
    manifest = _rue_vieux_manifest()
    unit = basil.unify_manifest(manifest)[0]
    plan = basil.plan_substitution(manifest, unit.id, "Via dei Muri Vecchi", "it")
    basil.apply_substitution(manifest, plan, "it")

    save_manifest(tmp_path, manifest.asset_id, manifest)
    restored = load_manifest(tmp_path, manifest.asset_id)

    assert restored is not None
    assert restored.semantic_units[0].source_text == "Rue des VIEUX MURS"
    assert restored.semantic_units[0].substitution["target_text"] == "Via dei Muri Vecchi"
    assert restored.instances[2].semantic_assignment["unit_id"] == "u1"


def test_basil_migrates_legacy_target_order_into_spatial_cubes():
    manifest = _rue_vieux_manifest()
    unit = basil.unify_manifest(manifest)[0]
    plan = basil.plan_substitution(manifest, unit.id, "Via dei Muri Vecchi", "it")
    # Simulate a plan persisted before anchor_id existed: semantic mapping is
    # present, but target text is still projected to source rN identities.
    legacy = {**plan, "assignments": [{key: value for key, value in item.items() if key != "anchor_id"} for item in plan["assignments"]]}
    manifest.semantic_units[0].substitution = {**legacy, "applied": True}
    for assignment in legacy["assignments"]:
        next(inst for inst in manifest.instances if inst.id == assignment["region_id"]).target_text = assignment["text"]

    assert basil.migrate_legacy_plating(manifest) is True
    assert basil.plated_texts(manifest) == {"r1": "Via dei", "r3": "Muri", "r2": "Vecchi"}
    assert {inst.id: inst.target_text for inst in manifest.instances} == {
        "r1": "Via dei", "r2": "Vecchi", "r3": "Muri",
    }
