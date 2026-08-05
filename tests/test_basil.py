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


def _rue_des_martyrs_ko_manifest() -> TextManifest:
    # rue-des-martyrs: a single blue enamel plaque split into three regions:
    # r1 "RUE", r2 "DES", r3 "MARTYRS".  Korean signage puts the designator
    # (로/길) as a suffix of the head, so the target order is r3 then r2 then r1.
    instances = [
        InstText("r1", BBox(530, 62, 61, 45), text="RUE", reading_order=0),
        InstText("r2", BBox(489, 111, 43, 39), text="DES", reading_order=1),
        InstText("r3", BBox(527, 89, 129, 59), text="MARTYRS", reading_order=2),
    ]
    manifest = TextManifest(
        asset_id="rue-des-martyrs", total_regions=3, instances=instances,
        src_lang="fr", targ_lang="ko-KR",
        scene_regions=[SceneRegion(BBox(480, 55, 180, 100), "bordered_region", 0.9)],
    )
    # Per-region Korean translations entered in the Translate table.  The
    # SlotEditor concatenates these in the user's chosen order to form the
    # target phrase, so the plan must recover that arrangement.
    by_id = {inst.id: inst for inst in manifest.instances}
    by_id["r1"].target_text = "르"
    by_id["r2"].target_text = "데"
    by_id["r3"].target_text = "마르티르"
    return manifest


def test_basil_plates_unspaced_korean_target_via_slot_arrangement():
    manifest = _rue_des_martyrs_ko_manifest()
    unit = basil.unify_manifest(manifest)[0]
    # The SlotEditor arranges r3 (head) before r1 (designator), which is the
    # Korean typology order suggest_plating should propose.
    phrase = "마르티르데르"

    plan = basil.plan_substitution(manifest, unit.id, phrase, "ko-KR")

    assert plan["review_required"] is False
    assert plan["method"] == "manual_slot_arrangement"
    assert plan["assignments"], "expected a plating plan for an unspaced target"
    assert {item["region_id"]: item["text"] for item in plan["assignments"]} == {
        "r1": "르", "r2": "데", "r3": "마르티르",
    }
    # The head (r3) is plated into the first cube, the designator (r1) into the last.
    assert plan["target_region_order"] == ["r3", "r2", "r1"]
    assert {item["region_id"]: item["anchor_id"] for item in plan["assignments"]} == {
        "r3": "r1", "r2": "r2", "r1": "r3",
    }

    basil.apply_substitution(manifest, plan, "ko-KR")
    assert basil.plated_texts(manifest) == {"r1": "마르티르", "r2": "데", "r3": "르"}


def test_basil_unspaced_target_requires_all_regions_translated():
    manifest = _rue_des_martyrs_ko_manifest()
    manifest.instances[1].target_text = None  # erase r2's translation
    unit = basil.unify_manifest(manifest)[0]

    plan = basil.plan_substitution(manifest, unit.id, "마르티르데르", "ko-KR")

    assert plan["assignments"] == []
    assert plan["review_required"] is True
    assert any("missing target text" in warning for warning in plan["warnings"])


def test_basil_unspaced_target_rejects_phrase_not_matching_any_arrangement():
    manifest = _rue_des_martyrs_ko_manifest()
    unit = basil.unify_manifest(manifest)[0]
    # A phrase that is not any concatenation of the per-region translations.
    plan = basil.plan_substitution(manifest, unit.id, "불일치문구", "ko-KR")

    assert plan["assignments"] == []
    assert plan["review_required"] is True
    assert any("does not match any arrangement" in warning for warning in plan["warnings"])
