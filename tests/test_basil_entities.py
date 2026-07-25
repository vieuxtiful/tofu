"""Basil's evidence-gated entity grouping, on the japan-subs geometry.

Coordinates, texts, confidences and detected languages are the real values
from the japan-subs manifest.  Before this gate existed, these eighteen
regions produced three units, one of which merged ten regions across three
physically separate signs -- including a Japanese line and its own romaji
transliteration -- into a single space-joined "sentence".
"""

from tofu.core.types import BBox, InstText, TextManifest
from tofu.layers import basil


def _inst(region_id, box, text, lang, confidence):
    return InstText(region_id, BBox(*box), text=text,
                    detected_language=lang, confidence=confidence)


def _japan_subs_manifest() -> TextManifest:
    instances = [
        _inst("r1", (257, 45, 50, 28), "御嶽", "ja", 0.984),
        _inst("r2", (199, 48, 34, 15), "岐阜の", "ja", 0.927),
        _inst("r3", (77, 65, 169, 38), "小坂み滝めぐり", "ja", 0.925),
        # 歓迎 fragmented into one region per glyph, the first misrecognised
        # and flagged low-confidence by the recognizer itself.
        _inst("r4", (280, 70, 24, 22), "欲", "ja", 0.631),
        _inst("r5", (299, 75, 129, 38), "小坂温泉郷", "ja", 0.999),
        _inst("r6", (249, 83, 54, 25), "迎", "ja", 0.972),
        # a Japanese line stacked directly over its own romanisation
        _inst("r7", (617, 85, 212, 46), "ひだおさか", "ja", 0.999),
        _inst("r8", (666, 126, 112, 24), "飛驒小坂", "ja", 0.956),
        _inst("r10", (679, 151, 86, 18), "Hida-osaka", "en", 0.999),
        _inst("r11", (565, 169, 68, 18), "ひだみやだ", "ja", 0.981),
        _inst("r14", (563, 181, 70, 17), "Hida-miyada", "en", 0.982),
        _inst("r12", (821, 170, 60, 20), "なぎさ", "ja", 0.917),
        _inst("r15", (824, 188, 39, 11), "Nagisa", "en", 0.999),
    ]
    return TextManifest(
        asset_id="japan-subs", total_regions=len(instances), instances=instances,
        src_lang="ja", targ_lang="zh-cn", img_dim=(940, 330),
    )


def test_no_unit_mixes_a_language_with_its_own_romanisation():
    manifest = _japan_subs_manifest()
    by_id = {inst.id: inst for inst in manifest.instances}

    units = basil.unify_manifest(manifest)

    for unit in units:
        languages = {by_id[region_id].detected_language for region_id in unit.region_ids}
        assert len(languages) == 1, f"{unit.id} mixes {languages}"


def test_cjk_source_text_is_never_space_joined():
    manifest = _japan_subs_manifest()

    units = basil.unify_manifest(manifest)

    for unit in units:
        assert " " not in unit.source_text, f"{unit.id} spaced a CJK unit like English"


def test_fragmented_entity_is_reassembled_and_its_misread_only_proposed():
    manifest = _japan_subs_manifest()
    original_text = {inst.id: inst.text for inst in manifest.instances}

    units = basil.unify_manifest(manifest)
    reassembled = [unit for unit in units if set(unit.region_ids) == {"r4", "r6"}]

    assert len(reassembled) == 1
    unit = reassembled[0]
    assert unit.entity_type == "gazetteer_entity"
    assert unit.ocr_repair["read"] == "欲迎"
    assert unit.ocr_repair["proposed"] == "歓迎"
    assert unit.ocr_repair["accepted"] is False
    assert unit.review_required is True
    # The proposal must never reach the regions until the user says so.
    assert unit.source_text == "欲迎"
    assert {inst.id: inst.text for inst in manifest.instances} == original_text


def test_accepting_a_repair_rewrites_only_the_unit_never_the_regions():
    manifest = _japan_subs_manifest()
    unit = next(u for u in basil.unify_manifest(manifest) if set(u.region_ids) == {"r4", "r6"})
    boxes = {inst.id: inst.bounding_box for inst in manifest.instances}

    basil.accept_repair(manifest, unit.id, True)

    assert unit.source_text == "歓迎"
    assert unit.ocr_repair["accepted"] is True
    assert unit.review_required is False
    assert {inst.id: inst.text for inst in manifest.instances}["r4"] == "欲"
    assert {inst.id: inst.bounding_box for inst in manifest.instances} == boxes

    basil.accept_repair(manifest, unit.id, False)
    assert unit.source_text == "欲迎"


def test_a_pair_that_cannot_reorder_registers_only_lexicon_evidence():
    # ja->zh is "unnecessary", so proximity and panel cohesion alone must
    # not manufacture a plating decision; only the gazetteer entity earns
    # a unit.  Ten unrelated regions across three signs earn nothing.
    manifest = _japan_subs_manifest()

    units = basil.unify_manifest(manifest)

    assert [unit.entity_type for unit in units] == ["gazetteer_entity"]
    assert units[0].pairing["verdict"] == "unnecessary"


def test_suggestion_orders_existing_targets_by_target_syntax():
    from tofu.core.types import SceneRegion

    manifest = TextManifest(
        asset_id="rue-vieux", total_regions=3,
        instances=[
            InstText("r1", BBox(228, 109, 235, 71), text="Rue des", target_text="Via dei"),
            InstText("r2", BBox(281, 206, 175, 72), text="MURS", target_text="MURI"),
            InstText("r3", BBox(89, 209, 176, 69), text="VIEUX", target_text="VECCHI"),
        ],
        src_lang="fr", targ_lang="it",
        scene_regions=[SceneRegion(BBox(36, 69, 493, 253), "bordered_region", 0.9)],
    )

    unit = basil.unify_manifest(manifest)[0]

    # Italian puts the adjective after its noun, so the head (MURI) comes
    # before the modifier (VECCHI) even though the source reads the other way.
    assert unit.suggestion["target_text"] == "Via dei MURI VECCHI"
    assert unit.suggestion["region_order"] == ["r1", "r2", "r3"]
    assert unit.suggestion["coverage"] == 1.0
