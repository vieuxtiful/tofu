from pathlib import Path

from tofu.core.types import BBox, InstText, TextManifest
from tofu.layers import basil
from tofu.utils.glossary import merge_glossaries, parse_glossary, to_basil_lexicon


def test_csv_preserves_full_locale_and_builds_lexicon():
    parsed = parse_glossary("brand.csv", b"source,target,src_lang,targ_lang\nRue,Via,en-US,en-EU\n")
    assert parsed.entry_count == 1
    assert parsed.language_pairs == [("en-US", "en-EU")]
    assert to_basil_lexicon(parsed)[("en-US", "en-EU")]["rue"] == "Via"


def test_tbx_is_namespace_agnostic():
    raw = b'''<martif xmlns="urn:iso:std:iso:30042:ed-2"><termEntry><langSet xml:lang="fr-FR"><tig><term>rue</term></tig></langSet><langSet xml:lang="it-IT"><tig><term>via</term></tig></langSet></termEntry></martif>'''
    parsed = parse_glossary("terms.tbx", raw)
    assert [(item.source_term, item.target_term) for item in parsed.entries] == [("rue", "via")]


def test_merge_modes_have_explicit_precedence():
    built = {("fr", "it"): {"rue": "via"}}
    uploaded = {("fr", "it"): {"rue": "strada"}, ("de", "it"): {"haus": "casa"}}
    assert merge_glossaries(built, uploaded, "replace") == uploaded
    assert merge_glossaries(built, uploaded, "merge")[("fr", "it")]["rue"] == "strada"
    assert merge_glossaries(built, uploaded, "auxiliary")[("fr", "it")]["rue"] == "via"
    assert merge_glossaries(built, uploaded, "auxiliary")[("de", "it")]["haus"] == "casa"


def test_basil_accepts_uploaded_locale_lexicon():
    manifest = TextManifest(
        asset_id="x", total_regions=2, src_lang="fr-FR", targ_lang="it-IT", img_dim=(300, 100),
        instances=[
            InstText(id="r1", text="Rue", bounding_box=BBox(0, 0, 30, 20)),
            InstText(id="r2", text="Murs", bounding_box=BBox(70, 0, 40, 20)),
        ],
    )
    lexicon = {("fr-FR", "it-IT"): {"rue": "via", "murs": "muri"}}
    plan = basil.plan_substitution(manifest, "u1", "Via Muri", "it-IT", external_lexicon=lexicon)
    assert plan["method"] == "uploaded_glossary_span_alignment"
    assert [item["text"] for item in plan["assignments"]] == ["Via", "Muri"]
