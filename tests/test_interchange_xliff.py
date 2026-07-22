"""XLIFF/CAT import must map safely without relying on one vendor's IDs."""

from tofu.core.types import BBox, InstText, TextManifest
from tofu.utils.interchange import decode_translation_bytes, detect_format, import_xliff_for_manifest


def _manifest() -> TextManifest:
    return TextManifest(
        asset_id="rue-vieux", total_regions=3,
        instances=[
            InstText(id="r1", bounding_box=BBox(4, 8, 90, 20), text="Rue des"),
            InstText(id="r2", bounding_box=BBox(4, 34, 55, 20), text="MURS"),
            InstText(id="r3", bounding_box=BBox(63, 34, 64, 20), text="VIEUX"),
        ],
    )


def test_standard_xliff_12_keeps_immutable_region_identity():
    imported = import_xliff_for_manifest("""<?xml version="1.0"?>
    <xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2"><file><body>
      <trans-unit id="r3"><source>VIEUX</source><target><mrk>OLD</mrk></target></trans-unit>
      <trans-unit id="r2"><source>MURS</source><target>WALLS</target></trans-unit>
    </body></file></xliff>""", _manifest())

    assert imported["translations"] == {"r3": "OLD", "r2": "WALLS"}
    assert imported["matched_by"] == {"id": 2, "bbox": 0, "source": 0}


def test_cat_xliff_resname_bbox_and_unique_source_are_supported():
    imported = import_xliff_for_manifest("""<?xml version="1.0"?>
    <xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" xmlns:sdl="http://sdl.com/FileFormats/SdlXliff/1.0"><file><body>
      <trans-unit id="vendor-123" resname="r2"><source>MURS</source><target><mrk mtype="seg">WALLS</mrk></target></trans-unit>
      <trans-unit id="vendor-456"><source>not used</source><target>OLD</target><note>bbox: 63, 34, 64, 20</note></trans-unit>
      <trans-unit id="vendor-789"><source>Rue des</source><target>Old Street</target></trans-unit>
    </body></file></xliff>""", _manifest())

    assert imported["translations"] == {"r2": "WALLS", "r3": "OLD", "r1": "Old Street"}
    assert imported["matched_by"] == {"id": 1, "bbox": 1, "source": 1}


def test_ambiguous_or_empty_cat_segments_fail_closed():
    manifest = _manifest()
    manifest.instances.append(InstText(id="r4", bounding_box=BBox(0, 60, 60, 20), text="MURS"))
    imported = import_xliff_for_manifest("""<xliff><file><body>
      <trans-unit id="new"><source>MURS</source><target>Walls?</target></trans-unit>
      <trans-unit id="empty"><source>VIEUX</source><target/></trans-unit>
    </body></file></xliff>""", manifest)

    assert imported["translations"] == {}
    assert imported["unresolved"] == [{"id": "new", "source": "MURS"}]
    assert imported["empty_targets"] == 1


def test_cat_extensions_are_detected_as_xliff():
    for filename in ("job.sdlxliff", "job.mxliff", "job.mqxliff", "job.txlf", "job.xlf"):
        assert detect_format(filename) == "xliff"


def test_cat_utf16_and_utf32_exports_decode_without_null_padded_xml():
    document = '<xliff version="1.2"><file><body><trans-unit id="r1"><target>Via</target></trans-unit></body></file></xliff>'
    for encoding in ("utf-8-sig", "utf-16", "utf-32"):
        decoded = decode_translation_bytes(document.encode(encoding))
        assert decoded == document
        assert "\x00" not in decoded
