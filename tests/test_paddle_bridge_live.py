## 🍢 PaddleOCRBackend live round trip (self-skips without .venv-paddle)
"""
exercises the real subprocess bridge against the isolated paddle venv.
skipped entirely when that venv isn't present — CI/other dev machines
without .venv-paddle still pass the app-venv suite; see CP-1 in
project-tofu-workplan-v3.md for why the isolation is mandatory.
"""
import pytest

from conftest import FIXTURES
from tofu.core.types import BBox
from tofu.layers.cicerone import PaddleOCRBackend

pytestmark = pytest.mark.skipif(
    not PaddleOCRBackend.is_available(),
    reason="isolated .venv-paddle not present on this machine",
)


class TestLiveRoundTrip:
    def test_detect_reads_vertical_cjk_column(self):
        backend = PaddleOCRBackend(languages=["ja"])
        dets = backend.detect(str(FIXTURES / "cjk-vertical.png"))
        texts = [d.text for d in dets]
        assert "居酒屋" in texts
        assert "ようこそ" in texts

    def test_detect_in_regions_matches_full_detect(self):
        backend = PaddleOCRBackend(languages=["ja"])
        region = BBox(x=148, y=90, width=92, height=262)
        per_region = backend.detect_in_regions(str(FIXTURES / "cjk-vertical.png"), [region])
        assert len(per_region) == 1
        assert any(d.text == "居酒屋" for d in per_region[0])

    def test_detections_carry_tofu_language_code(self):
        backend = PaddleOCRBackend(languages=["ja"])
        dets = backend.detect(str(FIXTURES / "cjk-vertical.png"))
        assert dets and all(d.language == "ja" for d in dets)

    def test_full_cicerone_pipeline_via_paddle_backend(self):
        """end-to-end through build_manifest — confirms RawDetection
        output from the bridge is fully compatible with the manifest
        assembly (column merge, script ID, pruning) EasyOCR normally
        feeds."""
        from tofu.layers import cicerone
        backend = PaddleOCRBackend(languages=["ja"])
        manifest = cicerone.detect(
            str(FIXTURES / "cjk-vertical.png"), backend=backend,
            multipass=False, zoom=False, polish=False,
        )
        texts = [i.text for i in manifest.instances]
        assert "居酒屋" in texts
        assert "ようこそ" in texts
