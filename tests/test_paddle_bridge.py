## 🍢 PaddleOCRBackend subprocess-bridge units
"""
paddlepaddle must never be importable in the app venv (CP-1: it force-
replaces numpy/opencv on install). these tests exercise the bridge's
path resolution and graceful-degradation behavior without needing the
isolated .venv-paddle to exist; the real round trip is covered by
test_paddle_bridge_live.py, which self-skips when it isn't present.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES
from tofu.layers.cicerone import PaddleOCRBackend


class TestVenvResolution:
    def test_default_venv_path(self, monkeypatch):
        monkeypatch.delenv("TOFU_PADDLE_VENV", raising=False)
        p = PaddleOCRBackend._venv_python()
        assert p.parent.parent.name == ".venv-paddle"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOFU_PADDLE_VENV", str(tmp_path))
        p = PaddleOCRBackend._venv_python()
        assert str(tmp_path) in str(p)

    def test_worker_path_exists_in_repo(self):
        # the worker script itself must always ship with the package
        assert PaddleOCRBackend._worker_path().is_file()

    def test_is_available_false_when_venv_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOFU_PADDLE_VENV", str(tmp_path / "nope"))
        assert PaddleOCRBackend.is_available() is False


class TestImagePathResolution:
    def test_path_passthrough_no_temp_file(self):
        backend = PaddleOCRBackend()
        path, tmp = backend._resolve_image_path(str(FIXTURES / "flat-sign.png"))
        assert path == str(FIXTURES / "flat-sign.png")
        assert tmp is None

    def test_ndarray_writes_temp_png(self):
        backend = PaddleOCRBackend()
        arr = np.full((20, 20, 3), 128, dtype=np.uint8)
        path, tmp = backend._resolve_image_path(arr)
        try:
            assert path is not None and tmp == path
            assert Path(path).exists()
            assert Path(path).suffix == ".png"
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    def test_unresolvable_asset_returns_none(self):
        backend = PaddleOCRBackend()
        path, tmp = backend._resolve_image_path(object())
        assert path is None and tmp is None


class TestGracefulDegradation:
    """with the venv pointed at a nonexistent location, the backend must
    degrade to empty results — never raise — matching the contract every
    other OCRBackend honors on failure."""

    def test_detect_returns_empty_when_venv_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOFU_PADDLE_VENV", str(tmp_path / "nope"))
        backend = PaddleOCRBackend(languages=["en"])
        assert backend.detect(str(FIXTURES / "flat-sign.png")) == []

    def test_detect_in_regions_returns_empty_lists_when_venv_missing(self, monkeypatch, tmp_path):
        from tofu.core.types import BBox
        monkeypatch.setenv("TOFU_PADDLE_VENV", str(tmp_path / "nope"))
        backend = PaddleOCRBackend(languages=["en"])
        regions = [BBox(x=0, y=0, width=10, height=10), BBox(x=5, y=5, width=10, height=10)]
        result = backend.detect_in_regions(str(FIXTURES / "flat-sign.png"), regions)
        assert result == [[], []]

    def test_run_worker_reports_error_not_exception(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOFU_PADDLE_VENV", str(tmp_path / "nope"))
        backend = PaddleOCRBackend()
        result = backend._run_worker({"op": "detect", "image_path": "x", "lang": "en"})
        assert result["ok"] is False
        assert "error" in result


class TestLangMapping:
    def test_primary_language_maps_to_tofu_code(self):
        backend = PaddleOCRBackend(languages=["ja"])
        assert backend.lang == "japan"
        assert backend.primary_language == "ja"

    def test_unmapped_language_passes_through(self):
        backend = PaddleOCRBackend(languages=["xx"])
        assert backend.lang == "xx"
