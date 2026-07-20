## 🍢 Cicerone — Layer 1
## vieuxtiful
"""
text detection & localization layer.

cicerone finds every text instance in the asset and produces the
TextManifest that all downstream layers consume: bounding boxes,
segmentation masks (Polygon contours), OCR text, confidence, and —
for video (future) — per-frame observations linked into tracks
(frame_index / track_id / temporal_span).

engines are swappable adapters behind the OCRBackend interface:
  - EasyOCRBackend (default): CRAFT detector + CRNN recognition.
    strongest public choice for stylized/curved/scene text; returns
    4-point polygons + text, mapping directly to InstText.
  - NullBackend: empty results when no engine is installed, so the
    pipeline contract keeps working in dev/test environments.
  - PaddleOCRBackend: PP-OCRv5 (DBNet + SVTR), measured 4x recall and
    4x transcription accuracy over EasyOCR on dense/vertical CJK scenes
    (CP-1, scripts/eval_paddle.py). runs out-of-process via
    scripts/paddle_worker.py under an ISOLATED .venv-paddle interpreter
    — paddlepaddle force-replaces the app venv's numpy/opencv on
    install, so it can never be imported in this process. select with
    OCR_ENGINE=paddleocr; falls back to NullBackend when the isolated
    venv (PaddleOCRBackend.is_available()) isn't present.

stacked vertical CJK signage (characters top-to-bottom, each upright):
CRAFT's link stage groups characters horizontally only, so columns
fragment into one box per character with garbage recognition. two
mitigations are built in (measured on scripts/eval_detect.py's dense
street scene):
  - merge_vertical_columns(): x-aligned, width-matched, tightly-stacked
    char boxes are unified into single column detections before manifest
    assembly, so each sign is one region with an accurate bbox.
  - _compose_crop_text(): when a region's crop is re-recognized (the
    language rescue / re-identification paths), ALL detections in the
    crop are joined in reading order instead of keeping only the single
    best one — recovering the full column text ('맥주', not '주').
rotation_info=[90, 270] and mag_ratio 1.5 were measured at zero benefit;
a vertical-aware detector (e.g. PaddleOCR PP-OCRv4) behind a new
OCRBackend adapter remains the eventual upgrade path.
"""

import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tofu.core.types import (
    TextManifest,
    InstText,
    BBox,
    Mask,
    Polygon,
    AssetInfo,
    SceneRegion,
    infer_asset_info,
)

# tofu language codes → easyocr language codes (identity where omitted)
EASYOCR_LANG_MAP: Dict[str, str] = {
    "zh-cn": "ch_sim", "zh-sg": "ch_sim",
    "zh-tw": "ch_tra", "zh-hk": "ch_tra", "zh-mo": "ch_tra",
    "sr-latn": "rs_latin", "sr-cyrl": "rs_cyrillic",
}

# easyocr language codes → tofu language codes (identity where omitted)
EASYOCR_TO_TOFU: Dict[str, str] = {
    "ch_sim": "zh-cn", "ch_tra": "zh-tw",
    "rs_latin": "sr-latn", "rs_cyrillic": "sr-cyrl",
}

# easyocr language sets that cannot be combined with anything except english
EASYOCR_EXCLUSIVE = {"ja", "ch_sim", "ch_tra", "ko", "th"}


def _to_easyocr_lang(lang: str) -> str:
    return EASYOCR_LANG_MAP.get(lang, lang)


def _from_easyocr_lang(lang: str) -> str:
    return EASYOCR_TO_TOFU.get(lang, lang)


def _det_lang_from_engine(det: "RawDetection", engine: "OCRBackend") -> Optional[str]:
    """map a detection's language field to a tofu language code,
    handling both EasyOCR and PaddleOCR backends."""
    if not det.language:
        return None
    if isinstance(engine, PaddleOCRBackend):
        return PaddleOCRBackend.PADDLE_TO_TOFU.get(det.language, det.language)
    return _from_easyocr_lang(det.language)


def expand_langset(langs: Sequence[str]) -> Tuple[str, ...]:
    """build a valid easyocr language set from source-language hints.

    easyocr constraint: ja/ch_sim/ch_tra/ko/th each combine only with
    english. we conservatively keep the FIRST hint plus english, which is
    valid for every supported language and keeps reader init cost flat.
    """
    if not langs:
        return ("en",)
    primary = _to_easyocr_lang(langs[0])
    if primary == "en":
        return ("en",)
    return (primary, "en")


@dataclass
class RawDetection:
    """engine-agnostic detection result an OCRBackend must produce."""
    polygon: Polygon
    text: str
    confidence: float
    language: Optional[str] = None


class OCRBackend(ABC):
    """swappable detection+recognition engine adapter."""

    name: str = "base"

    @property
    def primary_language(self) -> str:
        """canonical tofu language code the engine uses as its primary charset."""
        return "en"

    @abstractmethod
    def detect(self, asset: Any) -> List[RawDetection]:
        """run detection + recognition; return raw polygon/text results."""


class NullBackend(OCRBackend):
    """no-engine fallback: valid empty results (dev/test environments)."""

    name = "null"

    def detect(self, asset: Any) -> List[RawDetection]:
        return []


class EasyOCRBackend(OCRBackend):
    """CRAFT + CRNN via easyocr.

    performance notes baked in:
      - the easyocr.Reader is a process-wide singleton per (languages, gpu)
        key — initialization dominates cost (~15-20 s CPU).
      - large assets are pre-resized to max_dim before detection; tune
        canvas_size/mag_ratio rather than feeding 4K frames directly.
      - CRAFT thresholds (text_threshold/low_text/link_threshold) are
        exposed: lower catches faint stylized text, raises false positives.
      - batch_size > 1 speeds recognition when many boxes exist.
    """

    name = "easyocr"
    _readers: Dict[Tuple[Tuple[str, ...], bool], Any] = {}  # singleton cache

    @property
    def primary_language(self) -> str:
        return _from_easyocr_lang(self.languages[0]) if self.languages else "en"

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        gpu: bool = False,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        batch_size: int = 4,
        max_dim: Optional[int] = 2560,
        # box granularity (easyocr grouping stage):
        #   min_size 20→8: keep small distant signage instead of discarding
        #   add_margin 0.1→0.04: easyocr pads every box by 10% of height on
        #     ALL sides by default — the "boxes too big" complaint verbatim
        #   width_ths 0.5→0.3: horizontal merge distance; 0.5 chains
        #     separate adjacent signs into one box
        #   contrast_ths 0.1→0.3: retry more low-contrast crops with
        #     contrast adjustment before accepting a bad read
        min_size: int = 8,
        add_margin: float = 0.04,
        width_ths: float = 0.3,
        contrast_ths: float = 0.3,
        # small-source upscale: street photography below this on its
        # longer edge loses small/distant signage to CRAFT's effective
        # resolving power (measured on japan-street.jpeg, 627x489 —
        # every documented eval fixture at or above 960px on its longer
        # edge detects fine; both known-bad street scenes are well under
        # it). _prepare() only ever downscaled via max_dim before this;
        # there was no symmetric path for images that start too SMALL.
        min_upscale_dim: int = 850,
        upscale_factor: float = 2.0,
    ):
        self.languages = tuple(_to_easyocr_lang(l) for l in languages)
        self.gpu = gpu
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.min_size = min_size
        self.add_margin = add_margin
        self.width_ths = width_ths
        self.contrast_ths = contrast_ths
        self.min_upscale_dim = min_upscale_dim
        self.upscale_factor = upscale_factor

    def _reader(self):
        import easyocr  # deferred: heavy import
        key = (self.languages, self.gpu)
        if key not in self._readers:
            try:
                self._readers[key] = easyocr.Reader(
                    list(self.languages), gpu=self.gpu, cudnn_benchmark=True
                )
            except TypeError:  # older easyocr without cudnn_benchmark kwarg
                self._readers[key] = easyocr.Reader(
                    list(self.languages), gpu=self.gpu
                )
        return self._readers[key]

    def _prepare(self, asset: Any) -> Tuple[Any, Tuple[float, float]]:
        """pre-resize oversized image files; upscale undersized ones;
        pass everything else through.

        returns (prepared_asset, (sx, sy)) where sx/sy are the resize
        factors applied per axis: original_coord = detected_coord / s.
        (1.0, 1.0) means detection runs in original coordinate space.
        the same division (detected_coord / s) that maps a downscaled
        detection back to original coordinates also correctly maps an
        UPSCALED one — s is just >1 instead of <1, no separate math
        needed for the two directions.
        """
        no_scale = (1.0, 1.0)
        if not isinstance(asset, str):
            return asset, no_scale
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return asset, no_scale
        try:
            img = Image.open(asset)
        except Exception:
            return asset, no_scale
        orig_w, orig_h = img.size
        if self.max_dim is not None and max(img.size) > self.max_dim:
            img.thumbnail((self.max_dim, self.max_dim))
            scale = (img.size[0] / orig_w, img.size[1] / orig_h)
            return np.asarray(img.convert("RGB")), scale
        if self.min_upscale_dim and max(img.size) < self.min_upscale_dim:
            new_size = (
                round(orig_w * self.upscale_factor),
                round(orig_h * self.upscale_factor),
            )
            img = img.resize(new_size, Image.LANCZOS)
            scale = (img.size[0] / orig_w, img.size[1] / orig_h)
            return np.asarray(img.convert("RGB")), scale
        return asset, no_scale

    def detect(
        self,
        asset: Any,
        text_threshold: Optional[float] = None,
        low_text: Optional[float] = None,
    ) -> List[RawDetection]:
        """run detection + recognition; thresholds can be overridden per
        call (multi-pass detection reuses one reader across passes)."""
        reader = self._reader()
        prepared, (sx, sy) = self._prepare(asset)
        # note: mag_ratio 1.5 for small images was measured on the dense
        # street-scene eval (scripts/eval_detect.py) and bought zero
        # additional regions at +50% detection time — not applied.
        results = reader.readtext(
            prepared,
            text_threshold=text_threshold if text_threshold is not None else self.text_threshold,
            low_text=low_text if low_text is not None else self.low_text,
            link_threshold=self.link_threshold,
            canvas_size=self.canvas_size,
            mag_ratio=self.mag_ratio,
            batch_size=self.batch_size,
            min_size=self.min_size,
            add_margin=self.add_margin,
            width_ths=self.width_ths,
            contrast_ths=self.contrast_ths,
        )
        primary_lang = self.languages[0] if self.languages else None
        return [
            RawDetection(
                # map back to original coordinate space when pre-resized
                polygon=[(int(round(x / sx)), int(round(y / sy))) for x, y in box],
                text=text,
                confidence=float(conf),
                language=primary_lang,
            )
            for box, text, conf in results
        ]

    # CRNN recognition operates on 32-64 px text lines; crops below this
    # band are re-sampled up before recognition (standard practice for
    # small-text OCR — recognition accuracy collapses under ~16 px x-height)
    MIN_CROP_HEIGHT = 40
    MAX_CROP_UPSCALE = 3

    def detect_in_regions(
        self,
        asset: Any,
        regions: List[BBox],
        pad: int = 4,
        polygons: Optional[List[Optional[Polygon]]] = None,
    ) -> List[List[RawDetection]]:
        """detect + recognize inside bbox crops of the asset.

        faster and more accurate than full-image detection for small or
        re-examined regions; tiny crops are bicubically upscaled to the
        recognizer's operating resolution first. if a polygon is supplied
        for a region, the crop is perspective-rectified before recognition.
        returns one list per input region (parallel order); polygons are
        expressed in full-image coordinates.
        """
        from tofu.utils.imaging import load_rgb
        img = load_rgb(asset)
        if img is None:
            return [[] for _ in regions]
        reader = self._reader()
        h, w = img.shape[:2]
        primary_lang = self.languages[0] if self.languages else None
        out: List[List[RawDetection]] = []
        polys = polygons or [None] * len(regions)
        for bbox, poly in zip(regions, polys):
            x0, y0 = max(0, bbox.x - pad), max(0, bbox.y - pad)
            x1 = min(w, bbox.x + bbox.width + pad)
            y1 = min(h, bbox.y + bbox.height + pad)
            if x1 - x0 < 3 or y1 - y0 < 3:
                out.append([])
                continue
            crop = img[y0:y1, x0:x1]
            # build a relative polygon for rectification if provided
            if poly is not None:
                rel_poly = [(px - x0, py - y0) for px, py in poly if (x0 <= px < x1 and y0 <= py < y1)]
                if len(rel_poly) >= 4:
                    crop = _rectify_crop(crop, rel_poly)
            scale = 1
            short_side = min(crop.shape[0], crop.shape[1])
            if short_side < self.MIN_CROP_HEIGHT:
                scale = min(
                    self.MAX_CROP_UPSCALE,
                    max(2, round(self.MIN_CROP_HEIGHT / max(1, short_side))),
                )
                try:
                    import numpy as np
                    from PIL import Image
                    crop = np.asarray(
                        Image.fromarray(crop).resize(
                            (crop.shape[1] * scale, crop.shape[0] * scale),
                            Image.BICUBIC,
                        )
                    )
                except Exception:
                    crop = img[y0:y1, x0:x1]
                    scale = 1
            results = reader.readtext(
                crop,
                text_threshold=self.text_threshold,
                low_text=self.low_text,
                link_threshold=self.link_threshold,
                batch_size=self.batch_size,
                min_size=self.min_size,
                add_margin=self.add_margin,
                width_ths=self.width_ths,
                contrast_ths=self.contrast_ths,
            )
            out.append([
                RawDetection(
                    polygon=[
                        (int(x / scale + x0), int(y / scale + y0))
                        for x, y in box
                    ],
                    text=text,
                    confidence=float(conf),
                    language=primary_lang,
                )
                for box, text, conf in results
            ])
        return out


class PaddleOCRBackend(OCRBackend):
    """PP-OCRv4 / DB + SVTR via paddleocr.

    PaddleOCR uses DBNet for detection and SVTR_LCNet for recognition,
    which generally outperforms CRAFT+CRNN on rotated, curved, dense,
    and CJK scene text. The adapter exposes the same OCRBackend contract
    so it slots into cicerone.detect() and the streaming pipeline.
    """

    name = "paddleocr"

    # tofu code -> paddleocr lang code (use latin for all western european
    # latin-script languages; PaddleOCR has per-language rec models but
    # the 'latin' model is a practical catch-all for mixed western text)
    PADDLE_LANG_MAP: Dict[str, str] = {
        "en": "en",
        "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
        "ko": "korean",
        "ja": "japan",
        "zh-cn": "ch", "zh-sg": "ch",
        "zh-tw": "ch_tra", "zh-hk": "ch_tra", "zh-mo": "ch_tra",
        "ru": "ru", "uk": "uk", "bg": "bg",
        "ar": "ar",
        "he": "he",
        "hi": "hi", "mr": "mr", "ne": "ne",
        "th": "th",
        "el": "el",
        "sr-latn": "latin", "sr-cyrl": "cyrillic",
    }

    PADDLE_TO_TOFU: Dict[str, str] = {
        "ch": "zh-cn",
        "ch_tra": "zh-tw",
        "korean": "ko",
        "japan": "ja",
    }

    # subprocess bridge, not an in-process reader: paddlepaddle force-
    # replaces the app venv's numpy/opencv on install (measured in CP-1 —
    # one attempt corrupted numpy mid-install), so PaddleOCR only ever
    # runs under the ISOLATED .venv-paddle interpreter, via
    # scripts/paddle_worker.py. see that module's docstring for the wire
    # protocol (JSON over stdin, result written to a temp file — never
    # stdout, which PaddleOCR's own logging pollutes).
    _CICERONE_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CICERONE_DIR.parents[2]  # layers -> tofu -> src -> root
    WORKER_TIMEOUT_S = 180

    @property
    def primary_language(self) -> str:
        return self._to_tofu_lang()

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        gpu: bool = False,
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        drop_score: float = 0.3,
    ):
        self.languages = tuple(self.PADDLE_LANG_MAP.get(l, l) for l in languages)
        # PaddleOCR readers are one-language; use the primary language.
        # Mixed-script fallback is handled by the primary chosen from hints.
        self.lang = self.languages[0] if self.languages else "en"
        self.gpu = bool(gpu)
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.drop_score = drop_score

    def _to_tofu_lang(self) -> str:
        return self.PADDLE_TO_TOFU.get(self.lang, self.lang)

    @classmethod
    def _venv_python(cls) -> Path:
        """path to the isolated paddle interpreter; override with
        TOFU_PADDLE_VENV (a venv root, i.e. the dir containing Scripts/
        or bin/) for non-default layouts."""
        import os
        import sys
        override = os.environ.get("TOFU_PADDLE_VENV")
        venv_root = Path(override) if override else cls._PROJECT_ROOT / ".venv-paddle"
        exe = "python.exe" if sys.platform == "win32" else "python"
        subdir = "Scripts" if sys.platform == "win32" else "bin"
        return venv_root / subdir / exe

    @classmethod
    def _worker_path(cls) -> Path:
        return cls._PROJECT_ROOT / "scripts" / "paddle_worker.py"

    @classmethod
    def is_available(cls) -> bool:
        """True when the isolated paddle venv + worker script both exist —
        the availability check callers use in place of `import paddleocr`
        (which must never happen in the app process)."""
        return cls._venv_python().is_file() and cls._worker_path().is_file()

    def _resolve_image_path(self, asset: Any) -> Tuple[Optional[str], Optional[str]]:
        """asset -> (path, temp_path_to_clean_up_or_None).

        a path/str is used directly (no cross-venv object serialization
        needed — the worker just opens the file). a PIL image or ndarray
        is written to a temp PNG first, since passing in-memory arrays
        across the venv boundary risks a numpy ABI mismatch."""
        if isinstance(asset, (str, Path)):
            return str(asset), None
        try:
            import tempfile
            from tofu.utils.imaging import load_rgb
            img = load_rgb(asset)
            if img is None:
                return None, None
            from PIL import Image
            fd, tmp = tempfile.mkstemp(suffix=".png")
            import os
            os.close(fd)
            Image.fromarray(img).save(tmp)
            return tmp, tmp
        except Exception:
            return None, None

    def _run_worker(self, request: Dict[str, Any]) -> Dict[str, Any]:
        import json
        import os
        import subprocess
        import tempfile

        fd, out_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        request = dict(request, out_path=out_path)
        try:
            proc = subprocess.run(
                [str(self._venv_python()), str(self._worker_path())],
                input=json.dumps(request),
                capture_output=True, text=True, encoding="utf-8",
                timeout=self.WORKER_TIMEOUT_S,
            )
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception:
                stderr_tail = (proc.stderr or "")[-500:]
                return {"ok": False, "error": f"worker produced no result "
                                              f"(rc={proc.returncode}): {stderr_tail}"}
            return result
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "paddleocr worker timed out"}
        except OSError as exc:
            # e.g. the isolated venv vanished after is_available() passed
            return {"ok": False, "error": f"could not launch paddleocr worker: {exc}"}
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

    def _det_to_raw(self, det: Dict[str, Any]) -> RawDetection:
        pts = [(int(round(p[0])), int(round(p[1]))) for p in det["polygon"]]
        return RawDetection(
            polygon=pts, text=det["text"], confidence=float(det["confidence"]),
            language=self._to_tofu_lang(),
        )

    def detect(
        self,
        asset: Any,
        text_threshold: Optional[float] = None,
        low_text: Optional[float] = None,
    ) -> List[RawDetection]:
        # text_threshold / low_text: PaddleOCR exposes det_db_thresh, not
        # identical knobs; signature preserved to satisfy OCRBackend.
        image_path, tmp = self._resolve_image_path(asset)
        if image_path is None:
            return []
        try:
            result = self._run_worker({
                "op": "detect", "image_path": image_path, "lang": self.lang,
            })
        finally:
            if tmp:
                import os
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        if not result.get("ok"):
            return []
        return [self._det_to_raw(d) for d in result.get("detections", [])]

    def detect_in_regions(
        self,
        asset: Any,
        regions: List[BBox],
        pad: int = 4,
        polygons: Optional[List[Optional[Polygon]]] = None,
    ) -> List[List[RawDetection]]:
        # polygons (perspective rectification) are not yet applied by the
        # worker — PaddleOCR's own detector/angle-classifier already
        # handles a meaningful amount of rotation natively; a v2 worker
        # can add cv2 rectification if crops prove to need it.
        image_path, tmp = self._resolve_image_path(asset)
        if image_path is None:
            return [[] for _ in regions]
        try:
            result = self._run_worker({
                "op": "detect_regions", "image_path": image_path, "lang": self.lang,
                "regions": [[b.x, b.y, b.width, b.height] for b in regions],
                "pad": pad,
            })
        finally:
            if tmp:
                import os
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        if not result.get("ok"):
            return [[] for _ in regions]
        return [
            [self._det_to_raw(d) for d in region_dets]
            for region_dets in result.get("per_region", [])
        ]


# -- script identification ----------------------------------------------------

class ScriptDetector:
    """classify recognized text into a script family via unicode ranges.

    operates on RECOGNIZED text, not pixels: reliable and free whenever
    the reader's charset covered the script (e.g. a (ja, en) reader
    yields kana/han vs. latin per region). it cannot rescue text
    recognized with a charset that lacks the script entirely — that case
    is handled by passing source-language hints into detect().
    """

    RANGES: Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...] = (
        ("han", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
        ("hiragana", ((0x3040, 0x309F),)),
        ("katakana", ((0x30A0, 0x30FF), (0xFF66, 0xFF9D))),
        ("hangul", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
        ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))),
        ("hebrew", ((0x0590, 0x05FF),)),
        ("cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F))),
        ("greek", ((0x0370, 0x03FF),)),
        ("devanagari", ((0x0900, 0x097F),)),
        ("thai", ((0x0E00, 0x0E7F),)),
        ("latin", ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))),
    )

    def detect_script(self, text: str) -> Optional[str]:
        """dominant script of the text, or None if no script chars found.

        any kana at all classifies as "japanese" (ja text is a han+kana
        mix where kana are the discriminating signal vs. chinese).
        """
        counts: Dict[str, int] = {}
        for ch in text or "":
            cp = ord(ch)
            for name, ranges in self.RANGES:
                if any(lo <= cp <= hi for lo, hi in ranges):
                    counts[name] = counts.get(name, 0) + 1
                    break
        if not counts:
            return None
        if counts.get("hiragana") or counts.get("katakana"):
            return "japanese"
        return max(counts, key=lambda k: counts[k])


# script family → representative tofu language code
SCRIPT_TO_LANG: Dict[str, str] = {
    "latin": "en", "cyrillic": "ru", "greek": "el", "arabic": "ar",
    "hebrew": "he", "devanagari": "hi", "thai": "th",
    "japanese": "ja", "hangul": "ko", "han": "zh-cn",
}

# script family → easyocr language set able to recognize it
SCRIPT_TO_EASYOCR_SET: Dict[str, Tuple[str, ...]] = {
    "latin": ("en",), "cyrillic": ("ru", "en"), "greek": ("el", "en"),
    "arabic": ("ar", "en"), "hebrew": ("he", "en"),
    "devanagari": ("hi", "en"), "thai": ("th", "en"),
    "japanese": ("ja", "en"), "hangul": ("ko", "en"), "han": ("ch_sim", "en"),
}

# scripts each easyocr language's charset can emit (default: latin)
EASYOCR_LANG_SCRIPTS: Dict[str, set] = {
    "en": {"latin"},
    "ja": {"japanese", "han", "hiragana", "katakana", "latin"},
    "ko": {"hangul", "latin"},
    "ch_sim": {"han", "latin"}, "ch_tra": {"han", "latin"},
    "ru": {"cyrillic", "latin"}, "uk": {"cyrillic", "latin"},
    "bg": {"cyrillic", "latin"}, "mn": {"cyrillic", "latin"},
    "rs_cyrillic": {"cyrillic", "latin"},
    "ar": {"arabic", "latin"}, "fa": {"arabic", "latin"}, "ur": {"arabic", "latin"},
    "he": {"hebrew", "latin"},
    "hi": {"devanagari", "latin"}, "mr": {"devanagari", "latin"},
    "ne": {"devanagari", "latin"},
    "th": {"thai", "latin"},
    "el": {"greek", "latin"},
}


# latin-script language identification: script detection alone cannot
# distinguish spanish from english — both are "latin". a compact
# stopword + diacritic classifier over the recognized text does the
# disambiguation. deliberately conservative: returns None on weak signal.
LATIN_STOPWORDS: Dict[str, set] = {
    "en": {"the", "and", "of", "to", "in", "for", "on", "with", "at", "by",
           "street", "avenue", "exit", "open", "closed", "stop", "main",
           "no", "yes", "not", "this", "is", "are", "you", "all", "new"},
    "es": {"el", "la", "los", "las", "de", "del", "y", "en", "un", "una",
           "por", "para", "con", "que", "se", "su", "al", "calle", "avenida",
           "salida", "abierto", "cerrado", "alto", "mayor", "no", "si"},
    "fr": {"le", "la", "les", "de", "des", "du", "et", "en", "un", "une",
           "pour", "avec", "que", "au", "aux", "rue", "sortie", "ouvert",
           "ferme", "arret", "sur", "pas"},
    "de": {"der", "die", "das", "und", "von", "zu", "mit", "im", "am",
           "ein", "eine", "fur", "auf", "ist", "nicht", "strasse", "ausgang",
           "offen", "geschlossen", "halt", "aus"},
    "it": {"il", "lo", "la", "le", "di", "e", "in", "un", "una", "per",
           "con", "che", "del", "della", "via", "uscita", "aperto",
           "chiuso", "alt", "non"},
    "pt": {"o", "os", "as", "de", "do", "da", "dos", "das", "e", "em",
           "um", "uma", "para", "com", "que", "rua", "saida", "aberto",
           "fechado", "nao", "sim"},
}
LATIN_DIACRITICS: Dict[str, str] = {
    "es": "ñáéíóúü¿¡", "fr": "àâçèéêëîïôùûüœ", "de": "äöüß",
    "it": "àèéìòù", "pt": "ãõçáâêôú",
}


def guess_latin_language(texts: Sequence[str]) -> Optional[str]:
    """guess the language of latin-script texts via stopwords + diacritics.

    returns a tofu language code, or None when the signal is too weak to
    override the default. english must be BEATEN, not tied, to switch.
    """
    scores: Dict[str, float] = {lang: 0.0 for lang in LATIN_STOPWORDS}
    words = []
    joined = " ".join(t for t in texts if t)
    for token in joined.lower().split():
        words.append(token.strip(".,;:!?()[]\"'"))
    for lang, stops in LATIN_STOPWORDS.items():
        scores[lang] += 2.0 * sum(1 for w in words if w in stops)
    for lang, chars in LATIN_DIACRITICS.items():
        scores[lang] += sum(1 for ch in joined.lower() if ch in chars)
    best = max(scores, key=lambda k: scores[k])
    if best == "en" or scores[best] < 2.0 or scores[best] <= scores["en"]:
        return None
    return best


def _script_lang_for(script: str, reader_langs: Sequence[str]) -> str:
    """tofu language code for a script, preferring the reader's own langs."""
    if script == "latin":
        return "en" if "en" in reader_langs else _from_easyocr_lang(reader_langs[0])
    for lang in reader_langs:
        if script in EASYOCR_LANG_SCRIPTS.get(lang, {"latin"}):
            return _from_easyocr_lang(lang)
    return SCRIPT_TO_LANG.get(script, "en")


# -- detection geometry -------------------------------------------------------

def _polygon_bbox(polygon: Polygon) -> BBox:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return BBox(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))


def _overlap_frac(a: BBox, b: BBox) -> float:
    """intersection area over the SMALLER box's area (containment-aware:
    a small box inside a large one scores ~1.0, unlike plain IoU)."""
    ix = max(a.x, b.x)
    iy = max(a.y, b.y)
    ax = min(a.x + a.width, b.x + b.width)
    ay = min(a.y + a.height, b.y + b.height)
    if ax <= ix or ay <= iy:
        return 0.0
    inter = (ax - ix) * (ay - iy)
    smaller = min(a.width * a.height, b.width * b.height)
    return inter / smaller if smaller > 0 else 0.0


def _containment_frac(inner: BBox, outer: BBox) -> float:
    """fraction of inner's area that lies inside outer."""
    ix = max(inner.x, outer.x)
    iy = max(inner.y, outer.y)
    ax = min(inner.x + inner.width, outer.x + outer.width)
    ay = min(inner.y + inner.height, outer.y + outer.height)
    if ax <= ix or ay <= iy:
        return 0.0
    area = inner.width * inner.height
    return ((ax - ix) * (ay - iy)) / area if area > 0 else 0.0


def _rectify_crop(img: Any, polygon: Polygon, target_height: int = 48) -> Any:
    """return a fronto-parallel crop of a quadrilateral text region.

    guards:
      - polygon must have at least 4 points
      - aspect ratio must be reasonable (0.1 to 10)
      - the quadrilateral must be convex-ish (convex-hull area >= 60% of bbox)
    if any guard fails, returns the original image unchanged so recognition
    falls back to the axis-aligned crop.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return img
    if img is None or len(polygon) < 4:
        return img
    pts = np.array(polygon, dtype=np.float32)
    if pts.shape[0] < 4:
        return img

    # convex hull + bbox area for convexity guard
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    xs, ys = pts[:, 0], pts[:, 1]
    bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    if bbox_area <= 0 or hull_area / bbox_area < 0.6:
        return img

    # aspect guard using the 4 corners in order
    ordered = np.array(pts[:4], dtype=np.float32)
    w = np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])
    h = np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])
    w = max(w / 2, 1)
    h = max(h / 2, 1)
    aspect = max(w / h, h / w)
    if aspect > 10 or aspect < 0.1:
        return img

    # compute width/height for output using average edge lengths
    out_w = max(int(w), 8)
    out_h = max(int(h), 8)
    # normalize text height to recognizer sweet spot without extreme upscaling
    if h < target_height:
        scale = min(4, max(1, target_height / h))
        out_w = int(out_w * scale)
        out_h = int(out_h * scale)

    dst = np.array([
        [0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]
    ], dtype=np.float32)
    m = cv2.getPerspectiveTransform(ordered, dst)
    try:
        return cv2.warpPerspective(img, m, (out_w, out_h))
    except Exception:
        return img


def merge_detections(
    base: List[RawDetection], extra: List[RawDetection]
) -> List[RawDetection]:
    """NMS across passes: dedupe by overlap, keep the higher confidence."""
    for det in extra:
        db = _polygon_bbox(det.polygon)
        dup_idx = None
        for i, kept in enumerate(base):
            if _overlap_frac(db, _polygon_bbox(kept.polygon)) > 0.5:
                dup_idx = i
                break
        if dup_idx is None:
            base.append(det)
        elif det.confidence > base[dup_idx].confidence:
            base[dup_idx] = det
    return base


# vertical column merge: CRAFT links characters horizontally only, so
# stacked vertical CJK signage fragments into one box per character.
COLUMN_MAX_ASPECT = 1.6   # member boxes must be char-like/tall, not wide lines
COLUMN_X_ALIGN = 0.5      # x-center offset tolerance, fraction of max width
COLUMN_WIDTH_RATIO = 1.7  # max width disparity between members
COLUMN_MAX_GAP = 0.8      # vertical gap tolerance, fraction of max width


def merge_vertical_columns(detections: List[RawDetection]) -> List[RawDetection]:
    """merge per-character fragments of stacked vertical CJK signage into
    single column detections.

    two detections belong to the same column when their x-centers align,
    their widths are comparable, and the vertical gap between them is
    smaller than a character width (the height proxy for upright CJK).
    wide boxes (w > 1.6h) are text LINES and never join a column, so
    horizontal multi-line layouts are untouched. merged text is the
    members' text top-to-bottom; the language rescue downstream re-runs
    recognition on the merged crop, where the full column reads correctly.
    """
    n = len(detections)
    if n < 2:
        return detections
    boxes = [_polygon_bbox(d.polygon) for d in detections]

    def char_like(b: BBox) -> bool:
        return b.height > 0 and b.width <= COLUMN_MAX_ASPECT * b.height

    def same_column(a: BBox, b: BBox) -> bool:
        if not (char_like(a) and char_like(b)):
            return False
        wmax = max(a.width, b.width)
        if abs((a.x + a.width / 2) - (b.x + b.width / 2)) > COLUMN_X_ALIGN * wmax:
            return False
        if wmax > COLUMN_WIDTH_RATIO * max(1, min(a.width, b.width)):
            return False
        gap = max(a.y, b.y) - min(a.y + a.height, b.y + b.height)
        return gap <= COLUMN_MAX_GAP * wmax

    # union-find over all pairs (detection counts are small)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if same_column(boxes[i], boxes[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: List[RawDetection] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(detections[members[0]])
            continue
        members.sort(key=lambda i: boxes[i].y)
        x0 = min(boxes[i].x for i in members)
        y0 = min(boxes[i].y for i in members)
        x1 = max(boxes[i].x + boxes[i].width for i in members)
        y1 = max(boxes[i].y + boxes[i].height for i in members)
        confs = [detections[i].confidence for i in members]
        out.append(RawDetection(
            polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            text="".join((detections[i].text or "").strip() for i in members),
            confidence=sum(confs) / len(confs),
            language=next(
                (detections[i].language for i in members
                 if detections[i].language), None
            ),
        ))
    return out


# over-merged vertical stack re-split: CRAFT's own horizontal-only
# character linking (see merge_vertical_columns's docstring) has no
# symmetric protection against a TIGHTLY stacked vertical sign getting
# unified into one box at the DETECTION stage itself — measured: a
# single 127x603 box for a 6-character "茂昌眼镜公司" column. that's the
# mirror image of what merge_vertical_columns fixes (reassembling
# already-fragmented per-character detections): here there's nothing
# fragmented to reassemble, because detection already over-merged.
# recognizing 600px of stacked, unrelated glyphs as one text line
# produces garbage regardless of reader charset (measured conf 0.01-0.46
# on the SAME crop where the same reader reads short/isolated text at
# conf 0.95-0.99) -- the fix is to re-segment BEFORE recognition, not to
# read differently.
VERTICAL_STACK_MIN_ASPECT = 3.0  # height >= 3x width: too tall for one
                                  # line, likely several stacked chars
MIN_BAND_HEIGHT_PX = 12          # a band shorter than this can't hold
                                  # one legible character


def _segment_vertical_bands(asset: Any, bbox: BBox) -> List[BBox]:
    """split a tall/narrow bbox into per-character horizontal bands.

    finds character gaps via a horizontal ink-density profile off the
    shared imaging.text_mask() (rows with near-zero ink are gaps between
    stacked glyphs — the same connected-component-adjacent technique
    savor.py's _plate_up() already uses for per-glyph segmentation, one
    level up: character bands instead of individual glyph blobs). falls
    back to equal division by the aspect-implied character count (CJK
    characters are roughly square) when no clear gaps are found — dense
    or touching glyphs, or a background too complex for text_mask's
    Otsu split to separate cleanly. returns bands in the bbox's own
    (full-image) coordinate space, or [] when nothing usable resulted.
    """
    from tofu.utils.imaging import load_rgb, text_mask
    n_chars_guess = max(2, round(bbox.height / max(1, bbox.width)))

    img = load_rgb(asset)
    if img is not None:
        # refine=False: GrabCut is expensive and this only needs a
        # coarse ink/gap profile, not a precision stroke mask
        mask = text_mask(img, bbox, refine=False)
        if mask is not None:
            row_ink = mask.sum(axis=1)
            gap_floor = max(1, int(0.02 * mask.shape[1]))
            bands: List[Tuple[int, int]] = []
            start = None
            for y, ink in enumerate(row_ink):
                is_gap = ink <= gap_floor
                if not is_gap and start is None:
                    start = y
                elif is_gap and start is not None:
                    bands.append((start, y))
                    start = None
            if start is not None:
                bands.append((start, len(row_ink)))
            bands = [(a, b) for a, b in bands if (b - a) >= MIN_BAND_HEIGHT_PX]
            if len(bands) >= 2:
                return [
                    BBox(x=bbox.x, y=bbox.y + a, width=bbox.width, height=b - a)
                    for a, b in bands
                ]

    if n_chars_guess < 2:
        return []
    band_h = bbox.height / n_chars_guess
    if band_h < MIN_BAND_HEIGHT_PX:
        return []
    return [
        BBox(
            x=bbox.x, y=bbox.y + round(i * band_h),
            width=bbox.width, height=round(band_h),
        )
        for i in range(n_chars_guess)
    ]


def _split_tall_detections(
    asset: Any, engine: "EasyOCRBackend", detections: List[RawDetection],
) -> Optional[List[RawDetection]]:
    """re-segment and re-recognize over-tall/narrow detections that are
    likely an over-merged vertical CJK stack (see module note above).

    each detection whose box trips VERTICAL_STACK_MIN_ASPECT gets split
    into per-character bands (_segment_vertical_bands) and each band is
    re-recognized individually via the engine's existing
    detect_in_regions() — no new recognition machinery. the resulting
    per-character detections are handed back to the caller to feed
    through build_manifest(), whose own merge_vertical_columns() call
    reassembles them into one clean column with composed text, exactly
    as it already does for genuinely fragmented per-character
    detections — reusing that logic rather than duplicating it here.

    returns None when nothing needed splitting (caller can skip the
    extra build_manifest() re-run), or the full replacement detection
    list otherwise. a detection that trips the aspect trigger but can't
    be usefully split, or whose split doesn't clearly improve on the
    original read, passes through unchanged.

    the aspect trigger alone is NOT sufficient to commit to a
    replacement: a genuine SINGLE tall character can trip it too, and
    _segment_vertical_bands's coarse ink-gap profile can mistake that
    character's own internal structure for inter-character gaps.
    requiring the split's average confidence to clearly exceed the
    original whole-box read's confidence before committing lets the
    genuine multi-character cases (measured: china-street's garbage
    single-digit "1" read at confidence 0.010 vs. 4 correctly split/
    re-recognized characters averaging ~0.6) through — but this
    comparison alone does NOT fully protect Korean specifically: a
    single Hangul syllable block is visually COMPOSED of 2-3 jamo
    sub-glyphs with real internal gaps between them, so the segmenter
    can slice a genuine character into fragments that each still
    resemble a DIFFERENT, valid (but wrong) syllable, and the
    recognizer can read those wrong fragments confidently (measured
    live: gemini-street's correctly-read '놓' split into '노'+'방', both
    recognized at >99% confidence, comfortably beating the original's
    own low confidence despite being entirely wrong). Han/Kanji
    ideographs don't share this failure mode — they're monolithic
    blocks with no internal white-space gaps of their own — so Korean
    is excluded outright rather than papered over with a shakier
    confidence margin.
    """
    if engine.languages and engine.languages[0] == "ko":
        return None
    changed = False
    out: List[RawDetection] = []
    for det in detections:
        bbox = _polygon_bbox(det.polygon)
        if bbox.width <= 0 or bbox.height < VERTICAL_STACK_MIN_ASPECT * bbox.width:
            out.append(det)
            continue
        bands = _segment_vertical_bands(asset, bbox)
        if len(bands) < 2:
            out.append(det)
            continue
        try:
            per_band = engine.detect_in_regions(asset, bands)
        except Exception:
            out.append(det)
            continue
        composed_bands: List[Tuple[BBox, RawDetection]] = []
        for band_bbox, band_dets in zip(bands, per_band):
            composed = _compose_crop_text(band_dets)
            if composed is None or not (composed.text or "").strip():
                continue
            composed_bands.append((band_bbox, composed))
        if len(composed_bands) < 2:
            out.append(det)
            continue
        avg_conf = sum(c.confidence for _, c in composed_bands) / len(composed_bands)
        if avg_conf <= (det.confidence or 0):
            out.append(det)
            continue
        changed = True
        for band_bbox, composed in composed_bands:
            out.append(RawDetection(
                polygon=[
                    (band_bbox.x, band_bbox.y),
                    (band_bbox.x + band_bbox.width, band_bbox.y),
                    (band_bbox.x + band_bbox.width, band_bbox.y + band_bbox.height),
                    (band_bbox.x, band_bbox.y + band_bbox.height),
                ],
                text=composed.text, confidence=composed.confidence,
                language=det.language,
            ))
    return out if changed else None


def _compose_crop_text(
    dets: List[RawDetection], min_conf: float = 0.2
) -> Optional[RawDetection]:
    """compose a region's text from ALL detections found inside its crop.

    a vertical column crop yields one detection PER CHARACTER; taking only
    the single best detection truncates the region to one character.
    instead, order the detections by the crop's reading direction
    (vertical: top-to-bottom; horizontal: left-to-right) and join them.
    detections below min_conf are dropped so noise never contaminates
    the composed string.
    """
    usable = [
        d for d in dets
        if (d.text or "").strip() and d.confidence >= min_conf
    ]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]
    boxes = [_polygon_bbox(d.polygon) for d in usable]
    xcs = [b.x + b.width / 2 for b in boxes]
    ycs = [b.y + b.height / 2 for b in boxes]
    vertical = (max(ycs) - min(ycs)) >= (max(xcs) - min(xcs))
    order = sorted(
        range(len(usable)),
        key=lambda i: (ycs[i], xcs[i]) if vertical else (xcs[i], ycs[i]),
    )
    sep = "" if vertical else " "
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x + b.width for b in boxes)
    y1 = max(b.y + b.height for b in boxes)
    return RawDetection(
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        text=sep.join(usable[i].text.strip() for i in order),
        confidence=sum(d.confidence for d in usable) / len(usable),
        language=next((d.language for d in usable if d.language), None),
    )


def _probe_dim(asset: Any) -> Optional[Tuple[int, int]]:
    """resolve the original (width, height) of the asset, if determinable.

    this is the coordinate-space contract for the manifest: every bbox and
    polygon is expressed relative to these dimensions, regardless of any
    internal pre-resize the backend performs.
    """
    if hasattr(asset, "size") and hasattr(asset, "convert"):  # PIL image
        return tuple(asset.size)
    if isinstance(asset, (str, Path)):
        try:
            from PIL import Image
            with Image.open(asset) as img:
                return img.size
        except Exception:
            return None
    try:
        import numpy as np
        if isinstance(asset, np.ndarray) and asset.ndim >= 2:
            return (int(asset.shape[1]), int(asset.shape[0]))
    except ImportError:
        pass
    return None


# -- module-level engine management ------------------------------------------

_default_backend: Optional[OCRBackend] = None
_default_engine_name: Optional[str] = None


def _engine_from_env() -> str:
    import os
    return os.environ.get("OCR_ENGINE", "easyocr").lower()


def get_backend() -> OCRBackend:
    """default backend: EasyOCR when installed, PaddleOCR if requested
    via OCR_ENGINE env var, Null otherwise."""
    global _default_backend, _default_engine_name
    engine = _engine_from_env()
    if _default_backend is None or _default_engine_name != engine:
        _default_engine_name = engine
        if engine == "paddleocr":
            _default_backend = (
                PaddleOCRBackend() if PaddleOCRBackend.is_available()
                else NullBackend()
            )
        else:
            try:
                import easyocr  # noqa: F401
                _default_backend = EasyOCRBackend()
            except ImportError:
                _default_backend = NullBackend()
    return _default_backend


def set_backend(backend: OCRBackend) -> None:
    """swap the engine (e.g., PaddleOCR adapter) without touching the pipeline."""
    global _default_backend, _default_engine_name
    _default_backend = backend
    _default_engine_name = backend.name


# -- layer entry point --------------------------------------------------------

# multi-pass CRAFT thresholds: standard → stylized/faint → hard text
PASS_THRESHOLDS: Tuple[Tuple[float, float], ...] = (
    (0.7, 0.4), (0.5, 0.3), (0.3, 0.2),
)


# candidate language sets the auto-probe tries when an unhinted pass
# produced garbage (charsets the default english reader cannot express)
PROBE_LANGSETS: Tuple[Tuple[str, ...], ...] = (
    ("ja", "en"), ("ko", "en"), ("ch_sim", "en"),
)

# minimum count of real CJK-script (han/japanese/hangul) instances
# required before build_manifest()'s "no kana observed" heuristic is
# trusted enough to relabel ja instances as zh-cn — below this, absence
# of kana more likely means too little text survived detection to
# contain it than that the scene is genuinely kana-free.
MIN_DISAMBIGUATION_EVIDENCE = 2

# confidence floor for a script-bearing instance to count as evidence in
# _disambiguate_ja_zh — matches the floor _identify_languages already
# applies to low-confidence latin; a near-zero-confidence misread of a
# tiny/blurry garbage crop can accidentally shape-match a kana glyph and
# must not be trusted as a definitive Japanese signal.
MIN_KANA_CONFIDENCE = 0.3


def _script_bearing_conf(det: RawDetection, target_scripts: set) -> float:
    """confidence of a detection, counted only if its text actually
    contains characters of the candidate's own (non-latin) scripts.

    a digit or symbol recognized confidently by ANY charset carries zero
    language evidence — without this gate, a '4' at conf 0.95 in the
    english pass defeats every probe.
    """
    detector = ScriptDetector()
    script = detector.detect_script(det.text or "")
    if script is None or script == "latin":
        return 0.0
    if script in target_scripts:
        return det.confidence
    return 0.0


def _auto_probe_language(
    asset: Any,
    instances: List[InstText],
    engine: "EasyOCRBackend",
    probe_regions: int = 8,
    min_evidence: float = 0.2,
    max_winners: int = 2,
) -> List["EasyOCRBackend"]:
    """rescue path for unhinted detection of non-latin assets.

    an english-charset reader recognizes korean/japanese/chinese signage
    as garbage latin/digits — script identification of that text can
    never reveal the true language. when enough regions look like
    garbage, probe the largest GARBAGE regions with each candidate CJK
    reader and keep every reader that produces real script-bearing
    recognitions (mean script-bearing conf >= min_evidence). returns up
    to max_winners backends — mixed-script scenes (korean + japanese
    signage on one street) legitimately have two winners.
    """
    detector = ScriptDetector()
    scored = [i for i in instances if i.bounding_box is not None]
    if not scored:
        return []

    reader_langs = tuple(engine.languages) or ("en",)
    total_area = sum(
        i.bounding_box.width * i.bounding_box.height for i in scored
    ) or 1

    def is_garbage(i: InstText) -> bool:
        script = detector.detect_script(i.text or "")
        if script is None:
            return True
        # latin-only readings from an English reader on large regions
        # are almost always CJK misreads — real latin signage is rare on
        # storefront-scale crops. flag them as garbage so the probe runs.
        if script == "latin" and reader_langs == ("en",):
            area_frac = (i.bounding_box.width * i.bounding_box.height) / total_area
            if area_frac >= 0.005 and (i.confidence or 0) < 0.7:
                return True
        return (i.confidence or 0) < 0.4

    garbage_insts = [i for i in scored if is_garbage(i)]
    has_big_garbage = any(
        (i.bounding_box.width * i.bounding_box.height) / total_area >= 0.01
        and detector.detect_script(i.text or "") is None
        for i in scored
    )
    # all-latin trigger: when an English-only reader produced zero
    # non-latin script detections, the image is either genuinely latin
    # or CJK misread as latin. always probe — false positives are
    # harmless (CJK probes score ~0 on real latin text).
    has_non_latin = any(
        detector.detect_script(i.text or "") not in (None, "latin")
        for i in scored
    )
    if (
        len(garbage_insts) / len(scored) < 0.35
        and not has_big_garbage
        and not (reader_langs == ("en",) and not has_non_latin and len(scored) >= 2)
    ):
        return []  # recognition looks healthy; nothing to rescue

    # probe where the problem is: the largest garbage regions
    top = sorted(
        garbage_insts,
        key=lambda i: i.bounding_box.width * i.bounding_box.height,
        reverse=True,
    )[:probe_regions]
    if not top:
        return []
    bboxes = [i.bounding_box for i in top]

    winners: List[Tuple[int, float, EasyOCRBackend]] = []
    for langset in PROBE_LANGSETS:
        try:
            candidate = EasyOCRBackend(languages=langset, gpu=engine.gpu)
            per_region = candidate.detect_in_regions(asset, bboxes)
        except Exception:
            continue
        target_scripts = EASYOCR_LANG_SCRIPTS.get(langset[0], set()) - {"latin"}
        confs = sorted(
            (
                max(
                    (_script_bearing_conf(d, target_scripts) for d in dets),
                    default=0.0,
                )
                for dets in per_region
            ),
            reverse=True,
        )
        # top-2 evidence, not the mean: probe crops are the WORST regions
        # by construction, and one decisive script-bearing hit (e.g. a
        # storefront sign at conf 1.0) must not be diluted by crops that
        # are unreadable under every charset
        top2 = confs[:2]
        score = sum(top2) / len(top2) if top2 else 0.0
        # breadth beats depth: a single high-confidence-but-implausible
        # read (e.g. Korean's hangul plausibly-shaped-but-wrong on
        # Japanese kanji) must not outrank a correctly-scripted reader
        # whose crops are merely HARDER — measured: Korean beat Japanese
        # this way on japan-street, where the ja reader read real kanji
        # at conf 0.015-0.235 (genuinely low, not implausible). count of
        # independently-corroborating regions ranks first; confidence
        # only breaks ties within that.
        hits = sum(1 for c in confs if c > 0)
        if score >= min_evidence:
            winners.append((hits, score, candidate))
    winners.sort(key=lambda w: (-w[0], -w[1]))
    return [backend for _, _, backend in winners[:max_winners]]


SURFACE_PROBE_MAX = 8


def probe_uncovered_surfaces(
    asset: Any,
    scene_regions: List[SceneRegion],
    instances: List[InstText],
    engine: "EasyOCRBackend",
    min_evidence: float = 0.3,
) -> Tuple[Optional[Tuple[str, ...]], List[RawDetection]]:
    """probe scene surfaces that contain no real detection with candidate
    CJK readers; returns (winning_langset, surface_detections) or (None, []).

    the instance-based auto-probe fails on stacked vertical signage: the
    en pass yields tiny garbage fragments, and re-reading THOSE crops
    with a tuned reader still reads fragments. the scene pre-pass sees
    the actual sign panels, and per-panel recognition of a single CJK
    character is highly reliable (measured conf 1.0 on the cjk-vertical
    fixture vs 0.08 for full-image vertical detection). the returned
    detections are in full-image coordinates and flow through
    merge_vertical_columns, so per-panel characters reassemble into one
    column region downstream.
    """
    if not scene_regions:
        return None, []
    detector = ScriptDetector()

    # health gate: a scene with several confident script-bearing reads
    # needs no rescue — probing costs a reader init per candidate set
    healthy = [
        i for i in instances
        if detector.detect_script(i.text or "") is not None
        and (i.confidence or 0) >= 0.5
    ]
    garbage_frac = (
        1.0 - len(healthy) / len(instances) if instances else 1.0
    )
    if len(healthy) >= 2 and garbage_frac < 0.35:
        return None, []

    dim = _probe_dim(asset)
    frame_area = float(dim[0] * dim[1]) if dim else None
    real_boxes = [i.bounding_box for i in healthy if i.bounding_box]

    def uncovered(r: SceneRegion) -> bool:
        return not any(
            _containment_frac(b, r.bbox) > 0.5 for b in real_boxes
        )

    _PRIORITY = {"panel": 0, "bordered_region": 0, "text_cluster": 1, "surface": 2}
    surfaces = [
        r for r in scene_regions
        if uncovered(r)
        and (
            frame_area is None
            or (r.bbox.width * r.bbox.height) / frame_area <= 0.5
        )
    ]
    surfaces.sort(key=lambda r: (
        _PRIORITY.get(r.semantic_label, 2),
        -(r.bbox.width * r.bbox.height),
    ))
    surfaces = surfaces[:SURFACE_PROBE_MAX]
    if not surfaces:
        return None, []
    bboxes = [r.bbox for r in surfaces]

    best_hits, best_score, best_langset, best_dets = -1, 0.0, None, []
    for langset in PROBE_LANGSETS:
        try:
            candidate = EasyOCRBackend(
                languages=langset, gpu=getattr(engine, "gpu", False)
            )
            per_region = candidate.detect_in_regions(asset, bboxes)
        except Exception:
            continue
        targets = EASYOCR_LANG_SCRIPTS.get(langset[0], set()) - {"latin"}
        dets: List[RawDetection] = []
        confs: List[float] = []
        for crop_dets in per_region:
            composed = _compose_crop_text(crop_dets)
            if composed is None:
                continue
            conf = _script_bearing_conf(composed, targets)
            confs.append(conf)
            if conf > 0.2:
                composed.language = langset[0]
                dets.append(composed)
        top2 = sorted(confs, reverse=True)[:2]
        score = sum(top2) / len(top2) if top2 else 0.0
        # breadth beats depth -- see _auto_probe_language's identical
        # reasoning; count of independently-corroborating panels ranks
        # first, confidence only breaks ties within that
        hits = sum(1 for c in confs if c > 0)
        if (hits, score) > (best_hits, best_score):
            best_hits, best_score, best_langset, best_dets = hits, score, langset, dets
    if best_score >= min_evidence and best_dets:
        return best_langset, best_dets
    return None, []


def _identify_languages(
    asset: Any,
    instances: List[InstText],
    engine: "EasyOCRBackend",
    max_extra_readers: int = 2,
) -> None:
    """set detected_language per instance from the recognized text's script;
    re-recognize crops whose script the primary reader could not cover.

    extra readers are expensive (~15-20 s init each on CPU), so mismatched
    regions are grouped by required language set and only the
    max_extra_readers largest groups are re-recognized.
    """
    detector = ScriptDetector()
    reader_langs = tuple(engine.languages) or ("en",)
    covered = set()
    for lang in reader_langs:
        covered |= EASYOCR_LANG_SCRIPTS.get(lang, {"latin"})

    pending: Dict[Tuple[str, ...], List[InstText]] = {}
    for inst in instances:
        script = detector.detect_script(inst.text or "")
        if script is None:
            # no script chars: this region carries NO language evidence.
            # the reader's primary language must not masquerade as a
            # detection — it would outvote genuinely identified regions
            # in source-language inference.
            inst.detected_language = None
            continue
        if script in covered:
            # low-confidence latin is usually a misread of non-latin
            # signage, not english — it gets no language vote either
            if script == "latin" and (inst.confidence or 0) < 0.3:
                inst.detected_language = None
            else:
                inst.detected_language = _script_lang_for(script, reader_langs)
        else:
            # reader charset could not have produced this properly —
            # queue the crop for re-recognition with the right set
            inst.detected_language = SCRIPT_TO_LANG.get(script, "en")
            langset = SCRIPT_TO_EASYOCR_SET.get(script)
            if langset:
                pending.setdefault(langset, []).append(inst)

    groups = sorted(pending.items(), key=lambda kv: -len(kv[1]))
    for langset, insts in groups[:max_extra_readers]:
        try:
            alt = EasyOCRBackend(languages=langset, gpu=engine.gpu)
            per_region = alt.detect_in_regions(
                asset, [i.bounding_box for i in insts]
            )
        except Exception:
            continue
        for inst, dets in zip(insts, per_region):
            composed = _compose_crop_text(dets)
            if composed is None:
                continue
            if composed.text and composed.confidence > 0.2:
                inst.text = composed.text
                inst.confidence = composed.confidence

    # latin-language disambiguation: script identity can't tell spanish
    # from english — classify the pooled latin text by stopwords/diacritics.
    # CONFIDENT latin text only: garbage recognitions of CJK signage are
    # full of accidental stopword fragments ('il', 'e', 'di'...) that
    # otherwise vote in phantom languages.
    latin_insts = [
        i for i in instances
        if detector.detect_script(i.text or "") == "latin"
        and (i.confidence or 0) >= 0.5
    ]
    guess = guess_latin_language([i.text or "" for i in latin_insts])
    if guess:
        for inst in latin_insts:
            inst.detected_language = guess


def _identify_languages_paddle(
    asset: Any,
    instances: List[InstText],
    engine: "PaddleOCRBackend",
    max_extra_readers: int = 2,
) -> None:
    """set detected_language per instance from recognized text script.

    PaddleOCR handles multi-script recognition internally, so this is
    purely a text-analysis pass: detect the script of each instance's
    recognized text and map it to a tofu language code. no re-recognition
    is needed unless the script is entirely absent from the text (which
    suggests a misread — but PaddleOCR's charset coverage is broad enough
    that we trust its output).
    """
    detector = ScriptDetector()
    paddle_lang = engine.lang if hasattr(engine, "lang") else "en"
    tofu_lang = PaddleOCRBackend.PADDLE_TO_TOFU.get(paddle_lang, paddle_lang)

    for inst in instances:
        script = detector.detect_script(inst.text or "")
        if script is None:
            inst.detected_language = None
            continue
        if script == "latin" and (inst.confidence or 0) < 0.3:
            inst.detected_language = None
        else:
            inst.detected_language = _script_lang_for(script, (tofu_lang,))

    # latin-language disambiguation (same logic as EasyOCR path)
    latin_insts = [
        i for i in instances
        if detector.detect_script(i.text or "") == "latin"
        and (i.confidence or 0) >= 0.5
    ]
    guess = guess_latin_language([i.text or "" for i in latin_insts])
    if guess:
        for inst in latin_insts:
            inst.detected_language = guess


def refine_langset(
    instances: List[InstText], engine: OCRBackend
) -> Optional[Tuple[str, ...]]:
    """easyocr language set for a language-adaptive second detection pass,
    or None when the current engine's charset already covers the scene.

    the dominant detected language is chosen by region area (a storefront
    sign outvotes incidental fragments), but a SINGLE instance — however
    large its box — is never enough corroborating evidence to commit to
    a re-detection language: a lone script misread (Korean hangul
    plausibly matching a Chinese character's strokes) can otherwise
    hijack the whole adaptive stage-2 pass on its own, before
    probe_uncovered_surfaces's more careful surface-level probe (see its
    own breadth-of-evidence ranking) ever gets a chance to run — measured
    live on china-street: one 'ko'-labeled instance (a Korean misread of
    real Chinese signage) outvoted several genuine 'en' fragments purely
    on box area, and detect()'s adaptive stage only calls
    probe_uncovered_surfaces when THIS function returns None. requiring
    a second, independent instance before committing mirrors the same
    principle _auto_probe_language/probe_uncovered_surfaces/
    _disambiguate_ja_zh already apply.

    a second pass is only worth its cost when that language's script is
    one the engine could not emit — recognition under the wrong charset
    produces garbage that the hallucination pruner then deletes, which
    is where recall dies.
    """
    if not isinstance(engine, EasyOCRBackend):
        return None
    votes: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for inst in instances:
        if not inst.detected_language or inst.bounding_box is None:
            continue
        area = max(1, inst.bounding_box.width * inst.bounding_box.height)
        votes[inst.detected_language] = votes.get(inst.detected_language, 0.0) + area
        counts[inst.detected_language] = counts.get(inst.detected_language, 0) + 1
    if not votes:
        return None
    dominant = max(votes, key=lambda k: votes[k])
    if counts[dominant] < MIN_DISAMBIGUATION_EVIDENCE:
        return None
    target = expand_langset([dominant])
    needed: set = set()
    for lang in target:
        needed |= EASYOCR_LANG_SCRIPTS.get(_to_easyocr_lang(lang), {"latin"})
    covered: set = set()
    for lang in engine.languages:
        covered |= EASYOCR_LANG_SCRIPTS.get(lang, {"latin"})
    if needed - covered:
        return target
    return None


def union_prefer_primary(
    primary: List[RawDetection], secondary: List[RawDetection]
) -> List[RawDetection]:
    """union of two detection sets where PRIMARY is authoritative on
    overlaps regardless of confidence.

    plain merge_detections keeps the higher confidence, which is exactly
    wrong across charsets: an english reader recognizes a Korean glyph as
    '4' at conf 0.95, beating the tuned reader's correct '사' at 0.8.
    confidence is only comparable within one charset.
    """
    out = list(primary)
    kept_boxes = [_polygon_bbox(d.polygon) for d in out]
    for det in secondary:
        db = _polygon_bbox(det.polygon)
        if not any(_overlap_frac(db, kb) > 0.5 for kb in kept_boxes):
            out.append(det)
            kept_boxes.append(db)
    return out


def run_multipass(
    engine: OCRBackend, asset: Any
) -> List[RawDetection]:
    """three CRAFT passes at descending thresholds with cross-pass NMS.

    EasyOCR benefits from threshold sweeps; PaddleOCR's DB detector handles
    scale internally, so a single detect call is sufficient when it is active.
    """
    if not isinstance(engine, EasyOCRBackend):
        return engine.detect(asset)
    detections: List[RawDetection] = []
    for text_threshold, low_text in PASS_THRESHOLDS:
        passed = engine.detect(
            asset, text_threshold=text_threshold, low_text=low_text
        )
        detections = merge_detections(detections, passed)
    return detections


# coarse-to-fine zoom pass limits: surfaces above this fraction of the
# frame are the frame (re-detecting them buys nothing), and the pass is
# capped so pathological surface counts can't blow up runtime
ZOOM_MAX_SURFACE_FRAC = 0.5
ZOOM_MAX_SURFACES = 14
ZOOM_SCALE = 2
ZOOM_PAD = 8


def zoom_detect(
    engine: OCRBackend,
    asset: Any,
    scene_regions: List[SceneRegion],
) -> List[RawDetection]:
    """coarse-to-fine detection: re-detect inside each candidate scene
    surface at ZOOM_SCALE× resolution and map the boxes back.

    full-frame CRAFT resolves signage down to its heatmap resolution;
    distant/dense signs below that ceiling are recovered by zooming into
    the surfaces the scene pre-pass proposed (the standard multi-scale
    zoom-in strategy for dense scene text). the returned FINE boxes are
    meant to be unioned with fine-priority: a coarse full-frame box that
    spans several signs is replaced by the zoom pass's per-sign boxes.
    """
    from tofu.utils.imaging import load_rgb
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    img = load_rgb(asset)
    if img is None or not scene_regions:
        return []
    h, w = img.shape[:2]
    frame_area = float(h * w)
    primary_lang = getattr(engine, "primary_language", engine.languages[0] if engine.languages else None)

    _ZOOM_PRIORITY = {"text_cluster": 0, "panel": 1, "bordered_region": 2, "surface": 3}
    surfaces = [
        r for r in scene_regions
        if (r.bbox.width * r.bbox.height) / frame_area <= ZOOM_MAX_SURFACE_FRAC
    ]
    surfaces.sort(key=lambda r: (_ZOOM_PRIORITY.get(r.semantic_label, 3), -(r.bbox.width * r.bbox.height)))
    fine: List[RawDetection] = []
    for region in surfaces[:ZOOM_MAX_SURFACES]:
        bbox = region.bbox
        x0, y0 = max(0, bbox.x - ZOOM_PAD), max(0, bbox.y - ZOOM_PAD)
        x1 = min(w, bbox.x + bbox.width + ZOOM_PAD)
        y1 = min(h, bbox.y + bbox.height + ZOOM_PAD)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = np.asarray(
            Image.fromarray(img[y0:y1, x0:x1]).resize(
                ((x1 - x0) * ZOOM_SCALE, (y1 - y0) * ZOOM_SCALE),
                Image.BICUBIC,
            )
        )
        try:
            # single mid-threshold pass: the multipass sweep already ran at
            # frame scale; the zoom pass is for resolution, not thresholds.
            # Extra args are harmless for PaddleOCRBackend (it ignores them).
            dets = engine.detect(crop, text_threshold=0.5, low_text=0.3)
        except Exception:
            continue
        for d in dets:
            fine.append(RawDetection(
                polygon=[
                    (int(x / ZOOM_SCALE + x0), int(y / ZOOM_SCALE + y0))
                    for x, y in d.polygon
                ],
                text=d.text,
                confidence=d.confidence,
                language=primary_lang,
            ))
    return fine


def second_look(
    asset: Any,
    instances: List[InstText],
    engine: OCRBackend,
    conf_threshold: float = 0.55,
    max_regions: int = 24,
) -> int:
    """re-recognize low-confidence regions from upscaled crops, verify the
    read with a 180° rotation, and keep the better read. returns the number
    of regions improved.

    full-frame recognition reads each box at frame resolution;
    detect_in_regions re-reads the crop at the recognizer's operating
    height (tiny crops are upscaled), which is a strictly better view of
    the same pixels. the 180° check catches hallucinations on symmetric
    or noisy glyphs when the rotated read is unrecognizably different.
    """
    weak = [
        i for i in instances
        if i.bounding_box is not None
        and (i.confidence or 0) < conf_threshold
    ]
    weak.sort(key=lambda i: -(i.bounding_box.width * i.bounding_box.height))
    weak = weak[:max_regions]
    if not weak:
        return 0
    try:
        polys = [
            i.segmentation_mask.polygon if i.segmentation_mask else None
            for i in weak
        ]
        per_region = engine.detect_in_regions(
            asset, [i.bounding_box for i in weak], polygons=polys
        )
    except Exception:
        return 0
    try:
        import numpy as np
        from PIL import Image
        img = load_rgb(asset)
    except Exception:
        img = None

    def _rotated_text(crop: Any) -> Optional[str]:
        try:
            rot = np.rot90(crop, 2) if crop is not None else None
            if rot is None:
                return None
            rots = engine.detect(rot)
            comp = _compose_crop_text(rots)
            return (comp.text or "").strip() if comp else None
        except Exception:
            return None

    improved = 0
    for inst, dets in zip(weak, per_region):
        composed = _compose_crop_text(dets)
        if composed is None or not (composed.text or "").strip():
            continue
        # verification: a hallucinated read tends to be gibberish when
        # the crop is inverted. if the inverted read differs drastically,
        # do not trust this re-read.
        verified = True
        if img is not None and inst.bounding_box is not None:
            b = inst.bounding_box
            try:
                crop = img[b.y:b.y + b.height, b.x:b.x + b.width]
                rot_text = _rotated_text(crop)
                if rot_text is not None:
                    ed = _norm_ed(composed.text, rot_text)
                    if ed > 0.5:
                        verified = False
            except Exception:
                pass
        if not verified:
            continue
        if composed.confidence > (inst.confidence or 0):
            inst.text = composed.text
            inst.confidence = composed.confidence
            improved += 1
    return improved


def detect(
    asset: Any,
    asset_info: Optional[AssetInfo] = None,
    backend: Optional[OCRBackend] = None,
    languages: Optional[Sequence[str]] = None,
    scene_regions: Optional[List[SceneRegion]] = None,
    multipass: bool = True,
    scene_filter: bool = True,
    identify_languages: bool = True,
    max_extra_readers: int = 2,
    prune_garbage: bool = True,
    adaptive: bool = True,
    zoom: bool = True,
    vertical_split: bool = True,
    polish: bool = True,
    savor: bool = True,
) -> TextManifest:
    """detect and localize text instances in the asset.

    args:
        asset: image (or, later, video) — file path, URL, or in-memory.
        asset_info: ingestion descriptor; inferred from the asset if omitted.
        backend: explicit engine adapter; defaults to get_backend().
        languages: source-language hints; builds a language-tuned
            EasyOCR backend (expand_langset) when no explicit backend is
            given. hints are the reliable path to correct recognition of
            non-latin scripts — script identification can only refine
            what the reader's charset could express.
        scene_regions: scene pre-pass surfaces. when provided (non-empty),
            detections not contained (>50% of their area) in any surface
            are discarded as false positives. empty/None bypasses the
            filter entirely.
        multipass: run three detection passes at descending CRAFT
            thresholds (standard/stylized/hard) with cross-pass NMS.
            EasyOCR backends only; ~3x detection time on CPU.
        scene_filter: disable to keep detections outside scene surfaces.
        identify_languages: classify each region's script from its
            recognized text and set detected_language accordingly,
            re-recognizing crops the primary reader could not cover.
        max_extra_readers: cap on additional language-set readers the
            identification step may initialize (each ~15-20 s on CPU).
        adaptive: language-adaptive two-stage detection. when the first
            (unhinted) stage identifies a dominant language whose script
            the reader's charset could not emit, the FULL multipass
            detection is re-run with a language-tuned reader and the two
            stages are unioned (tuned stage authoritative on overlaps).
            this recovers the regions the wrong-charset stage recognized
            as garbage and consequently pruned. no-op when languages/
            backend are given (already tuned) or the scene is covered.
        zoom: coarse-to-fine pass — re-detect inside candidate scene
            surfaces at 2x resolution; fine boxes are authoritative over
            coarse frame-scale boxes they overlap (fixes merged multi-
            sign boxes and recovers signage below CRAFT's frame-scale
            resolving power).
        vertical_split: re-segment detections whose box is far taller
            than wide (likely CRAFT over-merging several stacked
            vertical-CJK characters into one box, defeating single-line
            recognition regardless of charset) into per-character bands
            and re-recognize each individually; the resulting fragments
            flow through the normal merge_vertical_columns() reassembly.
            see `_split_tall_detections`.
        polish: second-look recognition — re-read low-confidence regions
            from upscaled crops with the final engine and keep the
            better read.
        savor: taste-test the FINAL recognized text for digit/letter
            glyph confusion a CRNN can be confidently wrong about (5/S,
            0/O, 1/I, 8/B, 2/Z, 6/G — e.g. "5pm" read as "Spm"). unlike
            `polish`, this never re-runs the recognizer (the same model
            on the same pixels reproduces the same mistake); it verifies
            a candidate correction against the glyph's own pixel shape
            before ever rewriting anything. see `savor.taste()`.

    returns:
        TextManifest with one InstText per detected region, sorted into
        reading order (top-to-bottom, then left-to-right). for images,
        instances carry frame_index=None; video tracking (track_id /
        temporal_span via per-frame IoU linking) lands with the video
        pipeline.
    """
    start = time.time()
    asset_info = asset_info or infer_asset_info(asset)

    engine = backend
    if engine is None:
        if languages:
            if _engine_from_env() == "paddleocr":
                engine = (
                    PaddleOCRBackend(languages=languages, gpu=False)
                    if PaddleOCRBackend.is_available() else NullBackend()
                )
            else:
                try:
                    import easyocr  # noqa: F401
                    engine = EasyOCRBackend(languages=expand_langset(languages))
                except ImportError:
                    engine = NullBackend()
        else:
            engine = get_backend()

    # multi-pass detection: EasyOCR uses threshold sweeps, PaddleOCR a
    # single detect call (run_multipass dispatches internally).
    if multipass:
        detections = run_multipass(engine, asset)
    else:
        detections = engine.detect(asset)

    manifest = build_manifest(
        asset, detections,
        asset_info=asset_info,
        engine=engine,
        scene_regions=scene_regions,
        scene_filter=scene_filter,
        identify_languages=identify_languages,
        max_extra_readers=max_extra_readers,
        prune_garbage=prune_garbage,
        start=start,
    )

    # language-adaptive stage 2: only for unhinted runs whose identified
    # dominant language the stage-1 charset could not express. keyed on
    # the ENGINE being the plain english-charset default — callers that
    # passed language hints or a tuned backend already chose a charset.
    final_engine: OCRBackend = engine
    if (
        adaptive and not languages and identify_languages
        and isinstance(engine, EasyOCRBackend)
        and tuple(getattr(engine, "languages", ("en",))) == ("en",)
    ):
        target = refine_langset(manifest.instances, engine)
        # instance evidence found nothing to refine on — probe scene
        # surfaces the detector left uncovered (vertical CJK signage
        # fragments below the instance probe's resolving power)
        surface_dets: List[RawDetection] = []
        if target is None and scene_regions:
            target, surface_dets = probe_uncovered_surfaces(
                asset, scene_regions, manifest.instances, engine
            )
        if target:
            tuned = EasyOCRBackend(languages=target, gpu=engine.gpu)
            second = (
                run_multipass(tuned, asset) if multipass else tuned.detect(asset)
            )
            # per-surface reads are authoritative: they saw each panel at
            # crop resolution where full-frame detection reads fragments
            if surface_dets:
                second = union_prefer_primary(surface_dets, second)
            detections = union_prefer_primary(second, detections)
            final_engine = tuned
            manifest = build_manifest(
                asset, detections,
                asset_info=asset_info,
                engine=tuned,
                scene_regions=scene_regions,
                scene_filter=scene_filter,
                identify_languages=identify_languages,
                max_extra_readers=max_extra_readers,
                prune_garbage=prune_garbage,
                start=start,
            )

    # coarse-to-fine zoom pass with the final engine: fine boxes are
    # authoritative — a coarse frame-scale box that spans several signs is
    # replaced by its per-sign zoom boxes. runs for any non-null backend.
    if zoom and scene_regions and not isinstance(final_engine, NullBackend):
        fine = zoom_detect(final_engine, asset, scene_regions)
        if fine:
            detections = union_prefer_primary(fine, detections)
            manifest = build_manifest(
                asset, detections,
                asset_info=asset_info,
                engine=final_engine,
                scene_regions=scene_regions,
                scene_filter=scene_filter,
                identify_languages=identify_languages,
                max_extra_readers=max_extra_readers,
                prune_garbage=prune_garbage,
                start=start,
            )

    # vertical-stack re-split: runs LAST among the detection-refinement
    # passes, against whichever engine/detections survived every
    # earlier stage, so it benefits from the adaptive/zoom passes' own
    # language and coverage improvements rather than duplicating them.
    # see _split_tall_detections's module-level note.
    if vertical_split and isinstance(final_engine, EasyOCRBackend):
        split = _split_tall_detections(asset, final_engine, detections)
        if split is not None:
            detections = split
            manifest = build_manifest(
                asset, detections,
                asset_info=asset_info,
                engine=final_engine,
                scene_regions=scene_regions,
                scene_filter=scene_filter,
                identify_languages=identify_languages,
                max_extra_readers=max_extra_readers,
                prune_garbage=prune_garbage,
                start=start,
            )

    # second-look recognition on the surviving weak regions
    if polish and manifest.instances and not isinstance(final_engine, NullBackend):
        second_look(asset, manifest.instances, final_engine)

    # Savor's taste test runs LAST, once, on the FINAL text — after
    # every detection/refinement/re-read pass above has had its say.
    # best-effort: a tasting failure must never fail detection itself.
    if savor and manifest.instances:
        try:
            from tofu.layers.savor import taste
            taste(asset, manifest.instances)
        except Exception:
            pass

    return manifest


def _disambiguate_ja_zh(instances: List[InstText]) -> None:
    """Japanese vs Chinese disambiguation, in place:
    - if any instance contains kana, the scene is Japanese — kanji-only
      instances labeled "zh-cn"/"ko" are actually Japanese.
    - if NO instance contains kana but instances were labeled "ja"
      (han-only text read by the Japanese reader), the scene is
      POSSIBLY actually Chinese — the ja reader covers kanji but kana is
      the definitive Japanese signal.

    kana is the only unambiguous ja-vs-zh signal (only Japanese uses
    it), so its PRESENCE is normally trusted from even a single
    instance — PROVIDED that instance is confident enough to trust as
    real: a low-confidence misread of a tiny/blurry garbage crop can
    accidentally shape-match a kana glyph, and unconditionally trusting
    it can permanently lock a scene as "Japanese" even when the
    overwhelming majority of real, legible text is Chinese (measured
    live on china-street: a handful of near-zero-confidence kana-shaped
    misreads among many garbage fragments blocked the correct zh-cn
    downgrade from ever firing, despite 2+ real, confident han-only
    instances). the confidence floor mirrors the one
    _identify_languages already applies to low-confidence latin.

    kana's ABSENCE (once low-confidence noise is filtered out) is much
    weaker evidence — with only a handful of real instances recovered
    (the norm on hard dense-CJK scenes), "no kana observed" often just
    means too little text survived to contain it, not that the scene is
    genuinely kana-free. measured: an explicit, correct
    source_lang="ja" hint still got silently downgraded to zh-cn this
    way on a 2-3-instance manifest. requiring MIN_DISAMBIGUATION_EVIDENCE
    real CJK-script instances before the no-kana downgrade fires lets an
    upstream language selection (itself evidence-count-ranked — see
    _auto_probe_language/probe_uncovered_surfaces) stand when too little
    text survived to meaningfully contradict it.
    """
    detector = ScriptDetector()
    has_kana = any(
        detector.detect_script(i.text or "") == "japanese"
        and (i.confidence or 0) >= MIN_KANA_CONFIDENCE
        for i in instances
    )
    if has_kana:
        for inst in instances:
            if inst.detected_language in ("zh-cn", "ko"):
                script = detector.detect_script(inst.text or "")
                if script in ("han", "japanese", "hangul"):
                    inst.detected_language = "ja"
        return
    cjk_evidence = sum(
        1 for i in instances
        if detector.detect_script(i.text or "") in ("han", "japanese", "hangul")
        and (i.confidence or 0) >= MIN_KANA_CONFIDENCE
    )
    if cjk_evidence >= MIN_DISAMBIGUATION_EVIDENCE:
        # no kana anywhere, and enough real CJK text to trust that
        # absence: relabel ja→zh-cn for han-only instances
        for inst in instances:
            if inst.detected_language == "ja":
                script = detector.detect_script(inst.text or "")
                if script == "han":
                    inst.detected_language = "zh-cn"


def _prune_hallucinations(instances: List[InstText]) -> List[InstText]:
    """drop symbol-noise regions and renumber the survivors.

    a region is a hallucination when its text is empty, or contains no
    script characters AND no digits AND was recognized below 0.4 —
    pure punctuation/symbol junk. digit-only regions are KEPT: prices,
    phone numbers, and address plates are real localizable content.
    """
    detector = ScriptDetector()
    kept: List[InstText] = []
    for inst in instances:
        text = (inst.text or "").strip()
        if not text:
            continue
        has_script = detector.detect_script(text) is not None
        has_digit = any(ch.isdigit() for ch in text)
        if not has_script and not has_digit and (inst.confidence or 0) < 0.4:
            continue
        kept.append(inst)
    for order, inst in enumerate(kept):
        inst.id = f"r{order + 1}"
        inst.reading_order = order
    return kept


def build_manifest(
    asset: Any,
    detections: List[RawDetection],
    asset_info: Optional[AssetInfo] = None,
    engine: Optional[OCRBackend] = None,
    scene_regions: Optional[List[SceneRegion]] = None,
    scene_filter: bool = True,
    identify_languages: bool = True,
    max_extra_readers: int = 2,
    prune_garbage: bool = True,
    merge_columns: bool = True,
    start: Optional[float] = None,
) -> TextManifest:
    """assemble a TextManifest from raw detections.

    shared by detect() and the streaming endpoint so both produce
    identical manifests from identical detections: scene filtering,
    reading-order sort, instance construction, language identification.
    """
    if start is None:
        start = time.time()
    asset_info = asset_info or infer_asset_info(asset)

    # stacked vertical CJK signage: unify per-character fragments into
    # single column boxes BEFORE filtering — the merged crop is what the
    # language rescue re-recognizes, and one region per sign is what the
    # capture table should show
    if merge_columns:
        detections = merge_vertical_columns(detections)

    # scene constraint, PHASE A — survive-for-rescue: a deliberately
    # lenient GEOMETRIC-ONLY gate ("is this plausibly text-shaped,
    # geometrically," not "did we read it right"). containment fraction,
    # NOT IoU — small text inside a large sign must score ~1.0. an empty
    # region list bypasses the filter entirely.
    #
    # the STRICT confidence-based prune moves to PHASE B, after language
    # rescue (below) has had a chance to re-recognize every surviving
    # candidate with the correct charset. recognition confidence from
    # the wrong charset — or even the RIGHT charset on a hard crop
    # (bloom/glow from saturated neon signage measurably caps EasyOCR's
    # own confidence estimate) — is not a reliable signal for "is this
    # real text" at this point in the pipeline: measured on
    # japan-street's banner, a ja-charset read got 6/7 characters right
    # at confidence 0.015, LOWER than the wrong-charset English read's
    # garbage at 0.087. pruning by confidence before rescue has run
    # discards exactly the candidates rescue exists to save.
    if scene_filter and scene_regions:
        detections = [
            det for det in detections
            if det.confidence >= 0.5
            or any(
                _containment_frac(_polygon_bbox(det.polygon), region.bbox) > 0.5
                for region in scene_regions
            )
        ]

    # reading order: top-to-bottom, then left-to-right
    def _key(d: RawDetection) -> Tuple[int, int]:
        ys = [p[1] for p in d.polygon]
        xs = [p[0] for p in d.polygon]
        return (min(ys), min(xs))

    instances: List[InstText] = []
    for order, det in enumerate(sorted(detections, key=_key)):
        xs = [p[0] for p in det.polygon]
        ys = [p[1] for p in det.polygon]
        instances.append(
            InstText(
                id=f"r{order + 1}",
                bounding_box=BBox(
                    x=min(xs), y=min(ys),
                    width=max(xs) - min(xs), height=max(ys) - min(ys),
                ),
                segmentation_mask=Mask(
                    polygon=det.polygon, confidence=det.confidence
                ),
                text=det.text,
                confidence=det.confidence,
                detected_language=_det_lang_from_engine(det, engine),
                reading_order=order,
                frame_index=None,  # video: set per frame once tracking lands
            )
        )

    # per-region language identification from recognized script.
    # this runs for ALL backends — script detection operates on the
    # recognized text string, not on the engine internals.
    if identify_languages and instances:
        if isinstance(engine, EasyOCRBackend):
            # unhinted garbage rescue: an english-only pass over CJK signage
            # yields garbage no script analysis can fix — probe candidate
            # readers and re-recognize every region with each winner. mixed
            # ko/ja scenes get two winners; per region the best script-bearing
            # recognition wins, and script id labels ja vs ko correctly.
            if tuple(engine.languages) == ("en",):
                detector = ScriptDetector()
                winners = _auto_probe_language(asset, instances, engine)
                bboxes = [i.bounding_box for i in instances]
                for backend_w in winners:
                    try:
                        per_region = backend_w.detect_in_regions(asset, bboxes)
                    except Exception:
                        continue
                    targets = EASYOCR_LANG_SCRIPTS.get(
                        backend_w.languages[0], set()
                    ) - {"latin"}
                    for inst, dets in zip(instances, per_region):
                        # compose ALL detections in the crop, not just the
                        # best one — a vertical column yields one detection
                        # per character and picking a single winner truncates
                        # the sign to one character (e.g. '맥주' → '주')
                        composed = _compose_crop_text(dets)
                        if composed is None:
                            continue
                        conf = _script_bearing_conf(composed, targets)
                        old_garbage = detector.detect_script(inst.text or "") is None
                        # replace only with real script evidence, and never
                        # overwrite genuine content that scored higher
                        if conf > 0.2 and (old_garbage or conf > (inst.confidence or 0)):
                            inst.text = composed.text
                            inst.confidence = composed.confidence
                if winners:
                    engine = winners[0]  # script id below sees a rescued charset

                # Japanese priority: if the Japanese winner produced kana
                # on any region, the scene is Japanese — Korean's hangul
                # is a misrecognition of kanji/kana. Re-recognize ALL
                # regions with the Japanese backend to overwrite hangul,
                # and set Japanese as the primary engine so
                # _identify_languages uses the correct charset coverage.
                ja_backend = next(
                    (w for w in winners if w.languages[0] == "ja"), None
                )
                if ja_backend:
                    has_kana = any(
                        detector.detect_script(i.text or "") == "japanese"
                        for i in instances
                    )
                    if has_kana:
                        try:
                            ja_regions = ja_backend.detect_in_regions(asset, bboxes)
                        except Exception:
                            ja_regions = None
                        if ja_regions:
                            ja_targets = EASYOCR_LANG_SCRIPTS.get("ja", set()) - {"latin"}
                            for inst, dets in zip(instances, ja_regions):
                                composed = _compose_crop_text(dets)
                                if composed is None:
                                    continue
                                conf = _script_bearing_conf(composed, ja_targets)
                                # always overwrite when Japanese produces
                                # kana/kanji — kana is a definitive signal
                                # that Korean's hangul was wrong
                                if conf > 0.2:
                                    inst.text = composed.text
                                    inst.confidence = composed.confidence
                        engine = ja_backend
            _identify_languages(asset, instances, engine, max_extra_readers)
        elif isinstance(engine, PaddleOCRBackend):
            _identify_languages_paddle(asset, instances, engine, max_extra_readers)

        # Japanese vs Chinese disambiguation:
        # - if any instance contains kana, the scene is Japanese —
        #   kanji-only instances labeled "zh-cn" are actually Japanese.
        # - if NO instance contains kana but instances were labeled "ja"
        #   (han-only text read by the Japanese reader), the scene is
        #   actually Chinese — the ja reader covers kanji but kana is
        #   the definitive Japanese signal.
        _disambiguate_ja_zh(instances)

    # scene constraint, PHASE B — final confidence prune: now that
    # language rescue (above) has had its chance to rewrite inst.text/
    # inst.confidence with the correct charset, apply the confidence bar
    # phase A deferred. anything still low-confidence after rescue is
    # pruned exactly as the old single-phase filter would have; anything
    # rescue brought above the bar survives where the old filter would
    # have discarded it before rescue ever got a chance to run.
    #
    # adaptive thresholds: "panel" and "text_cluster" regions get a lower
    # bar (0.30) because text is very likely inside them — the scene
    # detector already confirmed a text-bearing surface. "bordered_region"
    # and "surface" keep the standard 0.50 bar.
    if scene_filter and scene_regions:
        _SCENE_CONF = {"panel": 0.30, "text_cluster": 0.30, "bordered_region": 0.40}
        instances = [
            inst for inst in instances
            if (inst.confidence or 0) >= 0.5
            or any(
                _containment_frac(inst.bounding_box, region.bbox) > 0.5
                and (inst.confidence or 0) >= _SCENE_CONF.get(region.semantic_label, 0.50)
                for region in scene_regions
            )
        ]

    # hallucination pruning runs AFTER rescue/identification so regions
    # that were salvageable got their chance first
    if prune_garbage:
        instances = _prune_hallucinations(instances)

    return TextManifest(
        # asset_info.source is a full file path (server/main.py's
        # convention: UPLOAD_DIR/{asset_id}.ext) -- the STEM is the real
        # asset_id; using the raw path leaked local filesystem paths into
        # every consumer keyed on manifest.asset_id (QAReport's
        # per_asset_instance_score, and now Memory's tm_records), caught
        # via a live TM round-trip where a stored record's asset_id
        # turned out to be an absolute Windows path instead of the clean
        # id the rest of the app uses everywhere else
        asset_id=(Path(asset_info.source).stem if asset_info.source
                  else f"asset-{uuid.uuid4().hex[:8]}"),
        total_regions=len(instances),
        instances=instances,
        img_dim=_probe_dim(asset),
        scene_regions=list(scene_regions) if scene_regions else [],
        asset_type=asset_info.asset_type,
        frame_count=asset_info.frame_count,
        fps=asset_info.fps,
        duration=asset_info.duration,
        prcssng_time=time.time() - start,
    )
