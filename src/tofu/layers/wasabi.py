## 🍢 wasabi — Japanese/simplified-Chinese glyph normalization
"""
PaddleOCR's "japan" language selector doesn't get a Japanese-specific
recognition model: in the installed paddleocr build, `lang="japan"`
resolves to the SAME shared PP-OCRv6_medium_rec model used for
`"ch"`/`"chinese_cht"`/`"en"` (confirmed by inspecting the cached
model's own README, which lists `language: [en, zh]` -- Japanese isn't
even an officially supported language of it). That model's CTC decode
vocabulary contains both a simplified-Chinese-only glyph form and the
correct Japanese shinjitai form as separate valid output tokens, with
no language-conditioning to prefer the right one -- so a `ja`-labeled
read can confidently emit a Chinese-only character (measured live:
`劇場通り` read back as `剧場通`, `焼肉` read back as `烧肉`). This is a
real, confirmed characteristic of the model itself, not something
fixable by how ToFU calls it -- so this module corrects it after the
fact instead.

`season()` is the module's one public entry point (mirrors every other
layer's single-verb API: cicerone.detect, scene.analyze, savor.taste,
menu.browse) -- called once by cicerone.detect(), after the PaddleOCR
rescue pass and Savor have already had their say, so it corrects
whatever text actually made it into the final manifest regardless of
which engine or pass produced it.
"""

from typing import Dict, List

from tofu.core.types import InstText

# curated, growable pairs of (simplified-Chinese-only glyph, Japanese
# shinjitai equivalent) confirmed to be confused by PaddleOCR's shared
# ja/zh recognizer. deliberately NOT a full Unihan variant table --
# most simplified forms actually MATCH Japanese shinjitai (both
# diverged independently from the same traditional form and landed on
# the same simplification), so only genuinely observed divergent pairs
# belong here. add a pair only once it's actually been seen in
# production output, the same discipline menu.py's gazetteer follows.
SIMPLIFIED_TO_JAPANESE: Dict[str, str] = {
    "剧": "劇",  # theater: 劇場通り read back as 剧場通
    "烧": "焼",  # burn/grill: 焼肉/焼皮 read back as 烧肉/烧皮
    "岛": "島",  # island: 下島 read back as 下岛
}


def normalize_japanese_kanji(text: str) -> str:
    """swap any known simplified-Chinese-only glyph in `text` for its
    Japanese shinjitai equivalent. characters not in the table (the
    vast majority, including every shinjitai form that already matches
    its simplified counterpart) pass through unchanged."""
    if not text:
        return text
    return "".join(SIMPLIFIED_TO_JAPANESE.get(ch, ch) for ch in text)


def season(instances: List[InstText]) -> int:
    """the full normalization pass, run once across every ja-labeled
    instance. corrections are recorded on inst.ocr_correction using the
    same {applied, original_text, corrected_text, reason} shape
    savor/menu already write, so review/audit stays in one place
    regardless of which layer made the call. returns the number of
    instances corrected."""
    corrected = 0
    for inst in instances:
        if (inst.detected_language or inst.language) != "ja":
            continue
        text = inst.text or ""
        if not text:
            continue
        fixed = normalize_japanese_kanji(text)
        if fixed == text:
            continue
        inst.ocr_correction = {
            "applied": True,
            "original_text": text,
            "corrected_text": fixed,
            "reason": "simplified-Chinese glyph form normalized to Japanese shinjitai",
        }
        inst.text = fixed
        corrected += 1
    return corrected
