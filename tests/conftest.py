## 🍢 test bootstrap
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = ROOT / "tests" / "fixtures"

# Where each platform keeps its fonts. Several suites need a real face on
# disk -- glyph coverage, shaping, vertical CJK -- and hard-coding
# C:/Windows/Fonts made every one of them silently skip off Windows, which
# is exactly where CI runs.
_FONT_DIRS = (
    Path("C:/Windows/Fonts"),                    # Windows
    Path("/usr/share/fonts"),                    # Linux
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("/System/Library/Fonts"),               # macOS
    Path("/Library/Fonts"),
)


def system_font(*names: str) -> Optional[Path]:
    """First of `names` that exists in any platform font directory.

    Pass equivalents in preference order -- ``system_font("arial.ttf",
    "DejaVuSans.ttf")`` -- so a Latin test runs on Windows AND on a Linux
    runner instead of skipping. Returns None when none is found, which is
    the caller's cue to skip: a missing Devanagari face on a CI box is a
    legitimate skip, whereas a missing Latin face means the environment is
    broken.

    Subdirectories are searched too, because Linux distributions nest fonts
    under per-foundry folders (``/usr/share/fonts/truetype/dejavu/...``).
    """
    for name in names:
        for directory in _FONT_DIRS:
            if not directory.is_dir():
                continue
            direct = directory / name
            if direct.is_file():
                return direct
            try:
                found = next(directory.rglob(name), None)
            except OSError:
                found = None
            if found is not None:
                return found
    return None


#: A Latin face that should exist on any developer machine or CI runner.
LATIN_FONT = system_font(
    "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"
)

#: Devanagari and Thai faces -- genuinely optional; absent on most CI images.
DEVANAGARI_FONT = system_font(
    "Nirmala.ttc", "NotoSansDevanagari-Regular.ttf", "lohit_deva.ttf"
)
THAI_FONT = system_font("LeelawUI.ttf", "NotoSansThai-Regular.ttf", "Waree.ttf")
