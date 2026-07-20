## 🍢 phash — perceptual hashing for visual translation-memory matching
## vieuxtiful
"""
DCT-based perceptual hash (Zauner 2010, "Implementation and Benchmarking
of Perceptual Image Hash Functions"): resize to 32x32 grayscale, take the
2D DCT, keep the top-left 8x8 low-frequency block (dropping the DC term),
threshold each AC coefficient against their median. Robust to the crop-
scale/compression differences between two captures of the same sign,
unlike a raw pixel hash.

kept dependency-soft like imaging.py: returns None when Pillow/numpy/cv2
are unavailable or the crop is degenerate, so callers can treat a missing
hash as "skip the visual match tier" rather than crashing.
"""

from typing import Any, Optional

HASH_SIZE = 32   # resize target (px)
LOW_FREQ = 8      # keep the top-left LOW_FREQ x LOW_FREQ DCT block


def phash(asset: Any) -> Optional[str]:
    """perceptual hash of an image crop as a 64-bit hex string, or None."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    from tofu.utils.imaging import load_rgb

    rgb = load_rgb(asset)
    if rgb is None or rgb.size == 0 or rgb.shape[0] < 4 or rgb.shape[1] < 4:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    resized = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized)
    low = dct[:LOW_FREQ, :LOW_FREQ]
    # drop the DC term (top-left coefficient): it encodes average
    # brightness, not structure, and would bias the median
    ac = low.flatten()[1:]
    median = float(np.median(ac))
    bits = (ac > median).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return format(value, "016x")


def hamming_distance(hash_a: Optional[str], hash_b: Optional[str]) -> Optional[int]:
    """bit-difference count between two hex pHash strings, or None if
    either is missing or they're different lengths (incomparable)."""
    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return None
    try:
        return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
    except ValueError:
        return None


def visual_similarity(hash_a: Optional[str], hash_b: Optional[str]) -> Optional[float]:
    """1.0 = identical, 0.0 = maximally different (all 63 AC bits flipped),
    None if the hashes can't be compared."""
    dist = hamming_distance(hash_a, hash_b)
    if dist is None:
        return None
    total_bits = (LOW_FREQ * LOW_FREQ) - 1
    return max(0.0, 1.0 - dist / total_bits)
