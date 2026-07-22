"""One-shot, isolated TorchScript LaMa runner for ToFU Cleanse.

The server passes an already-local RGB image and single-channel text mask.
This module never downloads a model and never opens a network connection: a
configured, local TorchScript weight is mandatory.  The output preserves the
source dimensions; Cleanse performs its own alpha composition and quality
gate after this runner exits.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-sha256", default="")
    return parser.parse_args()


def main() -> None:
    args = _args()
    model = Path(args.model).expanduser().resolve()
    source = Path(args.input).resolve()
    mask = Path(args.mask).resolve()
    output = Path(args.output).resolve()
    for label, path in (("LaMa TorchScript model", model), ("input", source), ("mask", mask)):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.expected_sha256:
        digest = hashlib.sha256()
        with model.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().casefold() != args.expected_sha256.casefold():
            raise SystemExit("LaMa TorchScript model SHA-256 does not match configured revision")

    # The underlying package deliberately downloads only when LAMA_MODEL is
    # absent.  Supplying this absolute path makes a network request impossible
    # during an asset-cleaning operation.
    os.environ["LAMA_MODEL"] = str(model)
    import torch
    from PIL import Image
    from simple_lama_inpainting import SimpleLama

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("LaMa is configured for CUDA but this environment has no CUDA device")
    device = torch.device(args.device if args.device.startswith("cuda") else "cpu")
    original = Image.open(source).convert("RGB")
    mask_image = Image.open(mask).convert("L")
    if original.size != mask_image.size:
        raise SystemExit("input and mask dimensions must match")
    result = SimpleLama(device=device)(original, mask_image).convert("RGB")
    # LaMa pads its tensors to a multiple of eight.  The padding is appended
    # at the right/bottom by its input helper, so crop it back before handing
    # the candidate to Cleanse.  A result smaller than the source is never a
    # valid repair and remains a hard error.
    if result.width < original.width or result.height < original.height:
        raise SystemExit(f"LaMa shrank image dimensions from {original.size} to {result.size}")
    if result.size != original.size:
        result = result.crop((0, 0, original.width, original.height))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


if __name__ == "__main__":
    main()
