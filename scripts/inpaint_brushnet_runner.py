"""One-shot local BrushNet runner used by ToFU's isolated provider bridge.

Run this script with the Python environment created from TencentARC/BrushNet,
not with ToFU's application Python.  All `from_pretrained` calls are local-only
so this script can never upload an asset or silently fetch a model during a
localized render.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=("sd15", "sdxl"), default="sd15")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_dir).expanduser().resolve()
    base_model = Path(args.base_model).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for label, path in (("BrushNet checkout", repo), ("base model", base_model), ("checkpoint", checkpoint)):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    # The official repo ships the BrushNet-aware diffusers fork under src/.
    # It must win over a regular application-environment diffusers install.
    sys.path.insert(0, str(repo / "src"))

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import BrushNetModel, UniPCMultistepScheduler

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("BrushNet configured for CUDA but its isolated environment has no CUDA device")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if args.variant == "sdxl":
        from diffusers import StableDiffusionXLBrushNetPipeline
        pipeline_class = StableDiffusionXLBrushNetPipeline
    else:
        from diffusers import StableDiffusionBrushNetPipeline
        pipeline_class = StableDiffusionBrushNetPipeline

    brushnet = BrushNetModel.from_pretrained(str(checkpoint), torch_dtype=dtype, local_files_only=True)
    pipe = pipeline_class.from_pretrained(
        str(base_model), brushnet=brushnet, torch_dtype=dtype,
        low_cpu_mem_usage=False, local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device.startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    original = Image.open(args.input).convert("RGB")
    mask_l = Image.open(args.mask).convert("L")
    if original.size != mask_l.size:
        raise SystemExit("input and mask dimensions must match")
    source = np.asarray(original, dtype=np.uint8)
    mask = np.asarray(mask_l, dtype=np.uint8) > 0
    # The official demo feeds a masked image to the dual branch.  Keep a copy
    # of known pixels for exact post-composition in addition to ToFU's final
    # feathered blend and quality gate.
    masked = source.copy()
    masked[mask] = 0
    init_image = Image.fromarray(masked, "RGB")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), "L").convert("RGB")
    generator = torch.Generator("cuda" if device.startswith("cuda") else "cpu").manual_seed(args.seed)
    options = {
        "num_inference_steps": max(1, args.steps),
        "generator": generator,
        "brushnet_conditioning_scale": args.conditioning_scale,
    }
    if args.negative_prompt:
        options["negative_prompt"] = args.negative_prompt
    try:
        generated = pipe(args.prompt, init_image, mask_image, **options).images[0]
    except TypeError:
        # Older official forks did not expose negative_prompt on this pipeline.
        options.pop("negative_prompt", None)
        generated = pipe(args.prompt, init_image, mask_image, **options).images[0]
    result = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    if result.shape != source.shape:
        raise SystemExit(f"BrushNet changed image dimensions from {source.shape[:2]} to {result.shape[:2]}")
    result[~mask] = source[~mask]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, "RGB").save(args.output)


if __name__ == "__main__":
    main()
