"""Confidence-gated, self-hosted repair-provider routing for Cleanse.

Neural inpainting models are intentionally run outside the ToFU application
environment.  LaMa, BrushNet and future scene-text-removal models have
incompatible Torch/diffusers stacks; an explicit local configuration selects a
dedicated interpreter, checkout and checkpoint for each provider.  The backend
only exchanges temporary image/mask/output files with that local interpreter.
No request field can select a command, endpoint, model, or checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "server" / "inpainting-providers.json"
BRUSHNET_RUNNER = PROJECT_ROOT / "scripts" / "inpaint_brushnet_runner.py"
LAMA_RUNNER = PROJECT_ROOT / "scripts" / "inpaint_lama_runner.py"
NEURAL_PROVIDER_IDS = ("lama", "diffstr_experimental", "brushnet_experimental")
_lama_model: Any = None  # compatibility path for the legacy opt-in package


@dataclass(frozen=True)
class RepairRoute:
    provider: str
    strategy: str
    confidence: float
    auto_accept: bool
    review_required: bool
    reason: str


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    runtime: str
    enabled: bool
    promoted: bool
    available: bool
    reason: str
    config: Dict[str, Any]


@dataclass
class RepairOutcome:
    provider: str
    image: Optional[Any] = None
    elapsed_ms: int = 0
    error: Optional[str] = None
    seed: Optional[int] = None

    def evidence(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "provider": self.provider,
            "elapsed_ms": self.elapsed_ms,
            "ok": self.image is not None and self.error is None,
        }
        if self.seed is not None:
            result["seed"] = self.seed
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class MultiRepairOutcome:
    selected: Optional[RepairOutcome]
    candidates: list[Dict[str, Any]]
    reason: str


def config_path() -> Path:
    configured = os.environ.get("TOFU_INPAINT_CONFIG")
    return Path(configured).expanduser() if configured else DEFAULT_CONFIG_PATH


def _load_config() -> tuple[Dict[str, Any], Optional[str]]:
    path = config_path()
    if not path.exists():
        return {}, f"configuration not found: {path.name}"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"invalid provider configuration ({type(exc).__name__})"
    if not isinstance(parsed, dict) or parsed.get("schema") != 1:
        return {}, "configuration schema must be 1"
    if not isinstance(parsed.get("providers", {}), dict):
        return {}, "configuration.providers must be an object"
    return parsed, None


def _as_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def _python_exists(value: Any) -> bool:
    path = _as_path(value)
    return bool((path and path.is_file()) or (isinstance(value, str) and shutil.which(value)))


def _promoted(name: str) -> bool:
    """Configuration wins; legacy environment remains backwards-compatible."""
    config, _ = _load_config()
    raw = config.get("providers", {}).get(name)
    if isinstance(raw, dict) and "promoted" in raw:
        return raw.get("promoted") is True
    return os.environ.get(f"TOFU_{name.upper()}_PROMOTED", "0") == "1"


def _validate_provider(provider_id: str, raw: Any, config_error: Optional[str]) -> ProviderSpec:
    if not isinstance(raw, dict):
        return ProviderSpec(provider_id, "", False, False, False,
                            config_error or "not configured", {})
    enabled = raw.get("enabled") is True
    runtime = str(raw.get("runtime", ""))
    promoted = raw.get("promoted") is True
    if not enabled:
        return ProviderSpec(provider_id, runtime, False, promoted, False, "disabled in configuration", raw)
    if runtime == "official_lama":
        python = _python_exists(raw.get("python"))
        repo = _as_path(raw.get("repo_dir"))
        model = _as_path(raw.get("model_path"))
        predict = repo / "bin" / "predict.py" if repo else None
        missing = [name for name, ok in (("python", python), ("LaMa checkout", bool(predict and predict.is_file())),
                                         ("LaMa model", bool(model and model.exists()))) if not ok]
    elif runtime == "brushnet":
        python = _python_exists(raw.get("python"))
        repo = _as_path(raw.get("repo_dir"))
        base = _as_path(raw.get("base_model_path"))
        checkpoint = _as_path(raw.get("checkpoint_path"))
        missing = [name for name, ok in (("python", python), ("BrushNet checkout", bool(repo and repo.is_dir())),
                                         ("base model", bool(base and base.exists())),
                                         ("BrushNet checkpoint", bool(checkpoint and checkpoint.exists()))) if not ok]
    elif runtime == "external_command":
        command = raw.get("command")
        required = [_as_path(path) for path in raw.get("required_paths", [])]
        missing = []
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            missing.append("command")
        if any(path is None or not path.exists() for path in required):
            missing.append("required paths")
    elif runtime == "torchscript_lama":
        python = _python_exists(raw.get("python"))
        model = _as_path(raw.get("model_path"))
        missing = [name for name, ok in (
            ("python", python),
            ("LaMa runner", LAMA_RUNNER.is_file()),
            ("LaMa TorchScript model", bool(model and model.is_file())),
        ) if not ok]
    elif runtime == "simple_lama":
        missing = [] if importlib.util.find_spec("simple_lama_inpainting") is not None else ["simple_lama_inpainting package"]
    else:
        missing = ["recognized runtime (official_lama, torchscript_lama, brushnet, external_command, or simple_lama)"]
    if missing:
        return ProviderSpec(provider_id, runtime, True, promoted, False, "missing " + ", ".join(missing), raw)
    return ProviderSpec(provider_id, runtime, True, promoted, True, "ready", raw)


def provider_spec(provider_id: str) -> ProviderSpec:
    config, error = _load_config()
    raw = config.get("providers", {}).get(provider_id)
    spec = _validate_provider(provider_id, raw, error)
    # Preserve the original optional package for existing installations.  New
    # deployments should use the isolated official_lama configuration instead.
    if (provider_id == "lama" and not spec.available
            and os.environ.get("TOFU_LAMA_ENABLED", "0") == "1"
            and importlib.util.find_spec("simple_lama_inpainting") is not None):
        return ProviderSpec("lama", "simple_lama", True, _promoted("lama"), True,
                            "legacy local simple_lama adapter", {})
    return spec


def lama_available() -> bool:
    return provider_spec("lama").available


def provider_statuses() -> list[dict[str, Any]]:
    """Local capability report.  It validates paths but never starts a model."""
    statuses: list[dict[str, Any]] = [
        {"id": "analytic", "available": True, "promoted": True,
         "review_required": False, "kind": "deterministic", "detail": "always available"},
    ]
    for provider_id in NEURAL_PROVIDER_IDS:
        spec = provider_spec(provider_id)
        statuses.append({
            "id": provider_id,
            "available": spec.available,
            "enabled": spec.enabled,
            "promoted": spec.promoted,
            "review_required": not spec.promoted,
            "kind": "self_hosted" if provider_id == "lama" else "experimental",
            "runtime": spec.runtime or None,
            "detail": spec.reason,
            # Config authors can pin a checkpoint/dataset evaluation revision.
            # It becomes part of the Cleanse cache identity without exposing
            # local filesystem paths in the API.
            "revision": str(spec.config.get("revision", "unversioned")),
        })
    statuses.extend([
        {"id": "adobe_fill", "available": False, "promoted": False,
         "review_required": True, "kind": "external_disabled",
         "detail": "disabled: ToFU sends no asset to a cloud provider"},
        {"id": "manual", "available": True, "promoted": True,
         "review_required": False, "kind": "editor", "detail": "non-destructive patch layer"},
    ])
    return statuses


def _complexity_from_ring(image: Any, mask: Any) -> float:
    """Estimate how much local structure demands a diffusion candidate."""
    try:
        import cv2
        import numpy as np
        source = np.asarray(image, dtype=np.uint8)
        region = np.asarray(mask, dtype=bool)
        outer = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=3).astype(bool) & ~region
        if not outer.any():
            return 0.5
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        edge_density = float((cv2.Canny(gray, 70, 150)[outer] > 0).mean())
        chroma = float(np.std(source[outer].reshape(-1, 3), axis=0).mean())
        return max(0.0, min(1.0, edge_density * 3.5 + chroma / 110.0))
    except Exception:
        return 0.5


def _preferred_neural_provider(
    complexity: float,
    material_class: str = "unknown",
    material_confidence: float = 0.0,
) -> Optional[ProviderSpec]:
    # LaMa excels at repeating/global structure.  Diffusion paths are reserved
    # for high-detail scenes; BrushNet is the runnable general inpainting
    # fallback while a DiffSTR model is independently provisioned.
    trusted_material = material_class if material_confidence >= .70 else "unknown"
    if trusted_material in {"glass", "metal", "masonry", "wood"}:
        order = ("lama", "diffstr_experimental", "brushnet_experimental")
    elif trusted_material in {"fabric", "foliage"} or complexity >= 0.48:
        order = ("diffstr_experimental", "brushnet_experimental", "lama")
    else:
        order = ("lama", "diffstr_experimental", "brushnet_experimental")
    for provider_id in order:
        spec = provider_spec(provider_id)
        if spec.available:
            return spec
    return None


def route(inst: Any, surface: Any, image: Any = None, mask: Any = None) -> RepairRoute:
    profile = getattr(inst, "background_profile", None)
    texture = getattr(profile, "texture", None)
    surface_texture = getattr(surface, "texture", None)
    label = getattr(surface, "semantic_label", None)
    if texture == "flat" and surface_texture == "flat" and label == "panel":
        return RepairRoute("analytic", "flat", .99, True, False,
                           "scene and region agree on a flat panel")
    if (texture == "smooth_gradient" and surface_texture == "smooth_gradient"
            and label in {"panel", "bordered_region"}):
        return RepairRoute("analytic", "smooth_gradient", .96, True, False,
                           "scene and region agree on a planar gradient")
    complexity = _complexity_from_ring(image, mask) if image is not None and mask is not None else .5
    reconstruction = getattr(inst, "reconstruction_profile", None)
    material_class = getattr(reconstruction, "material_class", "unknown")
    material_confidence = float(getattr(reconstruction, "material_confidence", 0.0) or 0.0)
    selected = _preferred_neural_provider(complexity, material_class, material_confidence)
    if selected:
        family = "complex" if complexity >= .48 else "repeating"
        confidence = .70 if selected.promoted else .45
        return RepairRoute(selected.provider_id, "neural", confidence,
                           selected.promoted, not selected.promoted,
                           f"{family} texture routed to local {selected.provider_id} "
                           f"(complexity={complexity:.2f}, material={material_class}, "
                           f"material_confidence={material_confidence:.2f})")
    return RepairRoute("telea_fallback", "telea", .30, False, True,
                       "no configured local neural provider; deterministic fallback requires review")


def _run(command: list[str], *, cwd: Optional[Path], timeout: int) -> None:
    result = subprocess.run(
        command, cwd=str(cwd) if cwd else None, shell=False, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=max(1, timeout),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "provider exited without detail").strip().replace("\n", " ")
        raise RuntimeError(detail[:500])


def _read_candidate(path: Path, expected_shape: tuple[int, ...]) -> Any:
    from PIL import Image
    import numpy as np
    if not path.is_file():
        raise RuntimeError("provider produced no output image")
    candidate = np.asarray(Image.open(path).convert("RGB"))
    if candidate.shape != expected_shape:
        raise RuntimeError(f"provider output shape {candidate.shape[:2]} does not match input {expected_shape[:2]}")
    return candidate


def _run_official_lama(spec: ProviderSpec, source: Path, mask: Path, output: Path, work: Path) -> None:
    repo = Path(spec.config["repo_dir"]).expanduser()
    inputs = work / "input"; outputs = work / "output"
    inputs.mkdir(); outputs.mkdir()
    image_name = "tofu.png"; mask_name = "tofu_mask001.png"
    shutil.copy2(source, inputs / image_name)
    shutil.copy2(mask, inputs / mask_name)
    command = [
        str(spec.config["python"]), str(repo / "bin" / "predict.py"),
        f"model.path={Path(spec.config['model_path']).expanduser()}",
        f"indir={inputs}", f"outdir={outputs}",
        f"device={spec.config.get('device', 'cuda')}", "refine=False",
    ]
    _run(command, cwd=repo, timeout=int(spec.config.get("timeout_seconds", 300)))
    images = [p for p in outputs.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not images:
        raise RuntimeError("official LaMa output directory is empty")
    preferred = next((p for p in images if "tofu" in p.stem.lower()), images[0])
    shutil.copy2(preferred, output)


def _run_brushnet(spec: ProviderSpec, source: Path, mask: Path, output: Path) -> Optional[int]:
    if not BRUSHNET_RUNNER.is_file():
        raise RuntimeError("BrushNet runner is missing from ToFU")
    command = [
        str(spec.config["python"]), str(BRUSHNET_RUNNER),
        "--repo-dir", str(Path(spec.config["repo_dir"]).expanduser()),
        "--base-model", str(Path(spec.config["base_model_path"]).expanduser()),
        "--checkpoint", str(Path(spec.config["checkpoint_path"]).expanduser()),
        "--input", str(source), "--mask", str(mask), "--output", str(output),
        "--device", str(spec.config.get("device", "cuda")),
        "--variant", str(spec.config.get("variant", "sd15")),
        "--prompt", str(spec.config.get("prompt", "continuous natural background, no text, no letters, no logo")),
        "--negative-prompt", str(spec.config.get("negative_prompt", "text, letters, words, logo, watermark")),
        "--steps", str(int(spec.config.get("steps", 30))),
        "--conditioning-scale", str(float(spec.config.get("conditioning_scale", 1.0))),
    ]
    seed = int(spec.config.get("seed", 1337))
    command.extend(["--seed", str(seed)])
    _run(command, cwd=Path(spec.config["repo_dir"]).expanduser(), timeout=int(spec.config.get("timeout_seconds", 600)))
    return seed


def _run_torchscript_lama(spec: ProviderSpec, source: Path, mask: Path, output: Path) -> None:
    """Run the modern, isolated TorchScript form of LaMa.

    The original LaMa checkout pins an early Torch version which cannot make
    useful use of Ada GPUs.  This runner consumes the same LaMa big-lama
    TorchScript artifact inside an independently provisioned CUDA runtime.
    It receives no network endpoint and its model path is only configuration,
    never a request parameter.
    """
    command = [
        str(spec.config["python"]), str(LAMA_RUNNER),
        "--model", str(Path(spec.config["model_path"]).expanduser()),
        "--input", str(source), "--mask", str(mask), "--output", str(output),
        "--device", str(spec.config.get("device", "cuda")),
    ]
    expected_hash = spec.config.get("model_sha256")
    if isinstance(expected_hash, str) and expected_hash.strip():
        command.extend(["--expected-sha256", expected_hash.strip()])
    _run(command, cwd=PROJECT_ROOT,
         timeout=int(spec.config.get("timeout_seconds", 300)))


def _run_external_command(spec: ProviderSpec, source: Path, mask: Path, output: Path, request: Path) -> None:
    fields = {"input": str(source), "mask": str(mask), "output": str(output), "request": str(request)}
    try:
        command = [part.format(**fields) for part in spec.config["command"]]
    except KeyError as exc:
        raise RuntimeError(f"unknown command placeholder: {exc.args[0]}") from exc
    cwd = _as_path(spec.config.get("cwd"))
    _run(command, cwd=cwd, timeout=int(spec.config.get("timeout_seconds", 600)))


def repair(provider_id: str, image: Any, mask: Any) -> RepairOutcome:
    """Run one configured local provider through the stable file contract."""
    started = time.monotonic()
    spec = provider_spec(provider_id)
    if not spec.available:
        return RepairOutcome(provider_id, error=spec.reason)
    try:
        import numpy as np
        from PIL import Image
        source_arr = np.asarray(image, dtype=np.uint8)
        mask_arr = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
        max_pixels = int(spec.config.get("max_pixels", 4_194_304))
        if source_arr.shape[0] * source_arr.shape[1] > max_pixels:
            raise RuntimeError(f"input exceeds configured max_pixels ({max_pixels})")
        with tempfile.TemporaryDirectory(prefix="tofu-repair-") as temp:
            work = Path(temp); source = work / "input.png"; mask_path = work / "mask.png"; output = work / "output.png"
            Image.fromarray(source_arr, "RGB").save(source)
            Image.fromarray(mask_arr, "L").save(mask_path)
            request = work / "request.json"
            request.write_text(json.dumps({"schema": 1, "provider": provider_id, "input": str(source), "mask": str(mask_path), "output": str(output)}), encoding="utf-8")
            seed = None
            if spec.runtime == "official_lama":
                _run_official_lama(spec, source, mask_path, output, work)
            elif spec.runtime == "torchscript_lama":
                _run_torchscript_lama(spec, source, mask_path, output)
            elif spec.runtime == "brushnet":
                seed = _run_brushnet(spec, source, mask_path, output)
            elif spec.runtime == "external_command":
                _run_external_command(spec, source, mask_path, output, request)
            elif spec.runtime == "simple_lama":
                global _lama_model
                from simple_lama_inpainting import SimpleLama
                if _lama_model is None:
                    _lama_model = SimpleLama()
                generated = _lama_model(Image.open(source), Image.open(mask_path))
                generated.convert("RGB").save(output)
            else:  # guarded by _validate_provider, retained for defensive callers
                raise RuntimeError(f"unsupported runtime: {spec.runtime}")
            candidate = _read_candidate(output, source_arr.shape)
        return RepairOutcome(provider_id, candidate, int((time.monotonic() - started) * 1000), seed=seed)
    except Exception as exc:
        return RepairOutcome(provider_id, None, int((time.monotonic() - started) * 1000), str(exc)[:600])


def repair_lama(image: Any, mask: Any) -> Optional[Any]:
    """Backward-compatible convenience wrapper for direct LaMa callers."""
    return repair("lama", image, mask).image


def _structural_continuity_score(cv2, np, before_gray, after_gray, region) -> Optional[float]:
    """Measure support for source lines that enter and leave the erase mask.

    Absence of qualifying structure is unknown (None), not perfect evidence.
    """
    source_edges = cv2.Canny(before_gray, 70, 150)
    candidate_edges = cv2.Canny(after_gray, 70, 150)
    lines = cv2.HoughLinesP(
        source_edges, 1, np.pi / 180, threshold=18,
        minLineLength=12, maxLineGap=4,
    )
    if lines is None:
        return None
    support: list[float] = []
    h, w = region.shape
    proximity_kernel = np.ones((3, 3), np.uint8)
    candidate_near = cv2.dilate(
        (candidate_edges > 0).astype(np.uint8), proximity_kernel, iterations=1
    ).astype(bool)
    for raw in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in raw)
        count = max(abs(x2 - x1), abs(y2 - y1)) + 1
        if count < 2:
            continue
        xs = np.clip(np.rint(np.linspace(x1, x2, count)).astype(int), 0, w - 1)
        ys = np.clip(np.rint(np.linspace(y1, y2, count)).astype(int), 0, h - 1)
        on_mask = region[ys, xs]
        # A useful structural observation crosses both sides of the mask and
        # has enough samples within it to distinguish continuity from noise.
        if on_mask.sum() < 3 or (~on_mask).sum() < 4:
            continue
        support.append(float(candidate_near[ys[on_mask], xs[on_mask]].mean()))
    if not support:
        return None
    return float(sum(support) / len(support))


def quality_gate(image: Any, candidate: Any, mask: Any) -> tuple[bool, dict[str, Any]]:
    """Hard-gate, then rank a repair using provider-independent evidence."""
    try:
        import cv2
        import numpy as np
        before = np.asarray(image, dtype=np.float32)
        after = np.asarray(candidate, dtype=np.float32)
        region = np.asarray(mask, dtype=bool)
        hard_rejections: list[str] = []
        if before.shape != after.shape or region.shape != before.shape[:2] or not region.any():
            return False, {"passed": False, "score": 0.0, "reason": "invalid candidate geometry",
                           "hard_rejections": ["invalid_geometry"]}
        if not np.isfinite(after).all():
            return False, {"passed": False, "score": 0.0, "reason": "invalid candidate pixels",
                           "hard_rejections": ["non_finite_pixels"]}
        shell = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & ~region
        inner = region & ~cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        if not shell.any() or not inner.any():
            return False, {"passed": False, "score": 0.0, "reason": "insufficient known context"}
        # Providers may touch a narrow blend band, but never arbitrary known
        # context. Excluding the two-pixel shell avoids rejecting benign
        # antialiasing while preserving a strict protected-area contract.
        protected = ~(region | shell)
        outside_delta = float(np.abs(after[protected] - before[protected]).mean()) if protected.any() else 0.0
        outer = before[shell].reshape(-1, 3)
        edge = after[inner].reshape(-1, 3)
        color_gap = float(np.linalg.norm(np.median(edge, axis=0) - np.median(outer, axis=0)))
        texture_scale = float(np.linalg.norm(np.std(outer, axis=0)))
        tolerance = max(18.0, texture_scale * 1.8)
        seam_score = float(np.exp(-max(0.0, color_gap - tolerance) / max(18.0, tolerance)))
        outside_score = float(np.exp(-outside_delta / .25))

        # A seam can look smooth while the original high-contrast letter
        # edges remain inside the mask.  This is deliberately only a weak
        # conservative signal: natural structure may legitimately continue
        # through the erased area, so ambiguity routes to review rather than
        # claiming semantic correctness from an unobservable background.
        before_edges = cv2.Canny(cv2.cvtColor(before.astype(np.uint8), cv2.COLOR_RGB2GRAY), 70, 150)
        after_edges = cv2.Canny(cv2.cvtColor(after.astype(np.uint8), cv2.COLOR_RGB2GRAY), 70, 150)
        before_density = float((before_edges[inner] > 0).mean())
        after_density = float((after_edges[inner] > 0).mean())
        edge_persistence = after_density / max(.01, before_density)
        residual_score = float(np.exp(-max(0.0, edge_persistence - .72) / .45))
        gray_before = cv2.cvtColor(before.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(after.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        outer_grad = cv2.Laplacian(gray_before, cv2.CV_32F)[shell]
        inner_grad = cv2.Laplacian(gray_after, cv2.CV_32F)[inner]
        grad_gap = abs(float(np.std(inner_grad)) - float(np.std(outer_grad)))
        texture_score = float(np.exp(-grad_gap / max(12.0, float(np.std(outer_grad)))))
        structural_score = _structural_continuity_score(
            cv2, np, gray_before, gray_after, region
        )
        structural_term = structural_score if structural_score is not None else .5
        score = round(
            0.30 * seam_score + 0.25 * outside_score
            + 0.18 * texture_score + 0.15 * residual_score
            + 0.12 * structural_term,
            4,
        )
        if outside_delta > .25:
            hard_rejections.append("protected_area_changed")
        if seam_score < .35:
            hard_rejections.append("severe_boundary_seam")
        if residual_score < .45:
            hard_rejections.append("glyph_edge_persistence")
        if structural_score is not None and structural_score < .30:
            hard_rejections.append("structural_break")
        passed = not hard_rejections and score >= 0.82
        return passed, {"passed": passed, "score": score, "outside_delta": round(outside_delta, 4),
                        "seam_score": round(seam_score, 4),
                        "texture_score": round(texture_score, 4),
                        "structural_continuity_score": (
                            None if structural_score is None else round(structural_score, 4)
                        ),
                        "residual_edge_score": round(residual_score, 4),
                        "edge_persistence": round(edge_persistence, 4),
                        "hard_rejections": hard_rejections,
                        "reason": "candidate preserves known context" if passed else "candidate lacks high-confidence boundary or residual-text evidence"}
    except Exception:
        return False, {"passed": False, "score": 0.0, "reason": "candidate quality evaluation failed",
                       "hard_rejections": ["evaluation_error"]}


def repair_multi_candidate(
    image: Any,
    mask: Any,
    *,
    max_candidates: int = 3,
    include_unpromoted: bool = True,
    on_candidate: Optional[Callable[[Dict[str, Any], Any], None]] = None,
    candidate_transform: Optional[Callable[[Any], Any]] = None,
    quality_image: Any = None,
    quality_mask: Any = None,
) -> MultiRepairOutcome:
    """Run available neural providers against the same source and rank survivors."""
    specs_by_id: Dict[str, ProviderSpec] = {}
    for provider_id in NEURAL_PROVIDER_IDS:
        spec = provider_spec(provider_id)
        if spec.available and (include_unpromoted or spec.promoted):
            specs_by_id.setdefault(spec.provider_id, spec)
    specs = list(specs_by_id.values())
    specs.sort(key=lambda spec: (
        not spec.promoted,
        0 if spec.provider_id == "diffstr_experimental" else
        1 if spec.provider_id == "lama" else 2,
        spec.provider_id,
    ))
    evidence: list[Dict[str, Any]] = []
    eligible: list[tuple[float, int, RepairOutcome]] = []
    for index, spec in enumerate(specs[:max(0, max_candidates)]):
        outcome = repair(spec.provider_id, image, mask)
        rectified_passed, rectified_quality = (
            quality_gate(image, outcome.image, mask)
            if outcome.image is not None else
            (False, {"passed": False, "score": 0.0, "hard_rejections": ["provider_error"],
                     "reason": outcome.error or "provider returned no candidate"})
        )
        candidate = outcome.image
        transformed_error = None
        if candidate is not None and candidate_transform is not None:
            try:
                candidate = candidate_transform(candidate)
            except Exception as exc:
                transformed_error = f"{type(exc).__name__}: {exc}"[:300]
                candidate = None
        if quality_image is not None and quality_mask is not None:
            original_passed, original_quality = (
                quality_gate(quality_image, candidate, quality_mask)
                if candidate is not None else
                (False, {"passed": False, "score": 0.0,
                         "hard_rejections": ["inverse_warp_error"],
                         "reason": transformed_error or "candidate transform failed"})
            )
            passed = bool(rectified_passed and original_passed)
            quality = {
                "passed": passed,
                "score": min(
                    float(rectified_quality.get("score", 0.0)),
                    float(original_quality.get("score", 0.0)),
                ),
                "hard_rejections": list(dict.fromkeys(
                    list(rectified_quality.get("hard_rejections", []))
                    + list(original_quality.get("hard_rejections", []))
                )),
                "rectified": rectified_quality,
                "original_space": original_quality,
            }
        else:
            passed, quality = rectified_passed, rectified_quality
        selected_outcome = RepairOutcome(
            outcome.provider, candidate, outcome.elapsed_ms,
            outcome.error or transformed_error, outcome.seed,
        )
        auto_eligible = bool(passed and spec.promoted)
        item = {
            "candidate_id": f"{spec.provider_id}:{index}",
            "provider": spec.provider_id,
            "provider_revision": str(spec.config.get("revision", "unversioned")),
            "promoted": spec.promoted,
            "execution": outcome.evidence(),
            "quality_gate": quality,
            "eligible": auto_eligible,
            "selected": False,
        }
        evidence.append(item)
        if on_candidate is not None and candidate is not None:
            on_candidate(item, candidate)
        if auto_eligible:
            eligible.append((float(quality.get("score", 0.0)), -index, selected_outcome))
    if not eligible:
        return MultiRepairOutcome(None, evidence, "no promoted candidate passed hard gates")
    _, _, selected = max(eligible, key=lambda item: (item[0], item[1]))
    for item in evidence:
        if item["provider"] == selected.provider and item["eligible"]:
            item["selected"] = True
            break
    return MultiRepairOutcome(selected, evidence, "highest-ranked eligible candidate")
