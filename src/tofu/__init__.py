## 🍢 tofu — context-aware visual text localization
## vieuxtiful
"""ToFU: detect text in an image, erase it, and re-render it in another
language while preserving the scene's visual context.

The pipeline runs in a fixed stage order, each stage a module in
``tofu.layers``::

    tofu      pre-flight validation (glyph coverage, expansion feasibility)
    scene     candidate text-bearing surfaces, then per-instance style
    cicerone  text detection and recognition
    cleanse   source-text erasure and background reconstruction
    scribe    target-language rendering in the detected style
    verify    quality scoring
    memory    visual translation memory

Typical use is the orchestrator, which threads a single asset through all
of them and returns a report alongside the localized image::

    from tofu import TofuPipeline

    result = TofuPipeline().process("sign.png", targ_lang="es")

Individual layers are usable on their own -- they exchange a
:class:`~tofu.core.types.TextManifest` and nothing else -- which is what
makes it practical to swap in a different OCR engine or inpainter without
touching the rest::

    from tofu.layers import cicerone, scribe

    manifest = cicerone.detect("sign.png")
    image = scribe.render(cleansed, manifest, targ_lang="es")

Heavy backends are optional and resolved at call time, never at import:
PaddleOCR and LaMa run out-of-process under their own interpreters, and
scikit-image, uharfbuzz and freetype-py degrade to reduced-quality paths
when absent. Importing this package pulls in none of them.
"""

__version__ = "1.0.0"
__author__ = "vieuxtiful"
__license__ = "MIT"

# Re-exported eagerly: dataclasses and enums only, no heavy imports. Layer
# modules stay lazy (see __getattr__) so `import tofu` never drags in torch.
from tofu.core.types import (
    AssetInfo,
    AssetType,
    BBox,
    BgProfil,
    CharactText,
    GarnishProfile,
    ImageLike,
    InstText,
    LayerMode,
    Mask,
    Polygon,
    PrcStatus,
    QAReport,
    RenderParams,
    SceneRegion,
    ScrptSpprt,
    StyleProfil,
    TextManifest,
    VldtnReport,
    VldtnSeverity,
)

__all__ = [
    "__version__",
    # orchestrator
    "TofuPipeline",
    "PipelineResult",
    # geometry and manifest
    "BBox",
    "Polygon",
    "Mask",
    "InstText",
    "TextManifest",
    "SceneRegion",
    "ImageLike",
    # profiles
    "StyleProfil",
    "BgProfil",
    "CharactText",
    "GarnishProfile",
    "RenderParams",
    # reports and enums
    "QAReport",
    "VldtnReport",
    "VldtnSeverity",
    "ScrptSpprt",
    "AssetInfo",
    "AssetType",
    "LayerMode",
    "PrcStatus",
]

_LAZY = {
    "TofuPipeline": "tofu.core.pipeline",
    "PipelineResult": "tofu.core.pipeline",
}


def __getattr__(name: str):
    """Resolve the orchestrator on first access.

    ``tofu.core.pipeline`` imports every layer, and those layers reach for
    torch and OpenCV. Keeping it behind a module-level ``__getattr__``
    (PEP 562) means ``import tofu`` stays fast enough to use from a CLI or
    a test that only wants the dataclasses, while ``from tofu import
    TofuPipeline`` still works exactly as written.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)


def __dir__():
    return sorted(__all__)
