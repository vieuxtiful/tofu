## pipeline
## vieuxtiful

from typing import Optional
from tofu.core.types import (
    PipelineCfg, PipelineResult, LayerMode,
    TextManifest, VldtnReport
)
from tofu.layers import tofu, cicerone, scene, cleanse, scribe, verify, memory

class TofuPipeline:
    """
    Orchestrates the visual L10n pipeline harmoniously.
    """
    
    def __init__(self, config: Optional[PipelineCfg] = None):
        self.config = config or PipelineCfg()
        self._current_manifest: Optional[TextManifest] = None
        self._current_report: Optional[VldtnReport] = None
        
    def process(self, asset, targ_lang: str) -> PipelineResult:
        """Run the full pipeline end-to-end."""
        
        # Layer 0: ToFU ## pre-flight validation
        validation_report = self._run_tofu(asset, targ_lang)
        """
        asset = img/vid (e.g., file path, URL, in-memory img/vid)
        targ_lang = str ## target language code (e.g., "es", "fr", "de")
        """
        if not validation_report.passed:
            return PipelineResult(
                success=False,
                validation_report=validation_report,
                logs=["ToFU validation failed. See report for details."]
            )
        
        # Layer 1: Cicerone — Text detection & localization
        text_manifest = self._run_cicerone(asset)
        
        # Layer 2: Scene — Semantic context & style analysis
        text_manifest = self._run_scene(asset, text_manifest)
        
        # Layer 3: Cleanse — Text erasure & inpainting
        cleansed_asset = self._run_cleanse(asset, text_manifest)
        
        # Layer 4: Scribe — Style-aware text regeneration
        localized_asset = self._run_scribe(cleansed_asset, text_manifest, targ_lang)
        
        # Layer 5: Verify — Quality verification
        qa_report = self._run_verify(localized_asset, text_manifest)
        
        # Layer 6: Memory — Visual translation memory update
        memory_updates = self._run_memory(text_manifest, targ_lang, localized_asset)
        
        return PipelineResult(
            success=True,
            output_asset=localized_asset,
            text_manifest=text_manifest,
            validation_report=validation_report,
            qa_report=qa_report,
            memory_updates=memory_updates
        )
    
    # --- Layer execution methods with mode checking ---
    
    def _run_tofu(self, asset, targ_lang: str) -> VldtnReport:
        if self.config.tofu_mode == LayerMode.MANUAL:
            # Return a "pending manual review" state
            return VldtnReport(
                passed=True,
                issues=[],
                suggested_actions=["Manual validation required for ToFU layer."]
            )
        return tofu.validate(asset, targ_lang)
    
    def _run_cicerone(self, asset) -> TextManifest:
        if self.config.cicerone_mode == LayerMode.MANUAL:
            # User provides manual annotations
            return self._get_manual_manifest()
        return cicerone.detect(asset)
    
    # ... similarly for other layers