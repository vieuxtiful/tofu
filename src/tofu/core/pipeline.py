## pipeline
## vieuxtiful

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from tofu.core.types import (
    PipelineCfg, PipelineResult, LayerMode, PrcStatus,
    TextManifest, VldtnReport, QAReport, VerificationReport, RenderParams, StyleProfil,
    SceneRegion, AssetType, AssetInfo, infer_asset_info
)
from tofu.layers import tofu, cicerone, scene, cleanse, scribe, garnish, verify, memory
from tofu.core.events import PipelineEvent, PipelineEventStatus, PipelineObserver

# HYBRID pause checkpoints (pause protocol, not a third mode branch):
# the pipeline yields control at these points so the user can refine
# intermediate results (auto → refine → auto → verify sign-off).
CHECKPOINT_MASKS = "post_cicerone_masks"
CHECKPOINT_SIGNOFF = "pre_verify_signoff"

# an on_checkpoint callback receives (checkpoint_name, payload) and returns
# the (possibly refined) payload, or None to pause the run at that point.
CheckpointFn = Callable[[str, Any], Optional[Any]]

class TofuPipeline:
    """
    Orchestrates the visual L10n pipeline harmoniously.

    Stage order (v2): tofu pre-flight → scene PRE-PASS (candidate
    surfaces constrain detection) → cicerone → tofu re-validation →
    scene enrichment → cleanse → scribe → verify → memory.

    Every run collects structured logs ({ts, stage, level, message,
    duration_ms?}) and errors on the PipelineResult — this is the single
    source of render observability for both the API and CLI paths.
    """

    def __init__(
        self,
        config: Optional[PipelineCfg] = None,
        on_checkpoint: Optional[CheckpointFn] = None,
        font_registry: Optional[Any] = None,
        on_event: Optional[PipelineObserver] = None,
    ):
        self.config = config or PipelineCfg()
        self.on_checkpoint = on_checkpoint
        self.on_event = on_event
        # threaded through to scribe.render: resolves requested weight/
        # italic to a real sibling font face when one exists, instead of
        # always synthesizing bold/italic from the base face
        self.font_registry = font_registry
        # ...and through to ToFU, which is the layer that actually needs
        # to know whether a font can DRAW the target language. the stage
        # used to call the module-level singleton, which is built with no
        # font library at all -- so every pipeline run silently validated
        # against the static script map while the caller was already
        # holding a real registry.
        self._validator = tofu.ToFU(font_registry=font_registry)
        self._current_manifest: Optional[TextManifest] = None
        self._current_report: Optional[VldtnReport] = None
        self._manual_manifest: Optional[TextManifest] = None
        self._logs: List[Dict[str, Any]] = []
        self._errors: List[str] = []
        self._observer_errors: List[str] = []

    def set_manual_manifest(self, manifest: TextManifest) -> None:
        """provide user-authored annotations for MANUAL cicerone mode."""
        self._manual_manifest = manifest

    # --- logging -------------------------------------------------------------

    def _emit(
        self,
        stage: str,
        operation: str,
        status: PipelineEventStatus,
        *,
        progress: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Notify a caller without allowing presentation code to break a run."""
        if self.on_event is None:
            return
        event = PipelineEvent(
            stage=stage,
            operation=operation,
            status=status,
            progress=progress,
            payload=payload or {},
            duration_ms=duration_ms,
        )
        try:
            self.on_event(event)
        except Exception as exc:
            # Observers are adapters, not pipeline stages.  Keep this separate
            # from result.errors so a disconnected UI cannot fail useful work.
            self._observer_errors.append(
                f"{type(exc).__name__}: {exc}"
            )

    def _begin(
        self, stage: str, operation: str, progress: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> float:
        self._emit(
            stage, operation, PipelineEventStatus.STARTED,
            progress=progress, payload=payload,
        )
        return time.time()

    def _finish(
        self, stage: str, operation: str, t0: float, progress: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            stage, operation, PipelineEventStatus.COMPLETED,
            progress=progress, payload=payload,
            duration_ms=int((time.time() - t0) * 1000),
        )

    def _log(self, stage: str, message: str, level: str = "info",
             t0: Optional[float] = None) -> None:
        entry: Dict[str, Any] = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "stage": stage, "level": level, "message": message,
        }
        if t0 is not None:
            entry["duration_ms"] = int((time.time() - t0) * 1000)
        self._logs.append(entry)

    def _fail(self, stage: str, exc: Exception) -> None:
        msg = f"{stage} failed: {type(exc).__name__}: {exc}"
        self._errors.append(msg)
        self._log(stage, msg, "error")
        self._emit(
            stage, "run", PipelineEventStatus.FAILED,
            payload={"error_type": type(exc).__name__, "message": str(exc)},
        )

    def _report_errors(self, report: VldtnReport) -> None:
        """surface validation ERROR issues in the run's errors list."""
        for issue in report.issues:
            if issue.severity.value == "error":
                self._errors.append(f"tofu {issue.code}: {issue.message}")

    def _result(self, **kw) -> PipelineResult:
        kw.setdefault("logs", self._logs)
        kw.setdefault("errors", self._errors)
        return PipelineResult(**kw)

    # --- entry point ---------------------------------------------------------

    def process(
        self,
        asset,
        targ_lang: str,
        asset_info: Optional[AssetInfo] = None,
        font: Optional[str] = None,
        render_params: Optional[Dict[str, RenderParams]] = None,
    ) -> PipelineResult:
        """Run the full pipeline end-to-end.

        Routing: asset_info (or inference from the asset itself) determines
        static vs. video processing. Images default to frames=1 and run the
        static pipeline; video is deferred until temporal layers land.

        font: optional font path applied per region, merged with each
        instance's scene-enriched style (color/weight survive the
        override). render_params: explicit per-region overrides; takes
        precedence over the font merge when provided.
        """
        self._logs = []
        self._errors = []
        self._observer_errors = []
        asset_info = asset_info or infer_asset_info(asset)
        self._emit(
            "pipeline", "process", PipelineEventStatus.STARTED, progress=0.0,
            payload={
                "asset_type": asset_info.asset_type.value,
                "target_language": targ_lang,
            },
        )

        if asset_info.asset_type == AssetType.VIDEO:
            self._log("pipeline", "video pipeline not yet implemented", "error")
            self._errors.append("video pipeline not yet implemented")
            self._emit(
                "pipeline", "process", PipelineEventStatus.FAILED, progress=1.0,
                payload={"reason": "video pipeline not yet implemented"},
            )
            return self._result(success=False, status=PrcStatus.FAILED)

        return self._process_static(asset, targ_lang, asset_info, font, render_params)

    def _process_static(
        self,
        asset,
        targ_lang: str,
        asset_info: AssetInfo,
        font: Optional[str] = None,
        render_params: Optional[Dict[str, RenderParams]] = None,
    ) -> PipelineResult:
        """Static (frames=1) image pipeline."""

        # Layer 0: ToFU ## pre-flight validation
        t0 = self._begin("tofu", "preflight", 0.02)
        validation_report = self._run_tofu(asset, targ_lang, font=font)
        self._log("tofu", f"pre-flight for '{targ_lang}': "
                          f"{'passed' if validation_report.passed else 'failed'}", t0=t0)
        self._finish(
            "tofu", "preflight", t0, 0.08,
            {"passed": validation_report.passed},
        )
        if not validation_report.passed:
            self._report_errors(validation_report)
            self._emit(
                "pipeline", "process", PipelineEventStatus.FAILED,
                progress=0.08, payload={"reason": "preflight validation failed"},
            )
            return self._result(
                success=False,
                status=PrcStatus.FAILED,
                validation_report=validation_report,
            )

        # Scene pre-pass: candidate surfaces constrain AUTO detection.
        # skipped for MANUAL cicerone (user-authored regions need no filter).
        scene_regions: List[SceneRegion] = []
        if self.config.cicerone_mode != LayerMode.MANUAL:
            t0 = self._begin("scene", "prepass", 0.10)
            try:
                scene_regions = scene.analyze_regions(asset)
                self._log("scene", f"pre-pass: {len(scene_regions)} candidate surface(s)", t0=t0)
                self._finish(
                    "scene", "prepass", t0, 0.18,
                    {"region_count": len(scene_regions)},
                )
            except Exception as exc:
                self._log("scene", f"pre-pass failed ({type(exc).__name__}); "
                                   "detection unconstrained", "warning", t0=t0)
                scene_regions = []
                self._emit(
                    "scene", "prepass", PipelineEventStatus.WARNING, progress=0.18,
                    payload={"error_type": type(exc).__name__, "message": str(exc)},
                    duration_ms=int((time.time() - t0) * 1000),
                )

        # Layer 1: Cicerone — Text detection & localization
        t0 = self._begin("cicerone", "detect", 0.20)
        text_manifest = self._run_cicerone(asset, asset_info, scene_regions)
        text_manifest.asset_type = asset_info.asset_type
        text_manifest.frame_count = asset_info.frame_count
        text_manifest.targ_lang = targ_lang
        if scene_regions and not text_manifest.scene_regions:
            text_manifest.scene_regions = scene_regions
        self._log("cicerone", f"{text_manifest.total_regions} text region(s)", t0=t0)
        self._finish(
            "cicerone", "detect", t0, 0.35,
            {"region_count": text_manifest.total_regions},
        )

        # HYBRID checkpoint 1: refine detected masks/regions
        refined = self._checkpoint(CHECKPOINT_MASKS, text_manifest)
        if refined is None:
            self._log("pipeline", f"paused at checkpoint '{CHECKPOINT_MASKS}'", "warning")
            self._emit(
                "pipeline", CHECKPOINT_MASKS, PipelineEventStatus.PAUSED,
                progress=0.36,
            )
            return self._result(
                success=False,
                status=PrcStatus.AWAITING_REFINEMENT,
                text_manifest=text_manifest,
                validation_report=validation_report,
            )
        text_manifest = refined

        # ToFU re-validation with real regions (expansion feasibility, ToFU_005)
        t0 = self._begin("tofu", "revalidate", 0.38)
        validation_report = self._run_tofu(asset, targ_lang, text_manifest, font=font)
        self._log("tofu", f"re-validation with {text_manifest.total_regions} region(s): "
                          f"{'passed' if validation_report.passed else 'failed'}", t0=t0)
        self._finish(
            "tofu", "revalidate", t0, 0.42,
            {"passed": validation_report.passed},
        )
        if not validation_report.passed:
            self._report_errors(validation_report)
            self._emit(
                "pipeline", "process", PipelineEventStatus.FAILED,
                progress=0.42, payload={"reason": "region validation failed"},
            )
            return self._result(
                success=False,
                status=PrcStatus.FAILED,
                text_manifest=text_manifest,
                validation_report=validation_report,
            )

        total = len(text_manifest.instances)
        dnt = sum(1 for i in text_manifest.instances if i.dnt)
        untranslated = [
            i.id for i in text_manifest.instances if not i.dnt and not i.target_text
        ]
        self._log("plan", f"{total} region(s): {total - dnt - len(untranslated)} to render, "
                          f"{dnt} DNT, {len(untranslated)} untranslated"
                          + (f" (font: {font})" if font else ""))
        if untranslated:
            self._log("plan", "untranslated regions will be cleansed but left empty: "
                              + ", ".join(untranslated), "warning")

        # Layer 2: Scene — enrichment; degrades output but is not fatal
        t0 = self._begin("scene", "enrich", 0.44)
        try:
            text_manifest = self._run_scene(asset, text_manifest)
            self._log("scene", f"{len(text_manifest.scene_regions)} surface region(s); "
                               f"profiles enriched for {total} text region(s)", t0=t0)
            self._finish(
                "scene", "enrich", t0, 0.52,
                {"region_count": len(text_manifest.scene_regions)},
            )
        except Exception as exc:
            self._fail("scene", exc)
            self._log("scene", "continuing with unenriched profiles", "warning")

        # per-region overrides: requested font merged with the enriched
        # style, built AFTER scene so the detected text color survives.
        # a per-region font (style_profile.font_family, user-selected in
        # the preview table) always beats the request-level font.
        if font and not render_params:
            render_params = {}
            for inst in text_manifest.instances:
                base = inst.style_profile
                render_params[inst.id] = RenderParams(
                    position=inst.bounding_box,
                    style=StyleProfil(
                        font_family=(base.font_family if base and base.font_family else font),
                        font_weight=base.font_weight if base else None,
                        color=base.color if base else None,
                        shadow=base.shadow if base else None,
                        effects=base.effects if base else None,
                    ),
                )

        # Layer 3: Cleanse — text erasure; fatal (scribe over un-erased text)
        t0 = self._begin("cleanse", "erase", 0.54)
        try:
            cleansed_asset = self._run_cleanse(asset, text_manifest)
            self._log("cleanse", f"erased {total - dnt} region(s)", t0=t0)
            self._finish(
                "cleanse", "erase", t0, 0.66,
                {"region_count": total - dnt},
            )
        except Exception as exc:
            self._fail("cleanse", exc)
            self._emit(
                "pipeline", "process", PipelineEventStatus.FAILED,
                progress=0.66, payload={"reason": "cleanse failed"},
            )
            return self._result(
                success=False,
                status=PrcStatus.FAILED,
                text_manifest=text_manifest,
                validation_report=validation_report,
            )

        # Layer 4: Scribe — target-text rendering; fatal
        t0 = self._begin("scribe", "render", 0.68)
        try:
            localized_asset = self._run_scribe(
                cleansed_asset, text_manifest, targ_lang, render_params
            )
            self._log("scribe", f"rendered {total - dnt - len(untranslated)} region(s) "
                                f"for '{targ_lang}'", t0=t0)
            self._finish(
                "scribe", "render", t0, 0.80,
                {"region_count": total - dnt - len(untranslated)},
            )
            # ToFU render-time guard: scribe swapped fonts for regions
            # whose requested face couldn't draw the target text at all —
            # the exact "tofu" (missing-glyph) failure the layer is
            # named for, now caught instead of silently reaching the user
            fallback_ids = [i.id for i in text_manifest.instances if i.glyph_fallback]
            if fallback_ids:
                self._log(
                    "tofu",
                    f"glyph fallback applied for region(s) {', '.join(fallback_ids)}: "
                    "the requested font lacked codepoints for the target text; "
                    "scribe swapped to the best-covering font it found",
                    "warning",
                )
        except Exception as exc:
            self._fail("scribe", exc)
            self._emit(
                "pipeline", "process", PipelineEventStatus.FAILED,
                progress=0.80, payload={"reason": "scribe failed"},
            )
            return self._result(
                success=False,
                status=PrcStatus.FAILED,
                text_manifest=text_manifest,
                validation_report=validation_report,
            )

        # Capture the canonical inventory from Scribe's clean render. Garnish
        # is downstream so presentation effects cannot contaminate the
        # localization evidence or its eventual component scores.
        verification_report: Optional[VerificationReport] = None
        try:
            verification_report = verify.build_verification_report(
                localized_asset, text_manifest, self.font_registry
            )
            self._log(
                "verify",
                "captured clean-render inventory before Garnish "
                f"({len(verification_report.regions)} region(s))",
            )
        except Exception as exc:
            self._fail("verify inventory", exc)

        # Layer 4b: Garnish — deterministic source-wear treatment, non-fatal.
        try:
            localized_asset = garnish.apply(localized_asset, text_manifest, cleansed_asset, self.font_registry)
            self._log("garnish", "applied source-derived text-edge treatment")
        except Exception as exc:
            self._fail("garnish", exc)

        # HYBRID checkpoint 2: sign-off on the rendered result before QA
        signed_off = self._checkpoint(CHECKPOINT_SIGNOFF, localized_asset)
        if signed_off is None:
            self._log("pipeline", f"paused at checkpoint '{CHECKPOINT_SIGNOFF}'", "warning")
            self._emit(
                "pipeline", CHECKPOINT_SIGNOFF, PipelineEventStatus.PAUSED,
                progress=0.84,
            )
            return self._result(
                success=False,
                status=PrcStatus.AWAITING_REFINEMENT,
                output_asset=localized_asset,
                text_manifest=text_manifest,
                validation_report=validation_report,
                verification_report=verification_report,
            )
        localized_asset = signed_off

        # Layer 5: Verify — QA scoring; non-fatal (output usable unscored)
        t0 = self._begin("verify", "assess", 0.86)
        qa_report: Optional[QAReport] = None
        try:
            qa_report = self._run_verify(localized_asset, text_manifest, asset, cleansed_asset)
            score = qa_report.overall_score
            self._log("verify", "overall QA "
                      + (f"{score:.2f}" if score is not None else "n/a"), t0=t0)
            self._finish(
                "verify", "assess", t0, 0.94,
                {"overall_score": score},
            )
        except Exception as exc:
            self._fail("verify", exc)

        qa_passed = (
            qa_report is not None
            and qa_report.overall_score is not None
            and qa_report.overall_score >= self.config.qa_threshold
        )
        if qa_report is not None and qa_report.overall_score is not None:
            self._log("verify", f"QA gate: {qa_report.overall_score:.2f} vs threshold "
                                f"{self.config.qa_threshold:.2f} → "
                                f"{'pass' if qa_passed else 'fail'}",
                      "info" if qa_passed else "warning")

        # Layer 6: Memory — only stores QA-approved results
        memory_updates: List[Dict[str, Any]] = []
        if qa_passed:
            t0 = self._begin("memory", "update", 0.95)
            try:
                memory_updates = self._run_memory(
                    text_manifest, targ_lang, localized_asset, qa_report, asset
                )
                self._log("memory", f"{len(memory_updates)} TM record(s) stored", t0=t0)
                self._finish(
                    "memory", "update", t0, 0.99,
                    {"update_count": len(memory_updates)},
                )
            except Exception as exc:
                self._fail("memory", exc)
        else:
            self._log("memory", "QA below threshold; memory update skipped", "warning")

        self._emit(
            "pipeline", "process",
            PipelineEventStatus.COMPLETED if qa_passed else PipelineEventStatus.FAILED,
            progress=1.0,
            payload={"qa_passed": qa_passed},
        )
        return self._result(
            success=qa_passed,
            status=PrcStatus.COMPLETED if qa_passed else PrcStatus.AUTO_REJECTED,
            output_asset=localized_asset,
            text_manifest=text_manifest,
            validation_report=validation_report,
            verification_report=verification_report,
            qa_report=qa_report,
            memory_updates=memory_updates,
        )

    def _checkpoint(self, name: str, payload: Any) -> Optional[Any]:
        """HYBRID pause protocol: when any layer is in HYBRID mode and a
        checkpoint callback is registered, yield the payload for refinement.
        returns the refined payload, or None to pause the run.
        without a callback (or outside HYBRID), the payload passes through."""
        hybrid_active = LayerMode.HYBRID in (
            self.config.tofu_mode, self.config.cicerone_mode,
            self.config.scene_mode, self.config.cleanse_mode,
            self.config.scribe_mode, self.config.verify_mode,
            self.config.memory_mode,
        )
        if hybrid_active and self.on_checkpoint is not None:
            return self.on_checkpoint(name, payload)
        return payload

    # --- Layer execution methods with mode checking ---

    def _run_tofu(
        self,
        asset,
        targ_lang: str,
        text_manifest: Optional[TextManifest] = None,
        font: Optional[str] = None,
    ) -> VldtnReport:
        if self.config.tofu_mode == LayerMode.MANUAL:
            # Return a "pending manual review" state
            return VldtnReport(
                passed=True,
                issues=[],
                suggested_actions=["Manual validation required for ToFU layer."]
            )
        return self._validator.validate(
            asset, targ_lang, self._tofu_context(font, text_manifest), text_manifest
        )

    @staticmethod
    def _tofu_context(
        font: Optional[str], text_manifest: Optional[TextManifest]
    ) -> Optional[Dict[str, Any]]:
        """What ToFU needs to score anything beyond the script map.

        The requested font is what makes glyph-coverage and text-expansion
        checks measurable rather than assumed -- process() has always
        received it and never passed it on, which left ToFU_005 falling
        back to a flat per-language ratio for every region.

        Region size and effect count come from the manifest once one
        exists, so the render-quality estimate sees the actual typography
        instead of its no-information default.
        """
        ctx: Dict[str, Any] = {}
        if font:
            ctx["font"] = font
        if text_manifest is not None:
            sizes = [
                inst.characteristics.size
                for inst in text_manifest.instances
                if inst.characteristics and inst.characteristics.size
            ]
            if sizes:
                # smallest region governs legibility: the run is only as
                # readable as the text most likely to fail
                ctx["font_px"] = min(sizes)
            effects = set()
            for inst in text_manifest.instances:
                sp = inst.style_profile
                if not sp:
                    continue
                if sp.shadow:
                    effects.add("shadow")
                if sp.stroke_width:
                    effects.add("stroke")
                if sp.italic:
                    effects.add("italic")
            if effects:
                ctx["effects"] = sorted(effects)
        return ctx or None

    def _run_cicerone(
        self,
        asset,
        asset_info: AssetInfo,
        scene_regions: Optional[List[SceneRegion]] = None,
    ) -> TextManifest:
        if self.config.cicerone_mode == LayerMode.MANUAL:
            # User provides manual annotations
            return self._get_manual_manifest(asset_info)
        return cicerone.detect(
            asset,
            asset_info,
            scene_regions=scene_regions,
            ocr_assessment_policy=self.config.ocr_assessment,
        )

    def _get_manual_manifest(self, asset_info: AssetInfo) -> TextManifest:
        """user-authored annotations for MANUAL mode; empty manifest if unset."""
        if self._manual_manifest is not None:
            return self._manual_manifest
        return TextManifest(
            asset_id=asset_info.source or "manual-asset",
            total_regions=0,
            instances=[],
            asset_type=asset_info.asset_type,
            frame_count=asset_info.frame_count,
        )

    def _run_scene(self, asset, text_manifest: TextManifest) -> TextManifest:
        if self.config.scene_mode == LayerMode.MANUAL:
            return text_manifest  # user supplies profiles via refinement
        return scene.analyze(asset, text_manifest)

    def _run_cleanse(self, asset, text_manifest: TextManifest):
        if self.config.cleanse_mode == LayerMode.MANUAL:
            return asset  # user supplies a manually cleansed asset downstream
        return cleanse.erase(
            asset,
            text_manifest,
            assessment_policy=self.config.inpaint_assessment,
        )

    def _run_scribe(
        self,
        cleansed_asset,
        text_manifest: TextManifest,
        targ_lang: str,
        render_params: Optional[Dict[str, RenderParams]] = None,
    ):
        if self.config.scribe_mode == LayerMode.MANUAL:
            return cleansed_asset  # user renders text manually
        return scribe.render(
            cleansed_asset, text_manifest, targ_lang, render_params,
            font_registry=self.font_registry,
        )

    def _run_verify(
        self, localized_asset, text_manifest: TextManifest, source_asset=None,
        cleansed_asset=None,
    ) -> QAReport:
        if self.config.verify_mode == LayerMode.MANUAL:
            # neutral report pending human sign-off; does NOT auto-pass the gate
            return QAReport(
                overall_score=None,
                recommendations=["Manual QA review required."]
            )
        return verify.assess(localized_asset, text_manifest, source_asset, cleansed_asset)

    def _run_memory(
        self,
        text_manifest: TextManifest,
        targ_lang: str,
        localized_asset,
        qa_report: QAReport,
        source_asset=None,
    ) -> List[Dict[str, Any]]:
        if self.config.memory_mode == LayerMode.MANUAL:
            return []  # user curates the TM by hand
        return memory.update(
            text_manifest, targ_lang, localized_asset, source_asset,
            qa_report, self.config.qa_threshold
        )
