import { useCallback, useState } from "react";
import {
  SemanticSubstitutionPlan,
  SemanticTextUnit,
  UploadResponse,
  semanticSubstitution,
  semanticRepair,
  semanticModifyMembers,
  semanticCreateUnit,
  semanticDeleteUnit,
} from "./api";
import type { ToastAction, ToastType } from "./ToastSystem";

/**
 * Basil's semantic reading units — the "plate" layer that groups source
 * regions into translatable phrases separately from the immutable region
 * table. This hook owns the four pieces of semantic state and the six
 * callbacks that talk to the server's semantic endpoints.
 *
 * External callers (manifest load, detect, import, asset reset) need to
 * push new semantic_units into the hook, so `setSemanticUnits` and
 * `resetSemantic` are exposed alongside the state.
 *
 * Cross-cutting dependencies that stay in App.tsx are passed in:
 *  - `asset`           for asset_id on every API call
 *  - `targLang`        for the substitution target language
 *  - `addToast`        for user feedback
 *  - `setManifest`     onSemanticPlan updates the region table on Apply
 *  - `setImgDim`       onSemanticPlan updates image dimensions on Apply
 *  - `setSceneRegions` onSemanticPlan updates scene regions on Apply
 *  - `importedHash`    onSemanticPlan marks edits-after-import on Apply
 *  - `setHasEditsAfterImport` ditto
 */
export function useSemanticUnits(opts: {
  asset: UploadResponse | null;
  targLang: string;
  addToast: (type: ToastType, message: string, autoDismiss?: boolean, action?: ToastAction, actions?: ToastAction[]) => void;
  setManifest: (instances: import("./api").InstText[]) => void;
  setImgDim: (dim: [number, number] | null) => void;
  setSceneRegions: (regions: import("./api").SceneRegion[]) => void;
  importedHash: string | null;
  setHasEditsAfterImport: (v: boolean) => void;
}) {
  const { asset, targLang, addToast, setManifest, setImgDim, setSceneRegions, importedHash, setHasEditsAfterImport } = opts;

  const [semanticUnits, setSemanticUnits] = useState<SemanticTextUnit[]>([]);
  const [semanticDrafts, setSemanticDrafts] = useState<Record<string, string>>({});
  const [semanticPlans, setSemanticPlans] = useState<Record<string, SemanticSubstitutionPlan>>({});
  const [semanticBusyId, setSemanticBusyId] = useState<string | null>(null);

  /** Clear all semantic state. Called on asset reset / project switch. */
  const resetSemantic = useCallback(() => {
    setSemanticUnits([]);
    setSemanticDrafts({});
    setSemanticPlans({});
    setSemanticBusyId(null);
  }, []);

  const onSemanticDraftChange = useCallback((unitId: string, text: string) => {
    setSemanticDrafts((previous) => ({ ...previous, [unitId]: text }));
    // A changed phrase invalidates only that unit's prior plan. The existing
    // per-region targets remain untouched until an explicit Apply.
    setSemanticPlans((previous) => {
      if (!(unitId in previous)) return previous;
      const { [unitId]: _discarded, ...rest } = previous;
      return rest;
    });
  }, []);

  const onSemanticRepair = useCallback(async (unitId: string, accepted: boolean) => {
    if (!asset) return;
    setSemanticBusyId(unitId);
    try {
      const result = await semanticRepair(asset.asset_id, unitId, accepted);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      addToast("info", accepted ? "source reading corrected; regions untouched" : "proposal rejected; the original reading stands");
    } catch (error) {
      addToast("error", `could not record that decision: ${error}`);
    } finally {
      setSemanticBusyId(null);
    }
  }, [asset, addToast]);

  const onSemanticModifyMembers = useCallback(async (
    unitId: string,
    options: { addRegionId?: string; removeRegionId?: string },
  ) => {
    if (!asset) return;
    setSemanticBusyId(unitId);
    try {
      const result = await semanticModifyMembers(asset.asset_id, unitId, options);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      // Clear any stale plan/draft for the modified unit.
      setSemanticPlans((prev) => { const next = { ...prev }; delete next[unitId]; return next; });
      setSemanticDrafts((prev) => { const next = { ...prev }; delete next[unitId]; return next; });
      const action = options.removeRegionId ? "removed" : "added";
      addToast("info", `region ${action} from reading unit`);
    } catch (error) {
      addToast("error", `could not modify unit: ${error}`);
    } finally {
      setSemanticBusyId(null);
    }
  }, [asset, addToast]);

  const onSemanticCreateUnit = useCallback(async () => {
    if (!asset) return;
    try {
      const result = await semanticCreateUnit(asset.asset_id);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      addToast("info", "new plate created");
    } catch (error) {
      addToast("error", `could not create plate: ${error}`);
    }
  }, [asset, addToast]);

  const onSemanticDeleteUnit = useCallback(async (unitId: string) => {
    if (!asset) return;
    setSemanticBusyId(unitId);
    try {
      const result = await semanticDeleteUnit(asset.asset_id, unitId);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      // Clear any stale plan/draft for the deleted unit.
      setSemanticPlans((prev) => { const next = { ...prev }; delete next[unitId]; return next; });
      setSemanticDrafts((prev) => { const next = { ...prev }; delete next[unitId]; return next; });
      addToast("info", "plate deleted");
    } catch (error) {
      addToast("error", `could not delete plate: ${error}`);
    } finally {
      setSemanticBusyId(null);
    }
  }, [asset, addToast]);

  const onSemanticPlan = useCallback(async (unitId: string, apply: boolean) => {
    if (!asset) return;
    const unit = semanticUnits.find((item) => item.id === unitId);
    if (!unit) return;
    const target = (semanticDrafts[unitId] ?? unit.substitution?.target_text ?? unit.suggestion?.target_text ?? "").trim();
    if (!target) {
      addToast("warning", "enter a complete target phrase before planning placement");
      return;
    }
    setSemanticBusyId(unitId);
    try {
      const result = await semanticSubstitution(asset.asset_id, unitId, target, targLang, apply);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      setSemanticPlans((previous) => ({ ...previous, [unitId]: result.plan }));
      if (result.applied) {
        // This is a normal manifest edit, so the existing Capture/Localized
        // undo machinery can still return to the pre-substitution text.
        setManifest(result.manifest.instances);
        setImgDim(result.manifest.img_dim);
        setSceneRegions(result.manifest.scene_regions ?? []);
        setSemanticDrafts((previous) => ({ ...previous, [unitId]: target }));
        if (importedHash) setHasEditsAfterImport(true);
      } else if (result.plan.review_required) {
        addToast("warning", "Basil kept this phrase in review: no verified alignment evidence was available.");
      } else {
        addToast("info", "plate ready; review cube placement, then apply."); /* prev.: placement plan is ready; review the spatial anchors, then apply it. */
      }
    } catch (error) {
      addToast("error", `plating failed: ${error}`); /* prev.: translation substitut */
    } finally {
      setSemanticBusyId(null);
    }
  }, [asset, semanticUnits, semanticDrafts, targLang, importedHash, addToast, setManifest, setImgDim, setSceneRegions, setHasEditsAfterImport]);

  return {
    semanticUnits,
    setSemanticUnits,
    semanticDrafts,
    semanticPlans,
    semanticBusyId,
    resetSemantic,
    onSemanticDraftChange,
    onSemanticRepair,
    onSemanticModifyMembers,
    onSemanticCreateUnit,
    onSemanticDeleteUnit,
    onSemanticPlan,
  };
}
