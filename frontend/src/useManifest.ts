import { useCallback, useRef, useState } from "react";
import type { InstText } from "./api";

/**
 * Manifest state and undo/redo history.
 *
 * The manifest is the array of `InstText` regions that flows through every
 * stage of the pipeline. This hook owns the raw state plus the undo/redo
 * stacks that let the user step back through edits.
 *
 * `manifestSkipHistory` is exposed as a mutable ref: external callers set
 * `manifestSkipHistory.current = true` immediately before calling
 * `setManifest` when a change should NOT enter the undo history (programmatic
 * loads from detect/import/render that replace the whole manifest, not user
 * edits). The flag is consumed and reset inside `setManifest`.
 *
 * Batch undo: during a drag or text-typing session, `beginManifestBatch()`
 * captures the pre-interaction snapshot and `endManifestBatch()` pushes it
 * as a single undo entry, so one undo returns to the state before the
 * interaction started rather than to each intermediate position.
 */
export function useManifest() {
  const [manifest, setManifestRaw] = useState<InstText[]>([]);

  // undo/redo history for manifest edits
  const manifestUndoStack = useRef<InstText[][]>([]);
  const manifestRedoStack = useRef<InstText[][]>([]);
  const manifestSkipHistory = useRef(false);

  // Batch undo: during a drag or text-typing session, suppress per-pixel /
  // per-keystroke undo entries and push a single pre-interaction snapshot
  // when the batch ends.  One undo then returns to the state before the
  // interaction started, not to each intermediate position.
  const manifestBatching = useRef(false);
  const manifestBatchAnchor = useRef<InstText[] | null>(null);

  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);

  const syncUndoRedo = useCallback(() => {
    setCanUndo(manifestUndoStack.current.length > 0);
    setCanRedo(manifestRedoStack.current.length > 0);
  }, []);

  const setManifest = useCallback((updater: InstText[] | ((prev: InstText[]) => InstText[])) => {
    setManifestRaw((prev) => {
      const next = typeof updater === "function" ? (updater as (p: InstText[]) => InstText[])(prev) : updater;
      if (!manifestSkipHistory.current && !manifestBatching.current) {
        manifestUndoStack.current.push(prev);
        if (manifestUndoStack.current.length > 50) manifestUndoStack.current.shift();
        manifestRedoStack.current = [];
      }
      manifestSkipHistory.current = false;
      syncUndoRedo();
      return next;
    });
  }, [syncUndoRedo]);

  const beginManifestBatch = useCallback(() => {
    if (manifestBatching.current) return;
    manifestBatchAnchor.current = manifest;
    manifestBatching.current = true;
  }, [manifest]);

  const endManifestBatch = useCallback(() => {
    if (!manifestBatching.current) return;
    const anchor = manifestBatchAnchor.current;
    if (anchor !== null) {
      manifestUndoStack.current.push(anchor);
      if (manifestUndoStack.current.length > 50) manifestUndoStack.current.shift();
      manifestRedoStack.current = [];
    }
    manifestBatching.current = false;
    manifestBatchAnchor.current = null;
    syncUndoRedo();
  }, [syncUndoRedo]);

  const undoManifest = useCallback(() => {
    setManifestRaw((prev) => {
      const stack = manifestUndoStack.current;
      if (stack.length === 0) return prev;
      const previous = stack.pop()!;
      manifestRedoStack.current.push(prev);
      syncUndoRedo();
      return previous;
    });
  }, [syncUndoRedo]);

  const redoManifest = useCallback(() => {
    setManifestRaw((prev) => {
      const stack = manifestRedoStack.current;
      if (stack.length === 0) return prev;
      const next = stack.pop()!;
      manifestUndoStack.current.push(prev);
      syncUndoRedo();
      return next;
    });
  }, [syncUndoRedo]);

  /** Clear undo/redo stacks and reset flags. Called on asset reset / project switch. */
  const clearManifestHistory = useCallback(() => {
    manifestUndoStack.current = [];
    manifestRedoStack.current = [];
    manifestSkipHistory.current = false;
    syncUndoRedo();
  }, [syncUndoRedo]);

  return {
    manifest,
    setManifest,
    setManifestRaw,
    canUndo,
    canRedo,
    syncUndoRedo,
    beginManifestBatch,
    endManifestBatch,
    undoManifest,
    redoManifest,
    manifestSkipHistory,
    clearManifestHistory,
  };
}
