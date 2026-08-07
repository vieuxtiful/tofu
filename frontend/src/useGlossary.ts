import { useCallback, useEffect, useState } from "react";
import {
  GlossaryStatus,
  fetchGlossaryStatus,
  uploadGlossary,
  deleteGlossary,
} from "./api";
import type { ToastAction, ToastType } from "./ToastSystem";

/**
 * Basil glossary state and the upload/delete callbacks that talk to the
 * server's glossary endpoints.
 *
 * The glossary status is refetched whenever the active project changes
 * (the `useEffect` watches `projectId`). Upload and delete refresh the
 * status in-place after the server confirms.
 *
 * Cross-cutting dependencies passed in:
 *  - `projectId`   scopes project-level glossary operations
 *  - `srcLang`     passed to the upload endpoint for language-pair routing
 *  - `targLang`    ditto
 *  - `addToast`    for user feedback
 */
export function useGlossary(opts: {
  projectId: string | undefined;
  srcLang: string | null;
  targLang: string;
  addToast: (type: ToastType, message: string, autoDismiss?: boolean, action?: ToastAction, actions?: ToastAction[]) => void;
}) {
  const { projectId, srcLang, targLang, addToast } = opts;

  const [glossaryStatus, setGlossaryStatus] = useState<GlossaryStatus | null>(null);
  const [glossaryUploading, setGlossaryUploading] = useState(false);
  const [glossaryUploadStep, setGlossaryUploadStep] = useState<"marinate" | "ferment" | "set" | "done" | null>(null);
  const [glossaryUploadError, setGlossaryUploadError] = useState<string | null>(null);

  useEffect(() => {
    fetchGlossaryStatus(projectId).then(setGlossaryStatus).catch(() => setGlossaryStatus(null));
  }, [projectId]);

  const onGlossaryUpload = useCallback(async (file: File, scope: "global" | "project", mode: "auxiliary" | "merge" | "replace") => {
    setGlossaryUploading(true);
    setGlossaryUploadError(null);
    setGlossaryUploadStep("marinate");
    try {
      await new Promise((resolve) => setTimeout(resolve, 150));
      setGlossaryUploadStep("ferment");
      await uploadGlossary(file, scope, mode, projectId, srcLang ?? undefined, targLang ?? undefined);
      setGlossaryUploadStep("set");
      setGlossaryStatus(await fetchGlossaryStatus(projectId));
      setGlossaryUploadStep("done");
      addToast("success", `Basil glossary set from ${file.name}`);
      setTimeout(() => setGlossaryUploadStep(null), 1800);
    } catch (error) {
      const message = String(error);
      setGlossaryUploadError(message);
      setGlossaryUploadStep(null);
      addToast("error", `Basil glossary upload failed: ${message}`);
    } finally {
      setGlossaryUploading(false);
    }
  }, [projectId, srcLang, targLang, addToast]);

  const onGlossaryDelete = useCallback(async (scope: "global" | "project") => {
    try {
      await deleteGlossary(scope, projectId);
      setGlossaryStatus(await fetchGlossaryStatus(projectId));
      addToast("success", `${scope} Basil glossary cleared`);
    } catch (error) {
      addToast("error", `Basil glossary could not be cleared: ${error}`);
    }
  }, [projectId, addToast]);

  return {
    glossaryStatus,
    glossaryUploading,
    glossaryUploadStep,
    glossaryUploadError,
    onGlossaryUpload,
    onGlossaryDelete,
  };
}
