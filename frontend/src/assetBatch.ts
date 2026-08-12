import type { UploadResponse } from "./api";

export const MAX_PROJECT_ASSETS = 30;

export function selectBatchFiles(files: File[], existingCount: number) {
  const capacity = Math.max(0, MAX_PROJECT_ASSETS - existingCount);
  return { accepted: files.slice(0, capacity), skipped: files.slice(capacity) };
}

export async function uploadWithConcurrency(
  files: File[],
  upload: (file: File) => Promise<UploadResponse>,
  onSettled: (result: { file: File; upload?: UploadResponse; error?: unknown }) => void,
  concurrency = 3,
): Promise<void> {
  let cursor = 0;
  async function worker() {
    while (cursor < files.length) {
      const file = files[cursor++];
      try {
        onSettled({ file, upload: await upload(file) });
      } catch (error) {
        onSettled({ file, error });
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, files.length) }, worker));
}
