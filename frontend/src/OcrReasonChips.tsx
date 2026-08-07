import Tooltip from "@mui/material/Tooltip";
import { Info } from "lucide-react";
import { ocrReasonInfo } from "./ocrReasons";

/**
 * Renders one chip per OCR quality reason code, each wrapped in an MUI
 * Tooltip carrying a one-sentence explanation. Unknown codes fall back to
 * the legacy space-replaced label with no tooltip (so a new backend reason
 * never renders as an empty chip or an empty tooltip bubble).
 *
 * The backend (src/tofu/layers/savor.py, assess_ocr_quality) is the source of
 * truth for the codes; this component only translates them for display.
 */
export default function OcrReasonChips({ reasons }: { reasons: string[] }) {
  return (
    <span className="flex max-w-[28rem] flex-wrap items-center gap-1">
      {reasons.map((code) => {
        const info = ocrReasonInfo(code);
        const chip = (
          <span className="ocr-reason-chip">
            {info.label}
            {info.description && <Info size={10} className="ocr-reason-chip-icon" />}
          </span>
        );
        return info.description ? (
          <Tooltip key={code} title={info.description} placement="top" arrow>
            {chip}
          </Tooltip>
        ) : (
          <span key={code}>{chip}</span>
        );
      })}
    </span>
  );
}
