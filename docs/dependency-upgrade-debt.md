# Dependency upgrade debt

## EasyOCR / PyTorch dynamic quantization

- Status: tracked; does not block Phase 3.
- Severity: low with the locked runtime, medium when upgrading OCR dependencies.
- Current lock: EasyOCR 1.7.2, PyTorch 2.13.0+cpu.
- Observed warnings: four `torch.ao.quantization` deprecations and one
  quantized-RNN tensor deprecation while EasyOCR initializes its CPU detector
  and recognizer.
- Ownership: upstream EasyOCR/PyTorch. ToFU does not call the deprecated APIs
  directly.
- Current impact: none observed; OCR and the complete test suite pass.
- Upgrade trigger: before changing either EasyOCR or PyTorch, run the Verify
  OCR tests with deprecations visible. Prefer an upstream EasyOCR release that
  migrates to torchao. If none exists, evaluate disabling EasyOCR CPU
  quantization before carrying a local compatibility patch.
- Do not suppress these warnings globally: they are the signal that the
  dependency-upgrade review is due.
