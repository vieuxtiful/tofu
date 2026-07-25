## 🍢 tofu.core — shared data contracts and the pipeline orchestrator
"""Data types every layer exchanges (:mod:`tofu.core.types`) and the
orchestrator that threads an asset through them (:mod:`tofu.core.pipeline`).

Nothing here is imported eagerly: ``pipeline`` reaches for every layer, and
those layers reach for torch and OpenCV. Import the submodule you need.
"""
