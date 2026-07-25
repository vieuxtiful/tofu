## 🍢 tofu.layers — the pipeline stages
"""One module per stage, plus the auxiliary passes they call into.

Pipeline order::

    tofu → scene → cicerone → tofu → scene → cleanse → scribe → verify → memory

Scene runs twice: once before Cicerone to constrain detection to candidate
surfaces, once after to enrich each detected instance with style and
background. ToFU likewise re-validates once real text exists.

Auxiliary passes: ``savor`` (glyph-confusion repair), ``menu`` (known-place
recovery), ``wasabi`` (CJK variant normalization) and ``basil`` (semantic
unification) run inside Cicerone; ``garnish`` and ``typography`` serve
Scene and Verify; ``knead`` provides HarfBuzz shaping to Scribe; ``fonts``
and ``font_matching`` back the font registry.

Deliberately NOT imported eagerly -- ``import tofu.layers`` must not pull
torch, OpenCV or an OCR engine into the process. Import the stage you want.
"""
