## 🍢 ToFU playground — Gradio demo of the visual localization pipeline
## vieuxtiful
"""A hosted, scope-reduced ToFU: drop an image, adjust the detected regions,
type translations, and download the localized asset with its XLIFF and VTM.

WHAT THIS DELIBERATELY DOES NOT DO, and why.

ToFU runs three Python interpreters on purpose. PP-OCRv5 (its CJK detector)
lives in its own venv because paddlepaddle force-replaces numpy and OpenCV on
install, and LaMa (neural inpainting) lives in a third with CUDA torch. A
Hugging Face Space is one container, so neither can be here. This demo
therefore runs:

    detection   EasyOCR only          (strong on Latin, weaker on dense CJK)
    erasure     analytic cleanse only (flat / gradient / Telea, no LaMa)
    rendering   the full scribe layer, including HarfBuzz shaping and RTL

That is not a limitation of the pipeline, it is a limitation of one
container, and the difference is stated in the UI rather than quietly
degraded -- a demo that silently produces worse output than the real tool
misrepresents it.

The second constraint is time. A full-resolution street scene takes over
three minutes on CPU, most of it cleanse. Free Space hardware is 2 vCPU, and
nobody waits three minutes for a demo, so inputs are downscaled to
MAX_DIMENSION before anything runs. The localized image comes back at that
working size.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
from PIL import Image

from tofu.core.types import BBox, TextManifest
from tofu.layers import cicerone, cleanse, scribe
from tofu.utils import interchange, vtm

#: Inputs are downscaled to this longest edge before processing. Cleanse cost
#: grows with area, and this is the single biggest lever on whether the demo
#: feels interactive.
MAX_DIMENSION = int(os.environ.get("TOFU_PLAYGROUND_MAX_DIM", "1100"))

TABLE_HEADERS = ["id", "source", "target", "x", "y", "width", "height"]

SOURCE_LANGUAGES = [
    ("auto-detect", ""), ("Japanese", "ja"), ("Korean", "ko"),
    ("Chinese (simplified)", "zh-cn"), ("English", "en"), ("Spanish", "es"),
    ("French", "fr"), ("German", "de"), ("Russian", "ru"), ("Arabic", "ar"),
]
TARGET_LANGUAGES = [
    ("English", "en"), ("Spanish", "es"), ("French", "fr"), ("German", "de"),
    ("Japanese", "ja"), ("Korean", "ko"), ("Chinese (simplified)", "zh-cn"),
    ("Arabic", "ar"), ("Hebrew", "he"), ("Hindi", "hi"), ("Thai", "th"),
]


def _load_scaled(path: str) -> Tuple[Image.Image, float]:
    """Open an image at working size. Returns (image, scale_applied)."""
    image = Image.open(path).convert("RGB")
    longest = max(image.size)
    if longest <= MAX_DIMENSION:
        return image, 1.0
    scale = MAX_DIMENSION / longest
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
    return resized, scale


def _annotations(image: Image.Image, manifest: TextManifest) -> Tuple[Image.Image, List]:
    """(image, [(bbox_tuple, label)]) for gr.AnnotatedImage."""
    boxes = []
    for inst in manifest.instances:
        b = inst.adjusted_bbox or inst.bounding_box
        label = (inst.text or "?")[:24]
        boxes.append(((b.x, b.y, b.x + b.width, b.y + b.height), label))
    return image, boxes


def _to_table(manifest: TextManifest) -> List[List[Any]]:
    rows = []
    for inst in manifest.instances:
        b = inst.adjusted_bbox or inst.bounding_box
        rows.append([inst.id, inst.text or "", inst.target_text or "",
                     b.x, b.y, b.width, b.height])
    return rows


def _apply_table(manifest: TextManifest, table: Any) -> TextManifest:
    """Fold user edits back into the manifest.

    The table is the editable surface for BOTH the translation and the
    geometry, so a user can nudge a box that detection got slightly wrong and
    re-render without re-detecting. Rows are matched on id; unknown ids are
    ignored rather than trusted.
    """
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    by_id = {inst.id: inst for inst in manifest.instances}
    for row in rows:
        if not row or len(row) < len(TABLE_HEADERS):
            continue
        inst = by_id.get(str(row[0]))
        if inst is None:
            continue
        inst.target_text = str(row[2]) if row[2] is not None else ""
        try:
            x, y, w, h = (int(float(row[3])), int(float(row[4])),
                          int(float(row[5])), int(float(row[6])))
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            inst.adjusted_bbox = BBox(x=max(0, x), y=max(0, y), width=w, height=h)
    return manifest


def detect(image_path: Optional[str], src_lang: str, progress=gr.Progress()):
    """Run pre-flight + detection, and hand back an editable region table."""
    if not image_path:
        raise gr.Error("Drop an image first.")

    progress(0.1, desc="Loading image")
    image, scale = _load_scaled(image_path)

    progress(0.25, desc="Detecting text (first run downloads ~100 MB of models)")
    languages = [src_lang] if src_lang else None
    # EasyOCR accepts a path, bytes or an ndarray -- NOT a PIL image -- so the
    # working-size copy goes to disk for detection. Cleanse and scribe take the
    # PIL image directly, so only this one stage needs the round trip.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        working_path = handle.name
    image.save(working_path)
    try:
        manifest = cicerone.detect(
            working_path,
            languages=languages,
            # PP-OCRv5 is not installable in a single container; asking for the
            # rescue pass here would spawn a subprocess that cannot exist
            paddle_rescue=False,
        )
    finally:
        try:
            os.unlink(working_path)
        except OSError:
            pass

    if not manifest.instances:
        return (
            (image, []), [], None,
            "No text detected. Try a clearer image, or set the source language "
            "explicitly instead of auto-detect.",
        )

    progress(0.85, desc="Analysing style and background")
    from tofu.layers import scene
    manifest = scene.analyze(image, manifest)
    manifest.img_dim = (image.width, image.height)

    note = f"Found {len(manifest.instances)} region(s)"
    if scale != 1.0:
        note += f" · image scaled to {image.width}x{image.height} for speed"
    note += ". Edit the **target** column, adjust boxes if needed, then Render."
    return _annotations(image, manifest), _to_table(manifest), (image, manifest), note


def render(state, table, targ_lang: str, progress=gr.Progress()):
    """Erase the source text and re-render the targets in its style."""
    if not state:
        raise gr.Error("Detect text first.")
    image, manifest = state

    manifest = _apply_table(manifest, table)
    if not any(i.target_text for i in manifest.instances):
        raise gr.Error("Type at least one translation in the target column.")

    progress(0.15, desc="Erasing source text")
    cleansed = cleanse.erase(image, manifest)

    progress(0.6, desc="Rendering target text")
    localized = scribe.render(cleansed, manifest, targ_lang)

    progress(0.9, desc="Writing exports")
    out_dir = Path(tempfile.mkdtemp(prefix="tofu-playground-"))
    png_path = out_dir / "localized.png"
    localized.convert("RGB").save(png_path)

    src_lang = manifest.src_lang or "en"
    xliff_path = out_dir / "translations.xliff"
    xliff_path.write_text(
        interchange.export_xliff(manifest, src_lang, targ_lang), encoding="utf-8")

    # the VTM export is the interesting one: unlike XLIFF it carries geometry
    # and typography, so the memory can be reused on a different asset
    vtm_path = out_dir / "memory.vtm.json"
    document = vtm.export_vtm(manifest, src_lang, targ_lang,
                              image_size=(image.width, image.height))
    vtm_path.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    return localized, [str(png_path), str(xliff_path), str(vtm_path)]


with gr.Blocks(title="ToFU — visual text localization") as demo:
    gr.Markdown(
        """
        # 🍢 ToFU — visual text localization

        Detect text inside an image, erase it, and re-render the translation
        **in the original's style** — face, size, colour, stroke, orientation.

        1. Drop an image and **Detect text**
        2. Edit the **target** column (and nudge boxes if detection was off)
        3. **Render**, then download the asset with its XLIFF and
           [VTM](https://github.com/vieuxtiful/tofu/blob/main/spec/vtm-1.0.md) memory
        """
    )
    with gr.Accordion("What this demo can and cannot do", open=False):
        gr.Markdown(
            f"""
            This Space is **one container**, and ToFU normally runs three
            interpreters. What that costs here:

            | | This demo | Full local install |
            |---|---|---|
            | Detection | EasyOCR | + PP-OCRv5 — 4× recall on dense CJK |
            | Erasure | analytic (flat/gradient/Telea) | + LaMa neural inpainting |
            | Rendering | full scribe: HarfBuzz shaping, RTL, vertical CJK | same |

            PaddleOCR needs its own interpreter (`paddlepaddle` force-replaces
            numpy and OpenCV on install); LaMa needs a third with CUDA torch.
            Neither fits in a Space, so **CJK-heavy signage will read worse
            here than it does locally**, and busy backgrounds will erase less
            cleanly.

            Images are downscaled to {MAX_DIMENSION}px on the longest edge —
            cleanse cost grows with area and free Space hardware is 2 vCPU.
            The first detection downloads ~100 MB of EasyOCR weights.
            """
        )

    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="filepath", label="Source image", height=320)
            src_lang = gr.Dropdown(SOURCE_LANGUAGES, value="", label="Source language")
            targ_lang = gr.Dropdown(TARGET_LANGUAGES, value="en", label="Target language")
            detect_btn = gr.Button("Detect text", variant="primary")
            render_btn = gr.Button("Render", variant="secondary")
        with gr.Column(scale=2):
            annotated = gr.AnnotatedImage(label="Detected regions", height=320)
            status = gr.Markdown()

    table = gr.Dataframe(
        headers=TABLE_HEADERS,
        datatype=["str", "str", "str", "number", "number", "number", "number"],
        label="Regions — edit 'target', and the box if detection was off",
        interactive=True,
        wrap=True,
    )

    with gr.Row():
        output = gr.Image(label="Localized asset", height=380)
        downloads = gr.File(label="Downloads: PNG · XLIFF · VTM", file_count="multiple")

    detect_btn.click(detect, [image_in, src_lang], [annotated, table, state, status])
    render_btn.click(render, [state, table, targ_lang], [output, downloads])

    examples_dir = Path(__file__).resolve().parents[1] / "images"
    example_files = [
        str(examples_dir / name)
        for name in ("japan-street.jpeg", "china-street.png", "gemini-street.png")
        if (examples_dir / name).is_file()
    ]
    if example_files:
        gr.Examples(examples=example_files, inputs=image_in,
                    label="Try one of these street scenes")


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        # Gradio 6 moved theme from the Blocks constructor to launch()
        theme=gr.themes.Soft(),
        server_name=os.environ.get("TOFU_PLAYGROUND_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("TOFU_PLAYGROUND_PORT", "7860")),
    )
