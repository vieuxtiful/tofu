---
title: ToFU — Visual Text Localization
emoji: 🍢
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: Detect text in an image, erase it, re-render the translation in the original's style.
---

# 🍢 ToFU playground

Drop an image, adjust the detected regions, type translations, and download the
localized asset together with its **XLIFF** and its
[**VTM**](https://github.com/vieuxtiful/tofu/blob/main/spec/vtm-1.0.md) memory.

ToFU detects text inside an image, erases it, reconstructs the background, and
re-renders the translation matching the source's face, size, colour, stroke,
alignment and orientation — including HarfBuzz shaping for Indic and
South-East Asian scripts, and right-to-left handling for Arabic and Hebrew.

Full project: **https://github.com/vieuxtiful/tofu**

**Documentation:** [Architecture](https://github.com/vieuxtiful/tofu/blob/main/docs/architecture.md) ·
[API reference](https://github.com/vieuxtiful/tofu/blob/main/docs/api.md) ·
[Contributing](https://github.com/vieuxtiful/tofu/blob/main/CONTRIBUTING.md) ·
[Changelog](https://github.com/vieuxtiful/tofu/blob/main/CHANGELOG.md)

---

## What this Space can and cannot do

ToFU runs **three Python interpreters** on purpose. A Hugging Face Space is
**one container**, so two of them cannot be here:

| Stage | This Space | Full local install |
|---|---|---|
| Detection | EasyOCR (CRAFT + CRNN) | **+ PP-OCRv5** — measured 4× recall and 4× transcription accuracy on dense vertical CJK |
| Erasure | analytic — flat / gradient / Telea | **+ LaMa** neural inpainting |
| Rendering | full `scribe`: HarfBuzz shaping, RTL, vertical CJK columns, style matching | same |

The reason is not laziness about packaging. `paddlepaddle` **force-replaces**
numpy and OpenCV when installed, so PP-OCRv5 has to live in its own
interpreter and is driven out-of-process; LaMa needs a third with CUDA torch.
Both are absent here, and the consequence is stated in the UI rather than
silently degraded:

- **CJK-heavy signage reads worse here than it does locally.**
- **Busy or textured backgrounds erase less cleanly.**

A demo that quietly produces worse output than the tool it demonstrates
misrepresents the tool.

## Performance

Free Space hardware is 2 vCPU with no GPU. Inputs are downscaled to **1100 px**
on the longest edge before anything runs, because cleanse cost grows with area
and that is the single largest lever on whether this feels interactive.

Measured locally on a 627×489 street scene: **~22 s** to detect, **~58 s** to
erase and render. Expect slower on Space hardware. The first detection also
downloads ~100 MB of EasyOCR weights.

Override with `TOFU_PLAYGROUND_MAX_DIM`.

## Running it locally

```bash
pip install -e ".[ocr,shaping]"
pip install gradio==6.20.0
python playground/app.py
```

Then open http://127.0.0.1:7860. Locally you can also provision the two
isolated interpreters (`server/requirements-paddle.txt` and
`server/requirements-inpaint.txt`) to get the full-quality path — ToFU
discovers them at runtime and degrades cleanly when they are absent.

See [CONTRIBUTING.md](https://github.com/vieuxtiful/tofu/blob/main/CONTRIBUTING.md)
for the full three-venv setup instructions.

## Deploying as a Space

`app.py` and `requirements.txt` in this directory are self-contained. Copy both
to a Gradio Space repository along with this README (its YAML front-matter is
the Space configuration). `requirements.txt` installs ToFU from git, so the
Space tracks `main`.
