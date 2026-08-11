# Open Visual Translation Memory (VTM) — Version 1.0

**Status**: Draft
**Editor**: vieuxtiful
**Namespace**: `urn:vieuxtiful:vtm:1.0`
**Schema**: [`vtm-1.0.schema.json`](vtm-1.0.schema.json)
**Reference implementation**: [`tofu.utils.vtm`](../src/tofu/utils/vtm.py)
**License**: MIT

The key words MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT, MAY and OPTIONAL
are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## 1. Motivation

A conventional translation memory stores a pair: source string, target
string. That is sufficient to reuse a *string* and insufficient to reuse how
the string *looked*.

Localizing text that lives inside an image — signage, game UI, menus,
packaging, screenshots — needs more. To place `Grilled Meat` where `焼肉`
was, a tool must know where the source sat, how large it was, what face and
colour it used, what it sat on, and how much any of that can be trusted. None
of that fits in a TMX segment, so in practice it is discarded and
re-established by hand for every asset.

VTM v1.0 defines the record that carries it.

**Non-goals.** VTM is not an image format, not a rendering specification, and
not a replacement for XLIFF or TMX. It describes *what was translated and how
it was presented*. Section 7 defines how it rides alongside XLIFF rather than
competing with it.

---

## 2. Conformance

| Level | Requirement |
|---|---|
| **Core** | Everything in §4 marked REQUIRED. A Core document is a valid visual translation memory. |
| **Styled** | Core, plus `style` on every entry that had one. Enables faithful re-rendering. |
| **Full** | Styled, plus `background`, `provenance` and `visual`. Enables erase-and-re-render and cross-asset matching. |

A **conforming reader** MUST accept any document valid at Core level, MUST
ignore keys it does not recognise, and MUST NOT let the presence or absence
of an extension key change how it interprets standard keys.

A **conforming writer** MUST emit Core, and SHOULD emit the highest level its
data supports.

---

## 3. Coordinate system

This is the specification's most consequential decision, so it is stated once
and without exception.

- Coordinates are in **pixels**, in the coordinate space of `asset`.
- The **origin is the top-left** corner of the asset.
- **x increases rightward, y increases downward.**
- `asset.width` and `asset.height` are **REQUIRED**.

The last point is the one that matters. A bounding box of
`(10, 20, 100, 30)` describes a completely different region on a 640-pixel
thumbnail than on a 4096-pixel original. A memory that records geometry
without its resolution is not merely lossy — it is silently wrong the moment
anything is resized, and the failure surfaces as text placed confidently in
the wrong place. Requiring the dimensions makes that error *detectable*: a
validator can reject a box that extends past the asset, which is the
signature of coordinates captured at another scale.

Consumers needing normalized coordinates divide by the asset dimensions.
Nobody has to reverse-engineer an origin.

---

## 4. Document structure

```json
{
  "vtm_version": "1.0",
  "generator": { "name": "ToFU", "version": "1.0.0" },
  "created": "2026-07-25T15:45:35+00:00",
  "source_language": "ja-JP",
  "target_language": "en-US",
  "asset": { "id": "street-01", "width": 1024, "height": 768 },
  "entries": [ ... ]
}
```

| Key | Level | Notes |
|---|---|---|
| `vtm_version` | **REQUIRED** | `"1.0"`. Readers implementing 1.x MUST accept any 1.y. |
| `source_language` | **REQUIRED** | BCP 47. Entries MAY override. |
| `target_language` | **REQUIRED** | BCP 47. Entries MAY override. |
| `asset` | **REQUIRED** | `width` and `height` REQUIRED; see §3. |
| `entries` | **REQUIRED** | Array; MAY be empty. |
| `generator` | SHOULD | Provenance only. Readers MUST NOT branch on it. |
| `created` | SHOULD | RFC 3339. |

### 4.1 Entry

```json
{
  "id": "r1",
  "source": "焼肉",
  "target": "Grilled Meat",
  "geometry": {
    "bbox": { "x": 10, "y": 20, "width": 120, "height": 40 },
    "polygon": [[10,20],[130,20],[130,60],[10,60]],
    "reading_order": 1
  },
  "style": {
    "font_family": "arial.ttf", "font_size": 28, "color": "#1a1a1a",
    "weight": "bold", "align": "center",
    "stroke": { "color": "#ffffff", "width": 2.0 }
  },
  "background": { "color": "#8a6a4f", "texture": "textured" },
  "provenance": { "confidence": 0.93, "qa_score": 0.88, "engine": "paddleocr" },
  "visual": { "phash": "b4f1…", "style_fingerprint": "…" },
  "x-tofu": { "style": { "tsume": 0.2 } }
}
```

| Key | Level | Notes |
|---|---|---|
| `id` | **REQUIRED** | Unique within the document. |
| `source` | **REQUIRED** | Text as it appears in the asset. MAY be empty for a region detected but not recognised. |
| `target` | **REQUIRED** | Localized text. MAY be empty when exporting a job *for* translation. |
| `geometry` | **REQUIRED** | `bbox` REQUIRED; `polygon`, `rotation`, `reading_order` OPTIONAL. |
| `style` | Styled | §5. |
| `background` | Full | What the text sat on — enough to erase and re-render. |
| `provenance` | Full | §6. |
| `visual` | Full | Fingerprints for cross-asset matching. |
| `source_language` / `target_language` | OPTIONAL | Per-entry override. Omit when identical to the document default. |

---

## 5. Style

`style` is deliberately a **subset** of what any given renderer knows. Every
key in it is one a competing implementation can reasonably be expected to
honour: family, size, colour, weight, italic, underline, alignment,
orientation, direction, tracking, leading, stroke, shadow.

Engine-specific controls do **not** belong here. A format that standardises
one implementation's internals is that implementation's serialization
wearing a spec's clothes, and nobody else can implement it. Those go to
extensions (§8) — ToFU, for instance, round-trips its CJK `tsume`
compression and its inpainting strategy under `x-tofu`, and a reader that
ignores both still gets a complete, useful record.

`font_family` is a **hint, not a guarantee**. A consumer will not always have
the face, and MUST be free to substitute. `provenance.glyph_fallback` records
when the *producer* already had to.

---

## 6. Provenance and trust

A visual TM that reuses a bad pair is worse than one that reuses nothing: it
propagates an error across every asset that matches it, with a confidence the
original never had.

`provenance.confidence` is recognition confidence for the **source**;
`provenance.qa_score` is measured quality of the rendered **target**. They
answer different questions and MUST NOT be conflated. `human_reviewed`
outranks both.

Consumers SHOULD apply a threshold before reusing an entry, and SHOULD prefer
`human_reviewed` entries over higher-scoring automatic ones.

---

## 7. XLIFF binding

VTM is designed to **ride alongside XLIFF, not to replace it.** Localization
runs on XLIFF; a format that demands a parallel pipeline does not get
adopted, however good its record is.

Under namespace `urn:vieuxtiful:vtm:1.0`, an entry attaches to the
`<trans-unit>` whose `id` matches its `id`:

```xml
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2"
       xmlns:vtm="urn:vieuxtiful:vtm:1.0">
  <file source-language="ja-JP" target-language="en-US" datatype="plaintext"
        original="street-01" vtm:asset-width="1024" vtm:asset-height="768">
    <body>
      <trans-unit id="r1">
        <source>焼肉</source>
        <target>Grilled Meat</target>
        <vtm:geometry x="10" y="20" width="120" height="40" reading-order="1"/>
        <vtm:style font-family="arial.ttf" font-size="28" color="#1a1a1a" weight="bold"/>
      </trans-unit>
    </body>
  </file>
</xliff>
```

A CAT tool with no VTM support ignores the foreign namespace and round-trips
the file unharmed — which is the property that makes adoption possible at
all. `vtm:asset-width`/`vtm:asset-height` sit on `<file>` because §3 requires
them and a `<trans-unit>` has no other way to anchor its geometry.

---

## 8. Extensions

Keys beginning `x-` are reserved for vendor data. Readers MUST ignore
unrecognised `x-` keys. Writers MUST NOT place in an extension anything the
standard keys can express — an implementation that writes its size to
`x-vendor.size` instead of `style.font_size` is not conforming, whatever else
it does.

### 8.1 `x-tofu` (informative)

ToFU writes two extension structures. Both are optional, and a document that
omits them is complete: every translation pair and its geometry lives in
`entries`. A conforming 1.0 reader that ignores `x-` keys loses no
translation data.

`x-tofu.plates` records which entries are read together as one unit — a
street name spelled across two signs is one plate over two entries. Each
plate carries:

| Key | Meaning |
|---|---|
| `plate_uid` | Durable identity. Stable across reordering, insertion and reload. |
| `display_number` | Presentation position only. MUST NOT be used to resolve a plate. |
| `origin` | `derived`, `guided` or `user`. |
| `revision` | Fingerprint of ordered `region_ids` + normalised source. |
| `source` | The plate's source text, which MAY differ from the concatenated entries when a source correction was accepted. |
| `region_ids` | Entry ids, in source reading order. |

A consumer returning a translation SHOULD match on `plate_uid` and compare
`revision`. A differing `revision` means the plate changed after the document
was written, and the translation MUST NOT be applied without review.

`x-tofu.lineage_ref` is a *reference*, not the data: `{asset_id, run_id,
digest}`. The detection lineage graph itself is deliberately not embedded —
it describes one detection run rather than portable translation memory, and
is not needed to reproduce the accepted visual result.

---

## 9. Versioning

`vtm_version` is `MAJOR.MINOR`. Minor versions are additive: a 1.x reader
MUST accept any 1.y document, ignoring unknown keys. A major version may
break compatibility and will change the namespace URI.

---

## 10. Reference implementation and conformance vectors

`tofu.utils.vtm` implements export, import and validation:

```python
from tofu.utils import vtm

doc = vtm.export_vtm(manifest, "ja", "en")
problems = vtm.validate(doc)          # [] when valid
instances = vtm.import_vtm(doc)       # raises on an invalid document
```

Validation is hand-written rather than delegated to a JSON Schema library,
so checking a document costs no dependency; `vtm-1.0.schema.json` is provided
for third-party tooling.

Conformance vectors are in [`conformance/`](conformance/): documents a
conforming reader MUST accept, and documents it MUST reject, each with the
reason. `tests/test_vtm_spec.py` runs the reference implementation against
all of them.
