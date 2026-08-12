"""Build paired material-review crops and empty reviewer slots."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _save_rgb(path: Path, array) -> None:
    from PIL import Image

    Image.fromarray(array.astype("uint8"), mode="RGB").save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    import cv2

    from eval_scene_material_substrate import _instances
    from tofu.core.types import TextManifest
    from tofu.layers import scene
    from tofu.utils.imaging import load_rgb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "crops"
    image_dir.mkdir(exist_ok=True)
    queue = []
    cards = []
    for case in args.case:
        image_text, annotation_text = case.split("=", 1)
        image_path, annotation_path = Path(image_text), Path(annotation_text)
        instances, exclusion_source = _instances(annotation_path)
        manifest = TextManifest(
            asset_id=image_path.stem,
            total_regions=len(instances),
            instances=instances,
        )
        scene.analyze(str(image_path), manifest)
        image = load_rgb(str(image_path))
        height, width = image.shape[:2]
        for index, region in enumerate(manifest.scene_regions):
            surface_id = f"{image_path.stem}-surface-{index + 1:02d}"
            b = region.bbox
            x0, y0 = max(0, int(b.x)), max(0, int(b.y))
            x1 = min(width, x0 + int(b.width))
            y1 = min(height, y0 + int(b.height))
            contaminated = image[y0:y1, x0:x1]
            margin = max(12, int(max(b.width, b.height) * 0.2))
            cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
            cx1, cy1 = min(width, x1 + margin), min(height, y1 + margin)
            context = image[cy0:cy1, cx0:cx1].copy()
            cv2.rectangle(
                context,
                (x0 - cx0, y0 - cy0),
                (x1 - cx0 - 1, y1 - cy0 - 1),
                (255, 48, 48),
                2,
            )
            evidence = scene.substrate_material_evidence(
                str(image_path), region, instances, include_preview=True
            )
            context_name = f"{surface_id}-context.png"
            contaminated_name = f"{surface_id}-contaminated.png"
            _save_rgb(image_dir / context_name, context)
            _save_rgb(image_dir / contaminated_name, contaminated)
            excluded_name = None
            metadata = None
            if evidence is not None:
                excluded_name = f"{surface_id}-glyph-excluded.png"
                preview = evidence.pop("analysis_preview")
                _save_rgb(image_dir / excluded_name, preview)
                metadata = evidence
            record = {
                "asset_id": image_path.stem,
                "surface_id": surface_id,
                "source_image": str(image_path),
                "source_annotations": str(annotation_path),
                "exclusion_source": exclusion_source,
                "bbox": {"x": b.x, "y": b.y, "width": b.width, "height": b.height},
                "context_crop": f"crops/{context_name}",
                "contaminated_crop": f"crops/{contaminated_name}",
                "glyph_excluded_crop": f"crops/{excluded_name}" if excluded_name else None,
                "observation": metadata,
                "review_slots": [
                    {"slot": 1, "status": "pending", "review": None},
                    {"slot": 2, "status": "pending", "review": None}
                ],
                "adjudication": {"status": "pending"},
                "eligible_for_scoring": False,
            }
            queue.append(record)
            excluded_html = (
                f'<img src="crops/{html.escape(excluded_name)}" alt="glyph-excluded">'
                if excluded_name else '<div class="missing">No surviving substrate</div>'
            )
            cards.append(
                f'<article><h2>{html.escape(surface_id)}</h2>'
                f'<div class="triplet"><figure><img src="crops/{html.escape(context_name)}"><figcaption>Context</figcaption></figure>'
                f'<figure><img src="crops/{html.escape(contaminated_name)}"><figcaption>Contaminated</figcaption></figure>'
                f'<figure>{excluded_html}<figcaption>Glyph-excluded</figcaption></figure></div>'
                '<p>Reviewer 1: ____________________ &nbsp; Reviewer 2: ____________________</p>'
                '<p>Visible evidence: ____________________________________________________</p></article>'
            )

    packet = {
        "schema": 1,
        "kind": "scene_material_review_queue",
        "review_schema": "../scene-material-review-schema-v1.json",
        "labels_prefilled": False,
        "eligible_for_scoring": False,
        "surfaces": len(queue),
        "records": queue,
    }
    (args.out_dir / "review-queue.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    page = """<!doctype html><html><head><meta charset="utf-8"><title>ToFU material review</title>
<style>body{font:14px system-ui;margin:24px;background:#f4f4f4}article{background:white;padding:16px;margin:0 0 20px;border-radius:8px}.triplet{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img,.missing{width:100%;height:240px;object-fit:contain;background:#222}.missing{display:grid;place-items:center;color:white}figcaption{text-align:center;font-weight:600;margin-top:6px}</style>
</head><body><h1>ToFU Scene material review</h1><p>Labels are intentionally blank. Review independently before adjudication. Solid-colour blocks in glyph-excluded crops are median-filled exclusions, not reconstructed material.</p>"""
    page += "".join(cards) + "</body></html>"
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# Scene material review packet\n\nOpen `index.html`. Reviewers work independently. "
        "Copy completed judgments into records conforming to "
        "`../scene-material-review-schema-v1.json`. Solid-colour blocks in glyph-excluded "
        "crops are median-filled exclusions, not reconstructed material. Draft queue entries "
        "are deliberately ineligible for scoring until both reviews and adjudication are complete.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "surfaces": len(queue),
        "with_glyph_excluded_crop": sum(
            record["glyph_excluded_crop"] is not None for record in queue
        ),
        "labels_prefilled": False,
        "out_dir": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
