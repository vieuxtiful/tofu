# Completed material reviews

Place one completed JSON record per surface in this directory. Each record must
conform to `../../scene-material-review-schema-v1.json`, contain two distinct
reviewers, and use the `asset_id` and `surface_id` from `../review-queue.json`.

Matching reviews require `adjudication.status: "agreed"` and the same resolved
material class. Disagreements require either a named adjudicator with a reason
or `status: "unresolvable"`. Do not copy placeholder reviewer identities or
machine material hypotheses into completed records.

Validate the directory with:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_scene_material_reviews.py `
  --queue evidence\scene-material-review-v1\review-queue.json `
  --schema evidence\scene-material-review-schema-v1.json `
  --reviews-dir evidence\scene-material-review-v1\completed `
  --out evidence\scene-material-review-status-v1.json
```
