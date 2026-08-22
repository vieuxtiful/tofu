# ToFU Vision 2 operations

Vision 2 defaults to `off`. It remains recommendation-only in every enabled
state: detection never rewrites source OCR text. A user must explicitly accept
a visible proposal, and that request is guarded by the persisted recommendation
revision.

## Configuration

| Variable | Values | Default |
|---|---|---|
| `TOFU_VISION2_STATE` | `off`, `shadow`, `guided_review`, `auto_gt_recommend` | `off` |
| `TOFU_VISION2_CHECKPOINT` | proof-encoder checkpoint path | bundled checkpoint when present and enabled |
| `TOFU_VISION2_FUSION_CALIBRATION` | compatible Fusion JSON artifact path | unset |
| `TOFU_VISION2_SCENE_COUNTERFACTUAL` | boolean | `false` |

`shadow` records evidence but cannot surface or apply recommendations.
`guided_review` permits a reviewer to accept a review-band candidate explicitly.
`auto_gt_recommend` exposes only candidates that clear the fitted recommendation,
survival, margin, contradiction, domain, and artifact-compatibility gates.

Missing or corrupt checkpoints, missing calibration, unsupported domains,
revision mismatch, and missing fitted features fail closed. They may record a
fallback reason but do not fail ordinary detection or change the incumbent OCR.

## Measurement channel provenance

Every region carries `channel` and `channel_id` (`src/tofu/layers/flight.py`)
naming the decoder route that produced its reading — which detection pass,
zoom, merge, Paddle rescue, or verifier. This is **not** gated on
`TOFU_VISION2_STATE`: the incumbent arm is the baseline every channel
comparison is measured against, so an index that only existed when Vision 2 was
enabled could never say what the shipped pipeline measured.

The fields are server-owned, pure provenance, and stamped after the reading has
settled. They change no text, geometry, or decision, and a region whose route
cannot be traced is reported as `unregistered` rather than folded into a
registered neighbour. Stamping failure is logged and leaves the region
untouched; it can never fail a detection.

Note for anyone diffing serialized output: `off` no longer produces
byte-identical manifests against pre-2026-08-16 baselines. `channel` and
`channel_id` are the registered exclusions, on the same footing as timing
fields.

## Monitoring and rollback

`GET /api/manifest/{asset_id}/vision2-metrics` reports coverage, decision counts,
reviewer outcomes, fallback reasons, calibration revisions, revision skew,
sample sizes, and a `channels` block with per-route counts, stamping coverage,
and the number of regions on unregistered routes. Precision is intentionally
`null` until adjudicated labels exist.

Rollback is `TOFU_VISION2_STATE=off` followed by a server restart. Persisted
evidence and audit history remain readable; disabling the feature does not erase
them. The counterfactual arm can be disabled independently by setting
`TOFU_VISION2_SCENE_COUNTERFACTUAL=false`.
