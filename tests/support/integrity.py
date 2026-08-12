"""One definition of persisted-manifest referential integrity.

Keep endpoint, sequence, corpus, repair, and export tests behind this oracle.
When the persistence contract grows, changing this file strengthens every
caller instead of leaving several nearly-identical assertions to drift.
"""

from __future__ import annotations

from typing import Any

from tofu.layers import couvert


def assert_manifest_integrity(manifest: Any, *, context: str = "") -> None:
    """Assert the invariants required at every manifest persistence boundary."""
    prefix = f"{context}: " if context else ""
    try:
        couvert.validate(manifest)
    except couvert.CompactionError as exc:
        raise AssertionError(f"{prefix}{exc}") from exc

    instances = list(getattr(manifest, "instances", None) or [])
    live = [region for region in instances if not getattr(region, "excluded", False)]
    retired = [region for region in instances if getattr(region, "excluded", False)]

    raw_orders = [getattr(region, "reading_order", None) for region in live]
    if any(order is not None for order in raw_orders):
        expected_orders = list(range(len(live)))
        assert all(isinstance(order, int) for order in raw_orders), (
            f"{prefix}reading_order is only partly assigned: {raw_orders}"
        )
        reading_orders = sorted(raw_orders)
        assert reading_orders == expected_orders, (
            f"{prefix}live reading_order is not contiguous: "
            f"{reading_orders} != {expected_orders}"
        )

    retired_ids = sorted(
        (region.id for region in retired),
        key=lambda region_id: int(region_id[1:]) if region_id[1:].isdigit() else -1,
    )
    expected_retired = [f"x{index}" for index in range(1, len(retired) + 1)]
    assert retired_ids == expected_retired, (
        f"{prefix}retired region ids are not contiguous: "
        f"{retired_ids} != {expected_retired}"
    )

    assert getattr(manifest, "total_regions", None) == len(live), (
        f"{prefix}total_regions={getattr(manifest, 'total_regions', None)} "
        f"but {len(live)} live regions exist"
    )

    for block in getattr(manifest, "guided_blocks", None) or []:
        assessment = getattr(block, "detection_assessment", None) or {}
        region_ids = set(assessment.get("region_ids") or [])
        explicit_ids = set(assessment.get("explicit_region_ids") or [])
        assert explicit_ids <= region_ids, (
            f"{prefix}Guided block {getattr(block, 'id', '<unknown>')} has "
            f"explicit regions outside region_ids: {sorted(explicit_ids - region_ids)}"
        )

    known = {region.id for region in instances}
    for unit in getattr(manifest, "semantic_units", None) or []:
        missing = set(getattr(unit, "region_ids", None) or []) - known
        assert not missing, (
            f"{prefix}semantic unit {getattr(unit, 'id', '<unknown>')} references "
            f"missing regions: {sorted(missing)}"
        )
