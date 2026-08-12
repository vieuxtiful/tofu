## 🍢 ToFU — contiguous region ids, and every reference that follows them
## vieuxtiful
"""The invariant: a persisted manifest's live regions are exactly r1…rN.

Every test here is really the same question asked from a different angle --
after this operation, does anything still point at a region that moved? The
defect these exist to prevent is not a wrong NUMBER, which is visible, but a
reference that silently resolves to a different region, which is not.
"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.core.types import BBox, InstText, SemanticTextUnit, TextManifest  # noqa: E402
from tofu.layers import couvert  # noqa: E402
from tofu.layers.mise import make_blocks  # noqa: E402


def region(rid, *, order=None, excluded=False, text=None):
    return InstText(
        id=rid, bounding_box=BBox(10, 10, 40, 20), text=text,
        reading_order=order, excluded=excluded,
    )


def unit(uid, region_ids, text="x"):
    """The REAL type. Dicts would skip the serialization safety net, which is
    the half of the compactor most worth exercising."""
    return SemanticTextUnit(
        id=uid, region_ids=list(region_ids), source_text=text,
        bbox=BBox(10, 10, 40, 20),
    )


def manifest(*instances, blocks=(), units=()):
    m = TextManifest(
        asset_id="a1", total_regions=len(instances), instances=list(instances),
        img_dim=(400, 200), scene_regions=[],
    )
    if blocks:
        m.guided_blocks = make_blocks(list(blocks), "fr-FR")
    if units:
        m.semantic_units = list(units)
    return m


def live_ids(m):
    return [i.id for i in m.instances if not i.excluded]


class TestTheInvariant:
    def test_an_empty_manifest_is_trivially_contiguous(self):
        m = manifest()
        assert couvert.compact_region_ids(m) == {}
        couvert.validate(m)

    def test_regions_become_r1_upward_in_reading_order(self):
        m = manifest(region("r9", order=1), region("r4", order=0), region("r7", order=2))
        couvert.compact_region_ids(m)
        assert [(i.id, i.reading_order) for i in m.instances] == [
            ("r2", 1), ("r1", 0), ("r3", 2),
        ]

    def test_a_gap_left_by_a_purge_closes(self):
        """prem-sais-gt exactly: seven regions numbered r9…r15."""
        m = manifest(*[region(f"r{n}", order=n - 9) for n in range(9, 16)])
        couvert.compact_region_ids(m)
        assert live_ids(m) == [f"r{n}" for n in range(1, 8)]

    def test_reading_order_is_renormalized_with_the_ids(self):
        ## prem-sais-gt's reading_order is `8, 1, 2, 3, 4, 5, 6` -- no 0 and
        ## no 7, gappy for the same reason the ids were. Renumbering off a
        ## field that is itself damaged would carry the damage forward.
        m = manifest(region("r9", order=8), region("r10", order=1),
                     region("r11", order=2))
        couvert.compact_region_ids(m)
        assert sorted(i.reading_order for i in m.instances) == [0, 1, 2]
        by_id = {i.id: i.reading_order for i in m.instances}
        assert by_id == {"r1": 0, "r2": 1, "r3": 2}

    def test_compaction_is_idempotent(self):
        m = manifest(region("r3", order=0), region("r1", order=1))
        couvert.compact_region_ids(m)
        first = live_ids(m)
        assert couvert.compact_region_ids(m) == {"r1": "r1", "r2": "r2"}
        assert live_ids(m) == first

    def test_ordering_is_total_when_reading_order_ties(self):
        ## Two regions with the same reading_order must not swap depending on
        ## iteration order; the current numeric id breaks the tie.
        m = manifest(region("r8", order=0), region("r2", order=0))
        couvert.compact_region_ids(m)
        assert [i.id for i in m.instances] == ["r2", "r1"]


class TestSoftDeletedRegionsLeaveTheOrdinalSpace:
    def test_deleting_the_middle_region_closes_the_gap(self):
        """r3 of five deleted leaves r1…r4, as specified."""
        m = manifest(*[region(f"r{n}", order=n - 1) for n in range(1, 6)])
        m.instances[2].excluded = True
        couvert.compact_region_ids(m)
        assert live_ids(m) == ["r1", "r2", "r3", "r4"]

    def test_a_deleted_region_is_retired_not_dropped(self):
        ## It must survive: `update_region` can set excluded=False and bring
        ## it back, and a dropped region could not be restored.
        m = manifest(region("r1", order=0), region("r2", order=1))
        m.instances[0].excluded = True
        couvert.compact_region_ids(m)
        assert [i.id for i in m.instances] == ["x1", "r1"]

    def test_restoring_a_retired_region_returns_it_to_the_ordinal_space(self):
        m = manifest(region("r1", order=0), region("r2", order=1))
        m.instances[0].excluded = True
        couvert.compact_region_ids(m)
        m.instances[0].excluded = False
        couvert.compact_region_ids(m)
        assert live_ids(m) == ["r1", "r2"]

    def test_a_retired_region_never_collides_with_a_live_one(self):
        m = manifest(*[region(f"r{n}", order=n - 1) for n in range(1, 4)])
        m.instances[0].excluded = True
        couvert.compact_region_ids(m)
        ids = [i.id for i in m.instances]
        assert len(set(ids)) == len(ids)


class TestReferencesFollowTheRegions:
    def test_guided_block_evidence_follows(self):
        m = manifest(region("r9", order=0), blocks=["SORTIE"])
        block = m.guided_blocks[0]
        block.detection_assessment = {
            "status": "complete", "region_ids": ["r9"],
            "explicit_region_ids": ["r9"],
            "evidence": [{"atom_id": "g1a1", "region_id": "r9", "basis": "exact"}],
        }
        couvert.compact_region_ids(m)
        assessment = m.guided_blocks[0].detection_assessment
        assert assessment["region_ids"] == ["r1"]
        assert assessment["explicit_region_ids"] == ["r1"]
        assert assessment["evidence"][0]["region_id"] == "r1"

    def test_semantic_units_follow(self):
        m = manifest(region("r9", order=0), region("r10", order=1),
                     units=[unit("u1", ["r10", "r9"])])
        couvert.compact_region_ids(m)
        assert m.semantic_units[0].region_ids == ["r2", "r1"]

    def test_a_regions_semantic_assignment_follows(self):
        m = manifest(region("r9", order=0), region("r10", order=1))
        m.instances[1].semantic_assignment = {"semantic_region_id": "r9"}
        couvert.compact_region_ids(m)
        assert m.instances[1].semantic_assignment["semantic_region_id"] == "r1"

    def test_the_returned_map_is_what_external_tables_need(self):
        ## `tm_records.region_id` lives in the DB, not the manifest, so the
        ## caller migrates it -- and can only do that with this map.
        m = manifest(region("r9", order=0), region("r4", order=1))
        assert couvert.compact_region_ids(m) == {"r4": "r2", "r9": "r1"}

    def test_ids_belonging_to_other_namespaces_are_untouched(self):
        ## Units are `u1`, atoms are `g1a1`. Renaming those would be this
        ## module exceeding its remit.
        m = manifest(region("r9", order=0), blocks=["SORTIE"])
        m.guided_blocks[0].detection_assessment = {
            "region_ids": ["r9"], "explicit_region_ids": [],
            "evidence": [{"atom_id": "g1a1", "region_id": "r9", "basis": "exact"}],
        }
        couvert.compact_region_ids(m)
        assert m.guided_blocks[0].detection_assessment["evidence"][0]["atom_id"] == "g1a1"

    def test_unknown_serialized_reference_site_fails_discovery_guard(self):
        m = manifest(region("r9", order=0))
        m.candidate_lineage = {"future_reference_site": {"region_id": "r9"}}
        with pytest.raises(couvert.CompactionError, match="does not rewrite"):
            couvert.compact_region_ids(m)


class TestDanglingReferencesAreDetectedNotDisguised:
    def _damaged(self):
        """prem-sais-gt's actual shape: units naming regions that are gone."""
        return manifest(
            region("r9", order=0), region("r10", order=1),
            units=[unit("u1", ["r1"])],
        )

    def test_compaction_refuses_by_default(self):
        with pytest.raises(couvert.CompactionError, match="do not exist"):
            couvert.compact_region_ids(self._damaged())

    def test_a_dangling_reference_never_silently_becomes_valid(self):
        ## The whole point. r9 is about to BECOME r1, so a stale "r1" would
        ## start resolving to a real -- and wrong -- region.
        m = self._damaged()
        with pytest.raises(couvert.CompactionError):
            couvert.compact_region_ids(m)
        assert m.semantic_units[0].region_ids == ["r1"]  ## unchanged
        assert [i.id for i in m.instances] == ["r9", "r10"]  ## nothing renamed

    def test_they_can_be_dropped_explicitly(self):
        m = self._damaged()
        couvert.compact_region_ids(m, on_dangling="drop")
        assert m.semantic_units[0].region_ids == []
        assert live_ids(m) == ["r1", "r2"]

    def test_the_plan_reports_them_without_mutating(self):
        m = self._damaged()
        proposal = couvert.plan(m)
        assert proposal.dangling == {"r1"}
        assert [i.id for i in m.instances] == ["r9", "r10"]

    def test_dropping_removes_evidence_naming_a_vanished_region(self):
        m = manifest(region("r9", order=0), blocks=["SORTIE"])
        m.guided_blocks[0].detection_assessment = {
            "region_ids": ["r9", "r3"], "explicit_region_ids": ["r3"],
            "evidence": [
                {"atom_id": "g1a1", "region_id": "r9", "basis": "exact"},
                {"atom_id": "g1a2", "region_id": "r3", "basis": "exact"},
            ],
        }
        couvert.compact_region_ids(m, on_dangling="drop")
        assessment = m.guided_blocks[0].detection_assessment
        assert assessment["region_ids"] == ["r1"]
        assert assessment["explicit_region_ids"] == []
        assert [e["region_id"] for e in assessment["evidence"]] == ["r1"]


class TestValidation:
    def test_it_rejects_a_gap(self):
        m = manifest(region("r1", order=0), region("r3", order=1))
        with pytest.raises(couvert.CompactionError, match="not contiguous"):
            couvert.validate(m)

    def test_it_rejects_duplicates(self):
        m = manifest(region("r1", order=0), region("r1", order=1))
        with pytest.raises(couvert.CompactionError):
            couvert.validate(m)

    def test_it_rejects_a_reference_to_a_missing_region(self):
        m = manifest(region("r1", order=0), units=[unit("u1", ["r2"])])
        with pytest.raises(couvert.CompactionError, match="do not exist"):
            couvert.validate(m)

    def test_a_compacted_manifest_passes(self):
        m = manifest(region("r5", order=1), region("r2", order=0),
                     units=[unit("u1", ["r5"])])
        couvert.compact_region_ids(m)
        couvert.validate(m)


class TestExternalReferencesAreHistorical:
    """`tm_records.region_id` records where a translation CAME FROM.

    It is written once and never used as a lookup key -- memory is found by
    project, target language and text. Rewriting it to follow a renumbering
    would claim a memory came from a region that did not exist when the
    memory was made.
    """

    def test_the_map_is_returned_so_a_caller_could_migrate(self):
        ## The decision not to migrate is a POLICY, not a missing capability:
        ## the information needed to do it is returned either way.
        m = manifest(region("r5", order=0), region("r2", order=1))
        mapping = couvert.compact_region_ids(m)
        assert mapping == {"r2": "r2", "r5": "r1"}

    def test_compaction_touches_nothing_outside_the_manifest(self):
        ## Compaction is a pure manifest operation -- it takes no database
        ## handle and no store path, so it CANNOT reach an external table.
        ## Asserted rather than assumed, because the docstring says so.
        import inspect
        signature = inspect.signature(couvert.compact_region_ids)
        assert list(signature.parameters) == ["manifest", "on_dangling"]


def test_random_mutation_sequences_always_settle_to_the_invariant():
    rng = random.Random(20260812)
    for _case in range(80):
        m = manifest()
        next_id = 1
        for _step in range(40):
            live = [item for item in m.instances if not item.excluded]
            retired = [item for item in m.instances if item.excluded]
            operation = rng.choice(("add", "delete", "restore", "reorder"))
            if operation == "add" or not m.instances:
                m.instances.append(region(f"r{next_id + 20}", order=rng.randrange(0, 20)))
                next_id += 1
            elif operation == "delete" and live:
                rng.choice(live).excluded = True
            elif operation == "restore" and retired:
                rng.choice(retired).excluded = False
            elif operation == "reorder" and live:
                rng.choice(live).reading_order = rng.randrange(0, 20)
            m.total_regions = sum(not item.excluded for item in m.instances)
            couvert.compact_region_ids(m)
            couvert.validate(m)
