## 🍢 PaddleOCR rescue pass: trigger + merge logic (no live subprocess)
"""
should_paddle_rescue/run_paddle_rescue are pure-logic units over
InstText/SceneRegion/RawDetection -- these tests never touch the real
isolated .venv-paddle subprocess (that's test_paddle_bridge_live.py's
job). PaddleOCRBackend is patched at the class level since
run_paddle_rescue constructs its own instance internally rather than
receiving one, unlike EasyOCRBackend's __new__-and-monkeypatch pattern
used elsewhere in this test suite.
"""
from unittest.mock import patch

from tofu.core.types import BBox, InstText, SceneRegion
from tofu.layers.cicerone import (
    PADDLE_RESCUE_CONF_FLOOR,
    RawDetection,
    _prefer_paddle_on_overlap,
    _subdivide_bbox,
    _surface_coverage_frac,
    _surfaces_needing_help,
    _uncovered_scene_surfaces,
    run_paddle_rescue,
    should_paddle_rescue,
)


def det(x, y, w, h, text="t", conf=0.9):
    return RawDetection(
        polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        text=text, confidence=conf, language="ja",
    )


def _inst(x, y, w, h, text, conf, lang="ja"):
    return InstText(
        id="r1", bounding_box=BBox(x=x, y=y, width=w, height=h),
        text=text, confidence=conf, detected_language=lang,
    )


def _surface(x, y, w, h, label="text_cluster"):
    return SceneRegion(
        bbox=BBox(x=x, y=y, width=w, height=h),
        semantic_label=label, confidence=0.8,
    )


class TestUncoveredSceneSurfaces:
    def test_covered_surface_excluded(self):
        surfaces = [_surface(10, 10, 40, 40)]
        refs = [(BBox(x=10, y=10, width=40, height=40), 0.9)]
        assert _uncovered_scene_surfaces(surfaces, refs) == []

    def test_uncovered_surface_returned(self):
        surfaces = [_surface(10, 10, 40, 40)]
        assert _uncovered_scene_surfaces(surfaces, []) == surfaces

    def test_min_confidence_ignores_weak_coverage(self):
        # regression guard: a weak/wrong existing detection (e.g. japan-
        # street r12, "2F" misread as "T" at 0.305 confidence) must not
        # block the targeted retry -- "covered" now requires the
        # covering detection to also clear min_confidence
        surfaces = [_surface(10, 10, 40, 40)]
        refs = [(BBox(x=10, y=10, width=40, height=40), 0.3)]
        assert _uncovered_scene_surfaces(surfaces, refs, min_confidence=0.5) == surfaces

    def test_min_confidence_still_excludes_strong_coverage(self):
        surfaces = [_surface(10, 10, 40, 40)]
        refs = [(BBox(x=10, y=10, width=40, height=40), 0.9)]
        assert _uncovered_scene_surfaces(surfaces, refs, min_confidence=0.5) == []


class TestShouldPaddleRescue:
    def test_no_cjk_instances_skips(self):
        instances = [_inst(0, 0, 40, 20, "MAIN STREET", 0.9, lang="en")]
        assert should_paddle_rescue(instances, None) == (False, None)

    def test_low_confidence_cjk_triggers(self):
        instances = [_inst(0, 0, 40, 20, "目", PADDLE_RESCUE_CONF_FLOOR - 0.1)]
        should, lang = should_paddle_rescue(instances, None)
        assert should is True and lang == "ja"

    def test_confident_cjk_with_all_surfaces_covered_skips(self):
        instances = [_inst(10, 10, 40, 40, "居酒屋", 0.9)]
        surfaces = [_surface(10, 10, 40, 40)]
        assert should_paddle_rescue(instances, surfaces) == (False, "ja")

    def test_confident_cjk_with_uncovered_surface_triggers(self):
        instances = [_inst(10, 10, 40, 40, "居酒屋", 0.9)]
        surfaces = [_surface(10, 10, 40, 40), _surface(200, 200, 30, 30)]
        should, lang = should_paddle_rescue(instances, surfaces)
        assert should is True and lang == "ja"

    def test_dominant_language_by_area(self):
        # a small, low-confidence 'en' fragment must not outvote a large,
        # confident 'ja' region -- dominance is area-weighted
        instances = [
            _inst(0, 0, 200, 100, "居酒屋", 0.9, lang="ja"),
            _inst(0, 0, 5, 5, "x", 0.1, lang="en"),
        ]
        _, lang = should_paddle_rescue(instances, None)
        assert lang == "ja"


class TestRunPaddleRescue:
    def test_full_frame_replaces_weak_easyocr_read(self):
        detections = [det(10, 10, 40, 20, "目", 0.3)]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: [det(10, 10, 40, 20, "劇場通り", 0.96)]
        fake.detect_in_regions = lambda asset, bboxes: [[] for _ in bboxes]
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, None, "ja")
        assert result is not None
        assert len(result) == 1
        assert result[0].text == "劇場通り"

    def test_paddle_wins_even_against_a_confidently_wrong_easyocr_read(self):
        # regression guard for the exact bug measured live on japan-
        # street: EasyOCR's own confidence is not a reliable correctness
        # signal on this failure class -- a wrong '目' misread scored
        # 0.795-0.96, comfortably beating PaddleOCR's own (correct, but
        # more modest) confidence in a plain confidence-max merge. once
        # should_paddle_rescue has already decided EasyOCR's read for
        # this language group can't be trusted, Paddle should win the
        # overlap as long as ITS OWN read clears a modest floor, not by
        # out-scoring a signal already known to be miscalibrated here.
        detections = [det(10, 10, 40, 20, "目", 0.96)]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: [det(10, 10, 40, 20, "劇場通り", 0.65)]
        fake.detect_in_regions = lambda asset, bboxes: [[] for _ in bboxes]
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, None, "ja")
        assert result is not None
        assert result[0].text == "劇場通り"

    def test_weak_paddle_read_does_not_clobber_existing_detection(self):
        # a Paddle read that itself doesn't clear the overlap-win floor
        # is not trusted either -- protects an already-fine EasyOCR
        # read from being overwritten by a low-confidence Paddle guess
        detections = [det(10, 10, 40, 20, "居酒屋", 0.9)]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: [det(10, 10, 40, 20, "喝酒尾", 0.2)]
        fake.detect_in_regions = lambda asset, bboxes: [[] for _ in bboxes]
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, None, "ja")
        assert result is not None
        assert result[0].text == "居酒屋"

    def test_region_call_skipped_when_nothing_uncovered(self):
        detections = [det(10, 10, 40, 20, "劇場通り", 0.96)]
        surfaces = [_surface(10, 10, 40, 20)]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: []
        calls = {"region": 0}
        def region_call(asset, bboxes):
            calls["region"] += 1
            return [[] for _ in bboxes]
        fake.detect_in_regions = region_call
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, surfaces, "ja")
        assert result is None
        assert calls["region"] == 0

    def test_weak_existing_detection_does_not_block_targeted_retry(self):
        # the exact japan-street r12 bug: a wrong "T" misread at 0.305
        # confidence already "covers" the surface by the old presence-
        # only check, silently blocking the very retry meant to fix it.
        # the full-frame pass alone doesn't find it either (measured
        # live -- small floor-number signs are easily missed at frame
        # scale), so the targeted per-region call must still fire.
        detections = [det(463, 371, 19, 11, "T", 0.305)]
        surfaces = [_surface(463, 371, 19, 11)]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: []  # full-frame pass finds nothing here
        fake.detect_in_regions = lambda asset, bboxes: [
            [det(463, 371, 19, 11, "2F", 0.93)] for _ in bboxes
        ]
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, surfaces, "ja")
        assert result is not None
        assert any(d.text == "2F" for d in result)

    def test_region_call_fires_for_uncovered_surface(self):
        detections = []
        surfaces = [_surface(100, 100, 30, 60)]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: []
        fake.detect_in_regions = lambda asset, bboxes: [
            [det(100, 100, 30, 60, "東南荘", 0.99)] for _ in bboxes
        ]
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, surfaces, "ja")
        assert result is not None
        assert any(d.text == "東南荘" for d in result)

    def test_none_when_paddle_adds_nothing(self):
        detections = [det(10, 10, 40, 20, "劇場通り", 0.96)]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: []
        fake.detect_in_regions = lambda asset, bboxes: [[] for _ in bboxes]
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, [_surface(10, 10, 40, 20)], "ja")
        assert result is None

    def test_escalated_retry_fires_when_first_retry_still_truncated(self):
        # the r11-class bug: the default-threshold retry finds SOMETHING
        # but it only covers part of the tall vertical sign (missing り);
        # the escalated retry (higher unclip_ratio) must get one more
        # attempt and its more complete read must win.
        detections = []
        surfaces = [_surface(53, 334, 20, 62)]  # true sign extent
        calls = {"n": 0}
        def region_call(asset, bboxes):
            calls["n"] += 1
            if calls["n"] == 1:
                # default-threshold pass: short box, only 79% of height
                return [[det(53, 334, 20, 49, "剧場通", 0.65)] for _ in bboxes]
            # escalated pass: full box, all 4 characters
            return [[det(53, 334, 20, 62, "劇場通り", 0.94)] for _ in bboxes]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: []
        fake.detect_in_regions = region_call
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, surfaces, "ja")
        assert calls["n"] == 2
        assert result is not None
        assert any(d.text == "劇場通り" for d in result)

    def test_escalated_retry_skipped_when_first_retry_already_complete(self):
        detections = []
        surfaces = [_surface(53, 334, 20, 62)]
        calls = {"n": 0}
        def region_call(asset, bboxes):
            calls["n"] += 1
            return [[det(53, 334, 20, 62, "劇場通り", 0.94)] for _ in bboxes]
        fake = type("Fake", (), {})()
        fake.detect = lambda asset: []
        fake.detect_in_regions = region_call
        with patch("tofu.layers.cicerone.PaddleOCRBackend", return_value=fake):
            result = run_paddle_rescue(None, detections, surfaces, "ja")
        assert calls["n"] == 1
        assert result is not None
        assert result[0].text == "劇場通り"


class TestSurfaceCoverageFrac:
    def test_full_coverage_is_one(self):
        surface = BBox(x=0, y=0, width=20, height=60)
        det_bbox = BBox(x=0, y=0, width=20, height=60)
        assert _surface_coverage_frac(det_bbox, surface) == 1.0

    def test_short_box_on_tall_surface_is_partial(self):
        surface = BBox(x=0, y=0, width=20, height=62)
        det_bbox = BBox(x=0, y=0, width=20, height=49)
        assert _surface_coverage_frac(det_bbox, surface) < 0.85

    def test_narrow_box_on_wide_surface_is_partial(self):
        surface = BBox(x=0, y=0, width=100, height=20)
        det_bbox = BBox(x=0, y=0, width=70, height=20)
        assert _surface_coverage_frac(det_bbox, surface) < 0.85


class TestSurfacesNeedingHelp:
    def test_uncovered_surface_needs_help(self):
        surfaces = [_surface(10, 10, 40, 40)]
        assert _surfaces_needing_help(surfaces, []) == surfaces

    def test_fully_covered_surface_does_not_need_help(self):
        surfaces = [_surface(10, 10, 40, 40)]
        refs = [(BBox(x=10, y=10, width=40, height=40), 0.9)]
        assert _surfaces_needing_help(surfaces, refs) == []

    def test_truncated_coverage_still_needs_help(self):
        # covered (>50% containment, confident) but the detection only
        # spans a fraction of the surface's own dominant-axis extent
        surfaces = [_surface(53, 334, 20, 62)]
        refs = [(BBox(x=53, y=334, width=20, height=49), 0.9)]
        result = _surfaces_needing_help(surfaces, refs)
        assert len(result) == 1


class TestPreferPaddleOnOverlapAllMatches:
    """regression guard for china-street's 茂昌眼镜公司镜司 duplicate-
    tail bug: a single incoming Paddle detection for the FULL sign must
    replace EVERY overlapping raw fragment, not just the first one it
    happens to match -- a leftover fragment surviving into
    merge_vertical_columns' later composition is what produced the
    duplicated tail."""

    def test_single_paddle_detection_consumes_all_overlapping_fragments(self):
        # two raw EasyOCR fragments of the same tall sign: an almost-
        # complete read AND a stray tail-only leftover, both overlapping
        # the incoming full-sign Paddle detection
        base = [
            det(552, 68, 118, 500, "茂昌眼镜公", 0.7),
            det(552, 500, 118, 125, "镜司", 0.6),
        ]
        paddle = [det(552, 68, 118, 557, "茂昌眼镜公司", 0.9)]
        result = _prefer_paddle_on_overlap(base, paddle)
        assert len(result) == 1
        assert result[0].text == "茂昌眼镜公司"

    def test_non_overlapping_fragments_survive_untouched(self):
        base = [
            det(552, 68, 118, 500, "茂昌眼镜公", 0.7),
            det(900, 900, 20, 20, "unrelated", 0.9),
        ]
        paddle = [det(552, 68, 118, 557, "茂昌眼镜公司", 0.9)]
        result = _prefer_paddle_on_overlap(base, paddle)
        assert len(result) == 2
        assert {d.text for d in result} == {"茂昌眼镜公司", "unrelated"}

    def test_shorter_paddle_detection_does_not_clobber_longer_existing_read(self):
        # regression guard for the exact china-street bug measured live:
        # a full-frame pass correctly read "王開照相" (4 chars, 0.999
        # conf), but a later escalated retry -- PaddleOCR's own
        # inference is not perfectly deterministic run-to-run -- spat
        # out a stray 2-character "ET" fragment overlapping the SAME
        # area at a confidence that cleared PADDLE_OVERLAP_WIN_FLOOR.
        # a shorter paddle read must never destroy a longer existing one.
        base = [det(32, 697, 302, 76, "王開照相", 0.999)]
        paddle = [det(32, 697, 302, 76, "ET", 0.55)]
        result = _prefer_paddle_on_overlap(base, paddle)
        assert len(result) == 1
        assert result[0].text == "王開照相"

    def test_longer_but_untrustworthy_existing_read_does_not_block_correction(self):
        # regression guard for the exact opposite failure the length
        # guard above introduced: japan-street's gate sign originally
        # had an EasyOCR garbage read "闘己_度町二せ国" (8 nonsense
        # characters at 0.0 confidence -- not a real word, pure noise).
        # PaddleOCR's correct, 1.0-confidence "歌舞伎町一番街" (7 chars)
        # is one character "shorter" but must still win -- character
        # count from near-zero-confidence noise is not informative.
        base = [det(190, 122, 241, 41, "闘己_度町二せ国", 0.0)]
        paddle = [det(191, 123, 241, 41, "歌舞伎町一番街", 1.0)]
        result = _prefer_paddle_on_overlap(base, paddle)
        assert len(result) == 1
        assert result[0].text == "歌舞伎町一番街"

    def test_length_guard_still_applies_against_moderately_confident_existing_read(self):
        # the trustworthiness floor is the same PADDLE_OVERLAP_WIN_FLOOR
        # as everything else in this function -- an existing read at or
        # above it still protects its length, even if not maximally
        # confident
        base = [det(0, 0, 20, 60, "ABCD", 0.5)]  # exactly at PADDLE_OVERLAP_WIN_FLOOR
        paddle = [det(0, 0, 20, 60, "AB", 0.55)]
        result = _prefer_paddle_on_overlap(base, paddle)
        assert len(result) == 1
        assert result[0].text == "ABCD"

    def test_equal_or_longer_paddle_detection_still_wins(self):
        # the length guard must not block the legitimate correction
        # cases this function exists for -- a paddle read that's at
        # least as long as what it overlaps still replaces it
        base = [det(0, 0, 20, 60, "目", 0.96)]
        paddle = [det(0, 0, 20, 60, "劇場通り", 0.65)]
        result = _prefer_paddle_on_overlap(base, paddle)
        assert len(result) == 1
        assert result[0].text == "劇場通り"

    def test_duplicate_tail_case_still_resolves_with_length_guard(self):
        # the length guard compares against the LONGEST single
        # overlapped fragment, not their combined length -- otherwise
        # this exact duplicate-tail scenario (the bug _all_matches was
        # built to fix) would itself be blocked by the new guard
        base = [
            det(552, 68, 118, 500, "茂昌眼镜公", 0.7),   # 5 chars
            det(552, 500, 118, 125, "镜司", 0.6),         # 2 chars
        ]
        paddle = [det(552, 68, 118, 557, "茂昌眼镜公司", 0.9)]  # 6 chars
        result = _prefer_paddle_on_overlap(base, paddle)
        assert len(result) == 1
        assert result[0].text == "茂昌眼镜公司"

    def test_same_batch_whole_and_tile_do_not_duplicate(self):
        # regression guard: run_paddle_rescue's escalated retry probes
        # BOTH a surface's whole bbox and its own sub-tiles in one
        # batch. two entries in that SAME incoming batch that both
        # overlap one base entry must not each independently "see no
        # remaining overlap" and both get accepted -- that would
        # duplicate the sign's content just like the china-street bug,
        # but from two PADDLE reads instead of one paddle + one leftover.
        base = [det(449, 235, 145, 253, "招你み焼本練", 0.51)]
        whole = det(449, 235, 145, 253, "お好み焼本陣", 0.9)
        tile = det(454, 278, 30, 94, "焼本", 0.7)  # a sub-tile's own partial read
        result = _prefer_paddle_on_overlap(base, [whole, tile])
        assert len(result) == 1
        assert result[0].text == "お好み焼本陣"

    def test_same_batch_tile_wins_when_it_alone_is_more_complete(self):
        # if the whole-surface read is itself weak/short and a tile's
        # read of the same area is strictly more complete, the tile
        # should win regardless of which order they were processed in
        base = [det(449, 235, 145, 253, "x", 0.51)]
        weak_whole = det(449, 235, 145, 253, "招", 0.6)  # 1 char
        better_tile = det(454, 278, 30, 94, "お好み焼本陣", 0.9)  # 6 chars
        result = _prefer_paddle_on_overlap(base, [weak_whole, better_tile])
        assert len(result) == 1
        assert result[0].text == "お好み焼本陣"


class TestSubdivideBbox:
    def test_small_bbox_unchanged(self):
        b = BBox(x=10, y=10, width=50, height=50)
        assert _subdivide_bbox(b) == [b]

    def test_large_bbox_split_into_grid(self):
        # the real japan-street "お好み焼本陣" surface that buried a
        # tiny unrelated "2F" placard, undetectable in one big crop
        b = BBox(x=449, y=235, width=145, height=253)
        tiles = _subdivide_bbox(b)
        assert len(tiles) > 1
        for t in tiles:
            assert t.width <= 145 and t.height <= 145

    def test_tiles_cover_a_point_inside_the_original(self):
        b = BBox(x=449, y=235, width=145, height=253)
        tiles = _subdivide_bbox(b)
        px, py = 463, 371  # the buried "2F" placard's location
        assert any(
            t.x <= px <= t.x + t.width and t.y <= py <= t.y + t.height
            for t in tiles
        )

    def test_wide_bbox_split_along_width_only(self):
        b = BBox(x=0, y=0, width=300, height=50)
        tiles = _subdivide_bbox(b)
        assert len(tiles) > 1
        assert all(t.height == 50 or t.height <= 50 + 20 for t in tiles)
