## 🍢 Menu — gazetteer-assisted correction for real-world place names
"""
A low-confidence OCR read of a real, well-known place or establishment
sign (a train-station gate, a named street, a chain storefront) can
often be recovered by checking it against a small list of known names,
even when the pixels alone weren't legible enough for the recognizer
to get right. Savor (glyph-confusion correction) already does something
similar for single-character digit/letter swaps, verified pixel-by-
pixel; whole-string place names can't be pixel-verified the same way
(there's no "render the candidate and compare stroke-for-stroke" for a
7-character sign), so this uses a stricter, cruder gate instead: only
touch reads the recognizer itself was already unsure about, and only
apply a candidate that's a strong fuzzy match.

`browse()` is the module's one public entry point (mirrors every other
layer's single-verb API: cicerone.detect, scene.analyze, savor.taste)
— called once by cicerone.detect(), on the FINAL manifest, after every
detection/refinement/recognition pass (including Savor's) has already
had its say.
"""

from typing import Any, Dict, List, NamedTuple, Optional

from tofu.core.types import ImageLike, InstText
from tofu.utils.correction_resources import gazetteer_entries, load_correction_resource
from tofu.utils.textmatch import fuzzy_similarity

# a read this confident is trusted over the gazetteer -- a correction
# only ever rescues text the recognizer was already unsure about, never
# overrides a confident (even if merely gazetteer-similar) read.
# measured live: japan-street's "招你み焼本練" (nonsense -- not a real
# Japanese word or name) scored 0.511 confidence, just above the old
# 0.5 floor, silently exempting it from a gazetteer check it would
# otherwise have passed (0.5 similarity against the real "お好み焼本陣"
# sign) -- 0.5-0.6 is still a genuinely uncertain read in absolute
# terms, not a confident one, so the floor moved to actually cover it.
CONFIDENCE_FLOOR = 0.6

# minimum fuzzy similarity to accept a gazetteer candidate. deliberately
# strict since, unlike Savor's per-glyph pixel check, there is no pixel
# verification step here -- a weak match is more likely coincidence
# than a genuine misread of the same sign.
SIMILARITY_FLOOR = 0.5

# the whole-string path only ever considers candidates of at least this
# length: a 2-character name sharing ONE character with a 2-character
# low-confidence read scores exactly 0.5 similarity -- coin-flip
# evidence that would rewrite ubiquitous short signage (e.g. "下り" ->
# "下島"). short names are served exclusively by the substring path,
# where pixel verification backs the weak string evidence.
WHOLE_STRING_MIN_CANDIDATE_LEN = 3

# substring (composite-read) matching: a directional post or stacked
# sign often OCRs as ONE instance concatenating several names, whose
# read confidence is an average across all of them -- meaningless for
# any single name, so this path is structurally gated (text strictly
# longer than the candidate) instead of confidence-gated.
SUBSTRING_SIMILARITY_FLOOR = 0.5
# string evidence alone may rewrite a span only when the candidate is
# long enough and the diff small enough that coincidence is implausible:
# at least 4 chars with exactly 1 differing (effective similarity 0.75+).
# measured counterexample that motivates the tightness: a correct
# "東南口" (3 chars, 1 diff) scores 0.667 against gazetteer "東南荘" --
# high-frequency short signage must never be rewritten on strings alone.
SUBSTRING_STRING_TIER_MIN_LEN = 4
SUBSTRING_STRING_TIER_MAX_DIFFS = 1

# corroboration tier: a composite read that ALREADY contains this many
# independently-confirmed gazetteer names (exact windows, or spans the
# string/pixel tiers applied) is, with high probability, a listing of
# real names (a directional post, a station board) -- document-level
# context that lets a remaining pixel-INCONCLUSIVE single-diff span
# apply where its own evidence alone couldn't decide (measured live:
# japan-subs' 13px 湯屋 glyphs score 湯 0.711 vs 周 0.676 -- leaning
# right but inside BITE_MARGIN). classic lexicon-driven OCR
# post-correction with document context. never fires over a pixel
# CONTRADICTION (those spans are discarded before this tier runs), and
# corroborated spans never bootstrap each other -- the sibling count is
# fixed before any promotion.
CORROBORATION_MIN_SIBLINGS = 2

# known place/establishment names likely to recur in street-signage
# photos. seeded for this project's dense-CJK-signage test scenes, but
# meant to grow with whatever real signage future assets turn up --
# not a fixed answer key for one image.
KNOWN_PLACES_RESOURCE = load_correction_resource("menu/known_places-1.0.0.json")
KNOWN_PLACES: List[tuple] = gazetteer_entries(KNOWN_PLACES_RESOURCE)

# common signage words -- not places, so they are deliberately NOT in
# KNOWN_PLACES (browse() must not rewrite a storefront read into a
# generic word). Basil consults this pool separately when deciding
# whether several fragmented regions spell one entity: cicerone
# routinely splits a two-glyph sign like 歓迎 into one region per glyph,
# and reassembling it is an entity question, not a place-name question.
KNOWN_SIGNAGE_RESOURCE = load_correction_resource("menu/known_signage-1.0.0.json")
KNOWN_SIGNAGE: List[tuple] = gazetteer_entries(KNOWN_SIGNAGE_RESOURCE)


class MenuMatch(NamedTuple):
    text: str
    similarity: float


class MenuSpan(NamedTuple):
    """one aligned gazetteer name inside a longer composite read."""
    start: int
    end: int              # half-open: text[start:end] is the window
    candidate: str
    similarity: float
    diffs: List[tuple]    # [(absolute_index, recognized_char, candidate_char)]


def _entry_parts(entry: tuple) -> tuple[str, Optional[str], Optional[str]]:
    candidate = str(entry[0])
    candidate_lang = entry[1] if len(entry) > 1 else None
    scope = entry[2] if len(entry) > 2 else None
    return candidate, candidate_lang, scope


def _ground_truth_identity(scope: str, terms: List[str]) -> Dict[str, Any]:
    return {
        "kind": "ground_truth",
        "scope": scope,
        "terms": sorted(set(terms)),
        "revision": "user-1",
    }


def consult_menu(text: str, lang: Optional[str], confidence: Optional[float],
                 ground_truth_pool: Optional[List[tuple]] = None) -> Optional[MenuMatch]:
    """check a low-confidence read against the known-places gazetteer.

    returns the best-matching known name if one clears SIMILARITY_FLOOR,
    or None if the read is already confident, empty, or nothing in the
    gazetteer resembles it closely enough.
    """
    if not text or (confidence or 0) >= CONFIDENCE_FLOOR:
        return None
    best: Optional[MenuMatch] = None
    entries = list(KNOWN_PLACES) + list(ground_truth_pool or [])
    for entry in entries:
        candidate, cand_lang, scope = _entry_parts(entry)
        minimum = 2 if scope else WHOLE_STRING_MIN_CANDIDATE_LEN
        if len(candidate) < minimum:
            continue
        if lang and cand_lang != lang:
            continue
        score = fuzzy_similarity(text, candidate)
        if score >= SIMILARITY_FLOOR and (best is None or score > best.similarity):
            best = MenuMatch(candidate, score)
    return best


def consult_menu_substring(text: str, lang: Optional[str],
                           ground_truth_pool: Optional[List[tuple]] = None) -> List[MenuSpan]:
    """find known names ALIGNED WITHIN a longer composite read.

    a directional post or stacked sign OCRs as one instance whose text
    concatenates several names (measured live: "周屋下島周河温泉" is
    really 湯屋 / 下島 / 濁河温泉 stacked on one post) -- the whole-
    string path can't see any single name inside that, so this slides
    an exact-length window per candidate and diffs it POSITIONALLY.
    positional (zip) diffing, not just fuzzy ratio, is load-bearing:
    for short names SequenceMatcher can't even disambiguate the
    alignment (measured: both "周屋" and the misaligned "屋下" score
    0.5 against 湯屋; only the zip diff -- 1-of-2 positions matching vs
    0-of-2 -- tells them apart).

    windows qualify only when: the text is STRICTLY longer than the
    candidate (equal-length reads belong to the whole-string and dakuten
    paths), the window isn't already correct, at least half its
    positions match exactly, and fuzzy similarity clears
    SUBSTRING_SIMILARITY_FLOOR. overlapping proposals across candidates
    are resolved greedily: higher similarity first, then longer
    candidate, then more matching positions, then leftmost. spans never
    overlap in the result; equal-length window replacement means
    corrections are length-preserving and can all be applied at once.
    """
    return align_spans(text, lang, list(KNOWN_PLACES) + list(ground_truth_pool or []))


def align_spans(text: str, lang: Optional[str], pool: List[tuple],
                allow_equal_length: bool = False) -> List[MenuSpan]:
    """The window/diff core of the substring path, over any candidate pool.

    Factored out so Basil can reuse the same alignment on a composite it
    builds by concatenating SEVERAL regions, rather than on one region's
    text.  ``allow_equal_length`` is the difference that case needs: when
    cicerone splits 歓迎 into one region per glyph, the reassembled
    composite is exactly as long as the candidate, which the within-region
    path deliberately refuses (an equal-length read there belongs to the
    whole-string and dakuten paths, which are confidence-gated).  Across
    regions there is no such sibling path, and the crossing of a region
    boundary is itself the structural evidence that gate was standing in
    for -- so Basil opts in explicitly and adds its own gates on top.
    """
    if not text or " " in text:
        # composite signage instances are never space-tokenized; a space
        # would also break the char-index-to-glyph-cluster alignment the
        # pixel tier depends on
        return []
    proposals: List[MenuSpan] = []
    for entry in pool:
        candidate, cand_lang, _scope = _entry_parts(entry)
        if lang and cand_lang != lang:
            continue
        n = len(candidate)
        if n < 2:
            continue
        if len(text) < n or (len(text) == n and not allow_equal_length):
            continue
        for start in range(0, len(text) - n + 1):
            window = text[start:start + n]
            if window == candidate:
                continue  # already correct -- nothing to propose
            diffs = [
                (start + i, a, b)
                for i, (a, b) in enumerate(zip(window, candidate))
                if a != b
            ]
            if 2 * len(diffs) > n:
                continue  # majority of positions must already match
            score = fuzzy_similarity(window, candidate)
            if score < SUBSTRING_SIMILARITY_FLOOR:
                continue
            proposals.append(MenuSpan(start, start + n, candidate, score, diffs))

    # greedy non-overlap resolution over the full pool
    proposals.sort(key=lambda s: (
        -s.similarity,
        -(s.end - s.start),
        len(s.diffs),
        s.start,
    ))
    kept: List[MenuSpan] = []
    for span in proposals:
        if any(span.start < k.end and k.start < span.end for k in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: s.start)
    return kept


def exact_spans(text: str, lang: Optional[str], pool: List[tuple]) -> List[MenuSpan]:
    """Verbatim (zero-diff) pool hits inside ``text``, non-overlapping.

    ``align_spans`` deliberately skips a window that already equals its
    candidate -- there is nothing to correct.  Basil still needs to know
    that a correctly-read entity spans a region boundary, so this is the
    zero-diff sibling: same greedy non-overlap resolution, similarity 1.0.
    """
    if not text or " " in text:
        return []
    found: List[MenuSpan] = []
    for entry in pool:
        candidate, cand_lang, _scope = _entry_parts(entry)
        if lang and cand_lang != lang:
            continue
        n = len(candidate)
        if n < 2 or len(text) < n:
            continue
        start = text.find(candidate)
        while start != -1:
            found.append(MenuSpan(start, start + n, candidate, 1.0, []))
            start = text.find(candidate, start + 1)
    found.sort(key=lambda s: (-(s.end - s.start), s.start))
    kept: List[MenuSpan] = []
    for span in found:
        if any(span.start < k.end and k.start < span.end for k in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: s.start)
    return kept


def _count_exact_known_names(text: str, lang: Optional[str],
                              exclude_spans: List[MenuSpan],
                              ground_truth_pool: Optional[List[tuple]] = None) -> int:
    """how many DISTINCT gazetteer names appear verbatim in `text`,
    outside the proposal spans -- the exact-match half of the
    corroboration count (the other half is spans the string/pixel tiers
    already applied). each candidate counts at most once, at its first
    non-overlapping occurrence."""
    count = 0
    taken = [(s.start, s.end) for s in exclude_spans]
    for entry in list(KNOWN_PLACES) + list(ground_truth_pool or []):
        candidate, cand_lang, _scope = _entry_parts(entry)
        if lang and cand_lang != lang:
            continue
        if len(candidate) < 2 or len(candidate) >= len(text):
            continue
        idx = text.find(candidate)
        while idx != -1:
            end = idx + len(candidate)
            if not any(idx < te and ts < end for ts, te in taken):
                count += 1
                taken.append((idx, end))
                break
            idx = text.find(candidate, idx + 1)
    return count


def _verify_span_pixels(asset: ImageLike, inst: InstText, span: MenuSpan,
                         lang: Optional[str], font_registry: Any) -> "dict[int, Optional[bool]]":
    """pixel verdicts for a span's diff positions via savor's shared
    glyph-swap machinery. wrapped fail-open: menu has always been pure
    string logic and must stay throw-proof now that a pixel tier (cv2/
    PIL/file IO) is reachable from it -- any failure just means "no
    pixel evidence", never a crashed detection. consult_menu_substring
    rejects spaced text, so span indices ARE the no-space glyph indices
    chew_swaps expects."""
    if asset is None:
        return {}
    try:
        from tofu.layers.savor import chew_swaps
        return chew_swaps(asset, inst, span.diffs, lang, font_registry)
    except Exception:
        return {}


def browse(instances: List[InstText], asset: Optional[ImageLike] = None,
           font_registry: Optional[Any] = None,
           ground_truth_pool: Optional[List[tuple]] = None) -> int:
    """the full menu pass, run once across every instance.

    two courses per instance, mutually exclusive:

    1. SUBSTRING (composite reads): if any gazetteer name aligns inside
       a strictly-longer text (consult_menu_substring), correct the
       aligned span(s) only. a long-enough, single-diff span
       (SUBSTRING_STRING_TIER_*) applies on string evidence alone --
       same precedent as the whole-string path below; a weaker span
       applies only when savor's pixel check confirms EVERY differing
       glyph (any position the pixels actively contradict discards the
       whole span; merely-inconclusive evidence records the proposal
       with applied=False for review, the established convention).
       when ANY substring proposal exists, the whole-string course is
       skipped -- rewriting an entire composite post to one of its
       names would destroy the other names on it.

    2. WHOLE-STRING: the original low-confidence fuzzy match against
       whole gazetteer entries, unchanged.

    `asset`/`font_registry` are optional and only feed the substring
    pixel tier; every existing caller that omits them keeps prior
    behavior exactly (weak spans simply stay unapplied).

    corrections are recorded on inst.ocr_correction using the same
    {applied, original_text, corrected_text, reason} shape Savor
    already writes, so review/audit stays in one place regardless of
    which layer made the call. returns the number of instances
    corrected (mirrors second_look()/savor.taste()'s return contract).
    """
    corrected = 0
    ground_truth_sources = {
        candidate: scope or "project"
        for candidate, _lang, scope in (
            _entry_parts(entry) for entry in (ground_truth_pool or [])
        )
    }
    for inst in instances:
        text = inst.text or ""
        if not text:
            continue
        lang = inst.detected_language or inst.language

        spans = consult_menu_substring(text, lang, ground_truth_pool)
        if spans:
            applied: List[MenuSpan] = []
            inconclusive: List[MenuSpan] = []
            notes: List[str] = []
            for span in spans:
                window = text[span.start:span.end]
                string_tier = (
                    len(span.candidate) >= SUBSTRING_STRING_TIER_MIN_LEN
                    and len(span.diffs) <= SUBSTRING_STRING_TIER_MAX_DIFFS
                )
                if string_tier:
                    applied.append(span)
                    notes.append(
                        f"[{span.start}:{span.end}] {window}->{span.candidate} "
                        f"({span.similarity:.2f} similarity, string evidence)"
                    )
                    continue
                verdicts = _verify_span_pixels(asset, inst, span, lang, font_registry)
                position_verdicts = [verdicts.get(pos) for pos, _, _ in span.diffs]
                if any(v is False for v in position_verdicts):
                    continue  # pixels actively contradict -- spit out, no record
                if all(v is True for v in position_verdicts):
                    applied.append(span)
                    notes.append(
                        f"[{span.start}:{span.end}] {window}->{span.candidate} "
                        f"({span.similarity:.2f} similarity, glyph-shape confirmed)"
                    )
                else:
                    inconclusive.append(span)

            # corroboration tier (see CORROBORATION_MIN_SIBLINGS): the
            # sibling count is computed ONCE from independently-confirmed
            # evidence, so promoted spans can't bootstrap each other
            if inconclusive:
                siblings = _count_exact_known_names(
                    text, lang, spans, ground_truth_pool
                ) + len(applied)
                if siblings >= CORROBORATION_MIN_SIBLINGS:
                    for span in list(inconclusive):
                        if len(span.diffs) != 1:
                            continue  # multi-diff stays a review item -- context can't carry that much
                        window = text[span.start:span.end]
                        applied.append(span)
                        inconclusive.remove(span)
                        notes.append(
                            f"[{span.start}:{span.end}] {window}->{span.candidate} "
                            f"({span.similarity:.2f} similarity, corroborated by {siblings} "
                            f"co-occurring known names on the same read)"
                        )

            if applied:
                original = text
                chars = list(text)
                for span in applied:  # equal-length windows: no index shifting
                    chars[span.start:span.end] = list(span.candidate)
                new_text = "".join(chars)
                reason = "known name(s) aligned inside composite read: " + "; ".join(notes)
                if inconclusive:
                    reason += "; span(s) " + ", ".join(
                        f"[{s.start}:{s.end}] {text[s.start:s.end]}->{s.candidate}"
                        for s in inconclusive
                    ) + " left unchanged, pixel evidence inconclusive"
                gt_terms = [s.candidate for s in applied if s.candidate in ground_truth_sources]
                gt_scope = next(
                    (ground_truth_sources[s.candidate] for s in applied
                     if s.candidate in ground_truth_sources), None
                )
                inst.ocr_correction = {
                    "applied": True,
                    "original_text": original,
                    "corrected_text": new_text,
                    "reason": reason,
                    "correction_resource": (
                        _ground_truth_identity(gt_scope, gt_terms)
                        if gt_scope else KNOWN_PLACES_RESOURCE.audit_identity()
                    ),
                }
                if gt_scope:
                    inst.source_override = {
                        "kind": "ground_truth",
                        "text": new_text,
                        "icon": "leaf",
                        "color": "emerald",
                        "resource": inst.ocr_correction["correction_resource"],
                    }
                inst.text = new_text
                corrected += 1
            elif inconclusive:
                chars = list(text)
                for span in inconclusive:
                    chars[span.start:span.end] = list(span.candidate)
                gt_terms = [s.candidate for s in inconclusive if s.candidate in ground_truth_sources]
                gt_scope = next(
                    (ground_truth_sources[s.candidate] for s in inconclusive
                     if s.candidate in ground_truth_sources), None
                )
                inst.ocr_correction = {
                    "applied": False,
                    "candidate_text": "".join(chars),
                    "reason": "possible known name(s) inside composite read ("
                              + "; ".join(
                                  f"[{s.start}:{s.end}] {text[s.start:s.end]}->{s.candidate}"
                                  f" ({s.similarity:.2f} similarity)"
                                  for s in inconclusive
                              )
                              + "), pixel evidence inconclusive",
                    "correction_resource": (
                        _ground_truth_identity(gt_scope, gt_terms)
                        if gt_scope else KNOWN_PLACES_RESOURCE.audit_identity()
                    ),
                }
            # a substring alignment is structural evidence of a composite
            # read -- never fall through to the whole-string rewrite,
            # which would collapse the whole post to a single name
            continue

        match = consult_menu(text, lang, inst.confidence, ground_truth_pool)
        if match is None or match.text == text:
            continue
        gt_scope = ground_truth_sources.get(match.text)
        inst.ocr_correction = {
            "applied": True,
            "original_text": text,
            "corrected_text": match.text,
            "reason": f"gazetteer match ({match.similarity:.2f} similarity) against a known place/establishment name",
            "correction_resource": (
                _ground_truth_identity(gt_scope, [match.text])
                if gt_scope else KNOWN_PLACES_RESOURCE.audit_identity()
            ),
        }
        if gt_scope:
            inst.source_override = {
                "kind": "ground_truth",
                "text": match.text,
                "icon": "leaf",
                "color": "emerald",
                "resource": inst.ocr_correction["correction_resource"],
            }
        inst.text = match.text
        corrected += 1
    return corrected
