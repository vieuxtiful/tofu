"""Optional local language and diacritic model adapters.

These adapters never download weights or make network calls.  They expose a
small, versioned evidence contract so Cicerone can remain conservative when a
host has not provisioned a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Sequence


@dataclass(frozen=True)
class LanguageEvidence:
    language: Optional[str]
    confidence: float
    provider: str
    model_version: Optional[str]
    reason: str


class LanguageModelProvider(Protocol):
    def identify(self, texts: Sequence[str]) -> LanguageEvidence: ...
    def status(self) -> Dict[str, object]: ...


_FASTTEXT_TO_TOFU = {
    "en": "en", "fr": "fr", "es": "es", "pl": "pl", "ro": "ro",
    "vi": "vi", "de": "de", "it": "it", "pt": "pt", "nl": "nl",
    "ru": "ru", "uk": "uk", "ja": "ja", "ko": "ko", "th": "th",
    "zh": "zh-cn",
}


class FastTextLanguageProvider:
    """Side-loaded fastText LID model, usually ``lid.176.ftz``.

    The provider deliberately returns unknown on absent/invalid models and on
    short or ambiguous evidence.  A false English label is worse than no
    label because it constrains later OCR passes to the wrong charset.
    """
    provider_id = "fasttext-lid"

    def __init__(self, model_path: Optional[str] = None, min_confidence: float = .72):
        self.model_path = Path(model_path or os.environ.get("TOFU_FASTTEXT_LID", ""))
        self.min_confidence = min_confidence
        self._model = None
        self._failed: Optional[str] = None

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        if not self.model_path or not self.model_path.is_file():
            self._failed = "no readable model configured in TOFU_FASTTEXT_LID"
            return None
        try:
            import fasttext  # type: ignore
            self._model = fasttext.load_model(str(self.model_path))
        except Exception as exc:  # optional native extension/model errors
            self._failed = f"model load failed: {type(exc).__name__}"
        return self._model

    def identify(self, texts: Sequence[str]) -> LanguageEvidence:
        text = " ".join(t.strip() for t in texts if t and t.strip())
        if len(text) < 3:
            return LanguageEvidence(None, 0.0, self.provider_id, None, "insufficient text")
        model = self._load()
        if model is None:
            return LanguageEvidence(None, 0.0, self.provider_id, None, self._failed or "unavailable")
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        raw = labels[0].removeprefix("__label__") if labels else ""
        confidence = float(probs[0]) if probs else 0.0
        lang = _FASTTEXT_TO_TOFU.get(raw)
        if lang is None or confidence < self.min_confidence:
            return LanguageEvidence(None, confidence, self.provider_id, self.model_path.name, "ambiguous prediction")
        return LanguageEvidence(lang, confidence, self.provider_id, self.model_path.name, "model prediction")

    def status(self) -> Dict[str, object]:
        ready = bool(self.model_path and self.model_path.is_file())
        return {
            "id": self.provider_id, "available": ready, "ready": ready,
            "version": self.model_path.name if ready else None,
            "reason": None if ready else "no readable model configured in TOFU_FASTTEXT_LID",
        }


class DiacriticRestorationProvider:
    """Versioned optional ONNX seam for conservative accent restoration.

    An artifact is intentionally not bundled.  Until a compatible local model
    is configured, callers get no candidate rather than a fabricated accent.
    The public contract lets a later artifact be enabled without changing OCR
    provenance or review behavior.
    """
    provider_id = "onnx-diacritic-restoration"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or os.environ.get("TOFU_DIACRITIC_MODEL", ""))

    def status(self) -> Dict[str, object]:
        ready = bool(self.model_path and self.model_path.is_file())
        return {
            "id": self.provider_id, "available": ready, "ready": ready,
            "version": self.model_path.name if ready else None,
            "reason": None if ready else "no readable model configured in TOFU_DIACRITIC_MODEL",
        }

    def candidates(self, text: str, language: Optional[str]) -> Iterable[str]:
        # A model artifact defines the tokenizer/output vocabulary.  Returning
        # no candidate is the only safe fallback; Cicerone's existing
        # dictionary/OCR arbitration remains responsible for review hints.
        return ()


## Scripts whose text is scored CHARACTER by character rather than word by
## word. CJK is written without word spaces, and its OCR errors are
## character substitutions rather than boundary errors, so a character
## n-gram is both simpler (no segmenter to depend on) and better matched
## to the failure it has to rank. Mixed kana/kanji/latin falls out of the
## same treatment for free -- every codepoint is a token.
## Scripts written one character per token rather than one word per token,
## which decides which model family scores a reading.
##
## Two vocabularies, deliberately, because two exist in this codebase and
## the mismatch was silent: ISO 15924 codes are what the manifest carries,
## while cicerone's ScriptDetector emits lowercase names ("han",
## "japanese", "hangul"). ocr_arbitration hands the DETECTOR's value
## straight to score(), so with only the ISO half here every CJK reading
## was quietly scored against the Latin model -- a wrong answer that looks
## exactly like a working one.
_CHARACTER_SCRIPTS = {
    "Hani", "Hang", "Hira", "Kana", "Jpan", "Hans", "Hant",
    "han", "hangul", "hiragana", "katakana", "japanese", "korean", "chinese",
}

## Punctuation is split into its own token rather than glued to the word
## beside it, because ranking punctuation is part of the job: decolonisons'
## 'nos rues !' comes back as 'nos rues 4', and a model that has only ever
## seen 'rues!' as one token cannot say which of those is likelier.
_LM_PUNCT = re.compile(r"([^\w\s]|_)", re.UNICODE)


def tokenize_for_lm(text: str, family: str) -> list:
    """Tokenize exactly the same way at training time and at scoring time.

    Exported and used by BOTH this provider and scripts/build_kenlm_models.py.
    Train/score skew is the quiet way an n-gram model underperforms: a model
    trained on 'rue de la paix' scores 'RUE DE LA PAIX' as unseen, and the
    signal degrades to noise without ever failing loudly.

    Latin is case-folded because signage is routinely set in caps and the
    model's job is the plausibility of the word sequence, not its casing --
    which savor's case course decides from pixels anyway. CJK is one token
    per codepoint, so no segmenter, and mixed kana/kanji/latin needs no
    special handling.
    """
    if family == "cjk":
        return [ch for ch in (text or "").strip() if not ch.isspace()]
    spaced = _LM_PUNCT.sub(r" \1 ", (text or "").casefold())
    return spaced.split()


class KenLMScoringProvider:
    """Side-loaded KenLM n-gram model for OCR candidate rescoring.

    A CRNN+CTC recognizer decodes character by character with no notion of
    whether the reading it produced is a plausible string -- savor.py's
    module note says exactly this, and it is why the glyph-confusion
    courses have to reason from pixels alone. An n-gram model is the
    cheapest thing that supplies the missing signal: it cannot read the
    image, but it can say that MAIN STREET is a likelier string than MAIN
    STBEET.

    One engine, one model file per script family, routed by the script
    already detected upstream. A single model spanning Latin and CJK is
    deliberately NOT supported: the vocabularies are disjoint, and mixing
    them dilutes exactly the n-gram statistics the ranking depends on.

    Like every other provider here it never downloads anything and never
    makes a network call. Absent ``TOFU_KENLM_DIR``, an unreadable model,
    or a missing ``kenlm`` package all resolve to "no score", and callers
    treat that as one signal being unavailable rather than as evidence.
    """
    provider_id = "kenlm-ngram"

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or os.environ.get("TOFU_KENLM_DIR", ""))
        self._models: Dict[str, object] = {}
        self._failed: Dict[str, str] = {}

    @staticmethod
    def _family(script: Optional[str]) -> str:
        return "cjk" if script in _CHARACTER_SCRIPTS else "latin"

    def _load(self, family: str):
        if family in self._models:
            return self._models[family]
        if family in self._failed:
            return None
        path = self.model_dir / f"{family}.klm"
        if not self.model_dir or not path.is_file():
            self._failed[family] = f"no readable model at {path}"
            return None
        try:
            import kenlm  # type: ignore
            model = kenlm.Model(str(path))
        except Exception as exc:  # optional native extension/model errors
            self._failed[family] = f"model load failed: {type(exc).__name__}"
            return None
        self._models[family] = model
        return model

    def score(self, text: str, script: Optional[str] = None) -> Optional[float]:
        """Mean log10 probability per token, or None when unavailable.

        Normalized by token count so a long correct line is not ranked
        below a short one purely for having more tokens to be charged for.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        family = self._family(script)
        model = self._load(family)
        if model is None:
            return None
        tokens = tokenize_for_lm(cleaned, family)
        if not tokens:
            return None
        try:
            total = model.score(" ".join(tokens), bos=True, eos=True)
        except Exception:
            return None
        return float(total) / len(tokens)

    def status(self) -> Dict[str, object]:
        families = {}
        for family in ("latin", "cjk"):
            path = self.model_dir / f"{family}.klm" if self.model_dir else None
            families[family] = bool(path and path.is_file())
        ready = any(families.values())
        return {
            "id": self.provider_id, "available": ready, "ready": ready,
            "version": str(self.model_dir) if ready else None,
            "families": families,
            "reason": None if ready else "no readable model configured in TOFU_KENLM_DIR",
        }


class KneserNeyScoringProvider:
    """Interpolated modified Kneser-Ney n-gram, with no native toolchain.

    Same job as KenLMScoringProvider and the same contract; the difference
    is what it costs to obtain one. KenLM's ``lmplz`` and ``build_binary``
    are C++ programs -- ``pip install kenlm`` ships the scoring module
    only -- so having the signal at all meant a compiler and a multi-GB
    Wikipedia dump. This trains from a local text file of tens of MB with
    the interpreter already in hand.

    Modified Kneser-Ney (Kneser & Ney 1995; Chen & Goodman 1999 for the
    three discounts) because the thing it fixes is precisely this
    application: a plain backoff model ranks a string by how often its
    words appeared, so a rare-but-real word loses to a common one, and
    ranking OCR candidates is exactly where that goes wrong. Kneser-Ney
    ranks the lower order by CONTINUATION count -- how many distinct
    contexts a token completes -- so a word that only ever appears in one
    phrase stops being treated as generally likely.

    Reads the artifact produced by scripts/build_ngram_models.py, which
    shares tokenize_for_lm with this class so train and score cannot skew.
    Absent an artifact every score is None, exactly as with KenLM, and the
    caller renormalizes the remaining signals.
    """
    provider_id = "kneser-ney-ngram"
    ## Out-of-vocabulary floor, in log10. Roughly one in ten million: low
    ## enough that an invented word loses decisively, finite so a single
    ## unknown token cannot make two readings incomparable.
    ##
    ## It is a FLOOR, not the OOV score. As a flat penalty it was a cliff,
    ## and the cliff fell in the wrong place: no corpus of tens of MB
    ## contains every proper noun on a street sign, so a real word the
    ## corpus happens to lack scored identically to a misread of it.
    ## RÉPUBLIQUE and RÉPUBUQUE, ORFEVRES and ORFEVBES -- exactly the pairs
    ## this signal exists to separate -- were the pairs it could not.
    ##
    ## When the artifact carries a character model, an OOV token is scored
    ## by how well-formed it is instead, and this value bounds that score
    ## from below. See _oov_logprob.
    OOV_LOGPROB = -7.0

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or os.environ.get("TOFU_NGRAM_DIR", ""))
        self._models: Dict[str, object] = {}
        self._failed: Dict[str, str] = {}

    @staticmethod
    def _family(script: Optional[str]) -> str:
        return "cjk" if script in _CHARACTER_SCRIPTS else "latin"

    def _load(self, family: str):
        if family in self._models:
            return self._models[family]
        if family in self._failed:
            return None
        path = self.model_dir / f"{family}.ngram.json.gz"
        if not str(self.model_dir) or not path.is_file():
            self._failed[family] = f"no readable model at {path}"
            return None
        try:
            import gzip
            import json as _json
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                model = _json.load(handle)
            if not isinstance(model, dict) or "unigram" not in model:
                raise ValueError("not an n-gram artifact")
        except Exception as exc:
            self._failed[family] = f"model load failed: {type(exc).__name__}"
            return None
        ## Computed once here rather than per OOV token: it is a min over the
        ## whole vocabulary, and _oov_logprob runs per token of every
        ## candidate reading of every region.
        import math

        values = (model.get("unigram") or {}).values()
        model["_oov_ceiling"] = (
            min(math.log10(max(min(values), 1e-12)), 0.0) if values else 0.0
        )
        self._models[family] = model
        return model

    def score(self, text: str, script: Optional[str] = None) -> Optional[float]:
        """Mean log10 probability per token, or None when unavailable.

        Normalized by token count for the same reason KenLM's is: a long
        correct line must not rank below a short one purely for having
        more tokens to be charged for.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        family = self._family(script)
        model = self._load(family)
        if model is None:
            return None
        tokens = tokenize_for_lm(cleaned, family)
        if not tokens:
            return None

        unigram = model["unigram"]
        bigram = model.get("bigram") or {}
        discount = float(model.get("discount", 0.75))
        charmodel = model.get("charmodel") or None
        ceiling = float(model.get("_oov_ceiling", 0.0))
        total = 0.0
        previous = "<s>"
        for token in tokens:
            total += self._token_logprob(
                token, previous, unigram, bigram, discount, charmodel, ceiling
            )
            previous = token
        return total / len(tokens)

    ## How many standard deviations of the corpus's own per-character score
    ## distribution span the full climb from OOV_LOGPROB to the ceiling. Two
    ## either side: at -2 sigma a string is worse-formed than all but a few
    ## percent of real tokens and gets no credit at all; at +2 sigma it is as
    ## ordinary-looking as text gets and earns the whole headroom.
    _OOV_SIGMA_SPAN = 4.0

    def _oov_logprob(self, token, ceiling, charmodel) -> float:
        """How plausible is a word the corpus never contained?

        The flat floor this replaces treated every unseen token alike, which
        is the sparsity a purely symbolic model cannot escape: it knows only
        whether it has SEEN a string, and a vocabulary built from tens of MB
        has not seen most of the proper nouns that appear on signage. A
        character model asks the different question -- whether this is a
        sequence of letters the language FORMS -- and that question has a
        graded answer for strings the word model can only call unknown.

        The two models do not speak the same units, and conflating them is
        the trap here: the word model returns a log-probability PER TOKEN
        while a character model returns one per character, so using the
        character score directly makes long words look implausible and short
        ones look certain, in proportion to nothing. What transfers between
        them is not the score but the token's POSITION in the corpus's own
        distribution of character scores -- recorded at build time as a mean
        and a spread -- and that position is what selects a point in the
        interval this token is allowed to occupy.

        The interval is bounded at both ends. Below by OOV_LOGPROB, so a
        well-formed nonsense word still cannot climb far. Above by the
        rarest continuation probability the corpus attests, so the best an
        unseen string can do is look as likely as the least likely word we
        have actually seen -- appearing in a corpus is evidence of a kind
        that being spellable is not.

        Stated precisely, because the loose version of it is wrong: the
        ceiling is the minimum UNIGRAM term, and an attested token is scored
        through the bigram path, which can fall below its own unigram term
        when its context is seen but it is rare within it. So this bounds an
        OOV token against the vocabulary's floor, not against every attested
        token in every context. The guarantee is that an unknown word cannot
        run away with the ranking, not that it always loses.
        """
        import math

        if not charmodel:
            return self.OOV_LOGPROB

        counts = charmodel.get("counts") or {}
        order = int(charmodel.get("order", 3))
        boundary = charmodel.get("boundary", "\x02")
        alphabet = max(int(charmodel.get("alphabet_size", 0)), 1)
        mean = charmodel.get("mean_logprob_per_char")
        spread = charmodel.get("spread_logprob_per_char")
        if mean is None or not spread:
            # An artifact built before the reference existed. Its counts
            # cannot be placed on a scale, so decline to grade rather than
            # invent one.
            return self.OOV_LOGPROB

        padded = boundary * (order - 1) + token + boundary
        logprob = 0.0
        observed = 0
        for i in range(order - 1, len(padded)):
            context, nxt = padded[i - order + 1:i], padded[i]
            entry = counts.get(context)
            if entry is None:
                # An unseen context is itself evidence against the string;
                # add-one over the alphabet is the same estimate the build
                # applies, with a count of zero.
                logprob += math.log10(1.0 / (alphabet + 1))
            else:
                seen = entry["counts"].get(nxt, 0)
                logprob += math.log10((seen + 1) / (entry["total"] + alphabet))
            observed += 1
        if not observed:
            return self.OOV_LOGPROB

        per_char = logprob / observed
        z = (per_char - float(mean)) / float(spread)
        quality = min(max(0.5 + z / self._OOV_SIGMA_SPAN, 0.0), 1.0)
        headroom = max(0.0, ceiling - self.OOV_LOGPROB)
        return self.OOV_LOGPROB + headroom * quality

    def _token_logprob(self, token, previous, unigram, bigram, discount,
                       charmodel=None, ceiling: float = 0.0) -> float:
        """Bigram probability interpolated with the continuation unigram."""
        import math

        lower = unigram.get(token)
        if lower is None:
            return self._oov_logprob(token, ceiling, charmodel)
        context = bigram.get(previous)
        if not context:
            return math.log10(max(lower, 1e-12))
        counts, distinct, total = context["counts"], context["distinct"], context["total"]
        seen = counts.get(token, 0)
        # discounted bigram mass, plus the mass reserved for backoff
        higher = max(seen - discount, 0.0) / total
        backoff = (discount * distinct / total) * lower
        return math.log10(max(higher + backoff, 1e-12))

    def status(self) -> Dict[str, object]:
        families = {}
        for family in ("latin", "cjk"):
            path = (self.model_dir / f"{family}.ngram.json.gz"
                    if str(self.model_dir) else None)
            families[family] = bool(path and path.is_file())
        ready = any(families.values())
        return {
            "id": self.provider_id, "available": ready, "ready": ready,
            "version": str(self.model_dir) if ready else None,
            "families": families,
            "reason": None if ready else "no readable model configured in TOFU_NGRAM_DIR",
        }


def _pick_default_scorer():
    """Prefer whichever artifact is actually installed.

    Both providers answer the same question and neither needs the other.
    KenLM keeps precedence when its model is present, because a host that
    went to the trouble of building one should get it.
    """
    kenlm_provider = KenLMScoringProvider()
    if kenlm_provider.status().get("ready"):
        return kenlm_provider
    return KneserNeyScoringProvider()


_default_provider: LanguageModelProvider = FastTextLanguageProvider()
_default_scorer = _pick_default_scorer()


def get_scoring_provider() -> KenLMScoringProvider:
    return _default_scorer


def set_scoring_provider(provider: KenLMScoringProvider) -> None:
    global _default_scorer
    _default_scorer = provider


def get_language_provider() -> LanguageModelProvider:
    return _default_provider


def set_language_provider(provider: LanguageModelProvider) -> None:
    global _default_provider
    _default_provider = provider

