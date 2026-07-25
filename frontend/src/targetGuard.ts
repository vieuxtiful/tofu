// 🍢 target-language guard for translation entry.
//
// This is a SUBMIT gate, not a keystroke filter. Typing is never
// interrupted, rewritten or deleted — the verdict only decides whether
// "arrange" and "Plate cubes" are available, and says why. The previous
// guard in RegionTable cleared the user's field on a mismatch, which lost
// real work to a heuristic.
//
// It is weighted rather than thresholded: every word contributes its own
// length, so one uncertain token cannot fail an otherwise correct phrase,
// and a phrase that is mostly wrong cannot pass on one right token.
//
// The Latin-in-CJK case is the one that matters most. Today's
// scriptMatchesLang() returns true for ANY Latin text regardless of
// target, so "Welcome" typed into a Chinese target passes unnoticed. The
// fix is not to ban Latin — romanised place names and brand names are
// legitimate — but to require that such a run be ATTESTED: present in
// Cicerone's own manifest, in a region the recognizer identified as
// Latin-script. japan-subs carries Hida-osaka, Hida-miyada and Nagisa as
// real `en` regions, so those pass into a zh-cn target while an arbitrary
// English word does not.

import { LANG_SCRIPT_MAP, LATIN_LANG_HINTS, detectScript, type Script } from "./detectLanguage";
import type { InstText } from "./api";

export interface GuardVerdict {
  ok: boolean;
  /** 0..1 weighted confidence that the entry is in the target language. */
  score: number;
  /** Human-readable explanation, shown on the disabled button. */
  reason: string;
  /** The words that scored zero, for highlighting in the halo. */
  offending: string[];
}

/** Words attested by Cicerone as Latin-script source text. */
export type AttestedSet = ReadonlySet<string>;

/** A phrase must clear this weighted score to be platable. Declared rather
 * than tuned per call so the number is auditable in one place. */
export const GUARD_PASS_SCORE = 0.75;

const WORD_RE = /[\p{L}\p{M}][\p{L}\p{M}‐-―'’-]*/gu;

/** Fold a word to its attestation key: case and hyphenation carry no
 * meaning for "did the source contain this name". */
export function attestKey(word: string): string {
  return word.toLowerCase().replace(/[‐-―'’\-\s]/g, "");
}

/** Build the attested set once per manifest. Only regions the recognizer
 * identified as Latin-script count — a romanisation the OCR read as
 * Japanese is not evidence that a Latin string belongs in the target. */
export function attestedFromManifest(instances: InstText[]): AttestedSet {
  const attested = new Set<string>();
  for (const inst of instances) {
    const text = inst.text ?? "";
    if (!text.trim()) continue;
    if (detectScript(text) !== "Latin") continue;
    attested.add(attestKey(text));
    for (const word of text.match(WORD_RE) ?? []) attested.add(attestKey(word));
  }
  return attested;
}

/** The scripts a TARGET field may contain, which is stricter than the
 * source-side map: the blanket Latin allowance on ja/ko exists so source
 * signage can carry romaji, and attestation replaces it here. */
function targetScripts(langCode: string): Script[] {
  const allowed = LANG_SCRIPT_MAP[langCode.toLowerCase()]
    ?? LANG_SCRIPT_MAP[langCode.toLowerCase().split("-")[0]];
  if (!allowed) return [];
  if (allowed.length > 1 && allowed.includes("Latin") && allowed[0] !== "Latin") {
    return allowed.filter((script) => script !== "Latin");
  }
  return allowed;
}

/** Weighted count of foreign-diacritic evidence in a Latin phrase.
 * Used as a proportional signal, not a hard reject: detectLatinLang()
 * returns on its first match and so reads "ö" as German over Swedish. */
function latinMismatch(text: string, langCode: string): number {
  const own = LATIN_LANG_HINTS[langCode] ?? [];
  const ownHits = own.filter((pattern) => pattern.test(text)).length;
  if (ownHits > 0) return 0;
  let foreign = 0;
  for (const [lang, patterns] of Object.entries(LATIN_LANG_HINTS)) {
    if (lang === langCode) continue;
    if (patterns.some((pattern) => pattern.test(text))) foreign += 1;
  }
  return foreign;
}

export function guardTargetText(
  text: string,
  langCode: string,
  attested: AttestedSet,
): GuardVerdict {
  const trimmed = (text ?? "").trim();
  if (!trimmed) {
    return { ok: false, score: 0, reason: "enter the target phrase", offending: [] };
  }
  const lang = (langCode ?? "").toLowerCase();
  const allowed = targetScripts(lang);
  if (allowed.length === 0) {
    // No declared script for this language: fail open rather than block on
    // ignorance, exactly as Basil's pairing verdict does.
    return { ok: true, score: 1, reason: `no script profile for '${lang || "?"}'`, offending: [] };
  }

  const words = trimmed.match(WORD_RE) ?? [];
  if (words.length === 0) {
    // Digits and punctuation only — nothing to judge, and plenty of real
    // signage is exactly that ("2024", "24h").
    return { ok: true, score: 1, reason: "no letters to check", offending: [] };
  }

  const latinTarget = allowed.includes("Latin");
  let weighted = 0;
  let total = 0;
  const offending: string[] = [];
  for (const word of words) {
    const weight = word.length;
    total += weight;
    const script = detectScript(word);
    if (script === "Unknown" || allowed.includes(script)) {
      weighted += weight;
      continue;
    }
    if (script === "Latin" && !latinTarget && attested.has(attestKey(word))) {
      // A romanised name the source itself carries: legitimate.
      weighted += weight;
      continue;
    }
    offending.push(word);
  }

  let score = total > 0 ? weighted / total : 1;
  let reason = "";
  if (offending.length > 0) {
    reason = latinTarget
      ? `not ${lang}: ${offending.slice(0, 3).join(", ")}`
      : `${lang} expects ${allowed.join("/")}; ${offending.slice(0, 3).join(", ")} is unattested in the source`;
  } else if (latinTarget) {
    // Both languages are Latin-script, so script alone proves nothing.
    // Weigh the diacritic evidence instead of rejecting on first hit.
    const foreign = latinMismatch(trimmed, lang);
    if (foreign > 0) {
      score = Math.max(0, score - Math.min(0.5, foreign * 0.25));
      reason = `spelling looks unlike ${lang}`;
    }
  }

  const ok = score >= GUARD_PASS_SCORE;
  return {
    ok,
    score: Math.round(score * 1000) / 1000,
    reason: ok ? reason || `reads as ${lang}` : reason || `does not read as ${lang}`,
    offending,
  };
}
