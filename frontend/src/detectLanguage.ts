import { getLanguageByCode } from "./languageData";

export type Script =
  | "Latin" | "Cyrillic" | "Greek" | "CJK" | "Hiragana" | "Katakana"
  | "Hangul" | "Arabic" | "Hebrew" | "Devanagari" | "Thai" | "Bengali"
  | "Gurmukhi" | "Gujarati" | "Tamil" | "Telugu" | "Kannada"
  | "Malayalam" | "Sinhala" | "Myanmar" | "Khmer" | "Lao"
  | "Armenian" | "Georgian" | "Ethiopic" | "Unknown";

interface Range { script: Script; re: RegExp }

const RANGES: Range[] = [
  { script: "Hiragana", re: /[\u3040-\u309F]/ },
  { script: "Katakana", re: /[\u30A0-\u30FF]/ },
  { script: "Hangul", re: /[\uAC00-\uD7AF\u1100-\u11FF]/ },
  { script: "CJK", re: /[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]/ },
  { script: "Arabic", re: /[\u0600-\u06FF\u0750-\u077F]/ },
  { script: "Hebrew", re: /[\u0590-\u05FF]/ },
  { script: "Devanagari", re: /[\u0900-\u097F]/ },
  { script: "Bengali", re: /[\u0980-\u09FF]/ },
  { script: "Gurmukhi", re: /[\u0A00-\u0A7F]/ },
  { script: "Gujarati", re: /[\u0A80-\u0AFF]/ },
  { script: "Tamil", re: /[\u0B80-\u0BFF]/ },
  { script: "Telugu", re: /[\u0C00-\u0C7F]/ },
  { script: "Kannada", re: /[\u0C80-\u0CFF]/ },
  { script: "Malayalam", re: /[\u0D00-\u0D7F]/ },
  { script: "Sinhala", re: /[\u0D80-\u0DFF]/ },
  { script: "Thai", re: /[\u0E00-\u0E7F]/ },
  { script: "Lao", re: /[\u0E80-\u0EFF]/ },
  { script: "Myanmar", re: /[\u1000-\u109F]/ },
  { script: "Khmer", re: /[\u1780-\u17FF]/ },
  { script: "Armenian", re: /[\u0530-\u058F]/ },
  { script: "Georgian", re: /[\u10A0-\u10FF]/ },
  { script: "Ethiopic", re: /[\u1200-\u137F]/ },
  { script: "Greek", re: /[\u0370-\u03FF]/ },
  { script: "Cyrillic", re: /[\u0400-\u04FF]/ },
  { script: "Latin", re: /[\u00C0-\u024F\u1E00-\u1EFFa-zA-Z]/ },
];

export function detectScript(text: string): Script {
  if (!text || !text.trim()) return "Unknown";
  const counts = new Map<Script, number>();
  for (const { script, re } of RANGES) {
    const matches = text.match(new RegExp(re.source, "g"));
    if (matches) counts.set(script, (counts.get(script) ?? 0) + matches.length);
  }
  if (counts.size === 0) return "Unknown";
  let best: Script = "Unknown";
  let bestCount = 0;
  for (const [s, c] of counts) {
    if (c > bestCount) { best = s; bestCount = c; }
  }
  return best;
}

const LANG_SCRIPT_MAP: Record<string, Script[]> = {
  "en": ["Latin"], "es": ["Latin"], "fr": ["Latin"], "de": ["Latin"],
  "it": ["Latin"], "pt": ["Latin"], "nl": ["Latin"], "sv": ["Latin"],
  "no": ["Latin"], "da": ["Latin"], "fi": ["Latin"], "is": ["Latin"],
  "pl": ["Latin"], "cs": ["Latin"], "sk": ["Latin"], "hu": ["Latin"],
  "ro": ["Latin"], "hr": ["Latin"], "sl": ["Latin"], "et": ["Latin"],
  "lv": ["Latin"], "lt": ["Latin"], "tr": ["Latin"], "vi": ["Latin"],
  "uz": ["Latin"], "az": ["Latin"],
  "bg": ["Cyrillic"], "sr": ["Cyrillic"], "sr-cyrl": ["Cyrillic"],
  "ru": ["Cyrillic"], "uk": ["Cyrillic"], "mn": ["Cyrillic"], "kk": ["Cyrillic"],
  "sr-latn": ["Latin"],
  "el": ["Greek"],
  "ja": ["Hiragana", "Katakana", "CJK", "Latin"],
  "ko": ["Hangul", "Latin"],
  "zh-cn": ["CJK"], "zh-sg": ["CJK"], "zh-tw": ["CJK"], "zh-hk": ["CJK"], "zh-mo": ["CJK"],
  "ar": ["Arabic"], "fa": ["Arabic"],
  "he": ["Hebrew"],
  "hi": ["Devanagari"], "mr": ["Devanagari"],
  "bn": ["Bengali"],
  "pa": ["Gurmukhi"],
  "gu": ["Gujarati"],
  "ta": ["Tamil"],
  "te": ["Telugu"],
  "kn": ["Kannada"],
  "ml": ["Malayalam"],
  "si": ["Sinhala"],
  "th": ["Thai"],
  "my": ["Myanmar"],
  "km": ["Khmer"],
  "lo": ["Lao"],
  "hy": ["Armenian"],
  "ka": ["Georgian"],
  "am": ["Ethiopic"], "ti": ["Ethiopic"],
};

export function scriptMatchesLang(script: Script, langCode: string): boolean {
  if (script === "Unknown" || script === "Latin") return true;
  const allowed = LANG_SCRIPT_MAP[langCode];
  if (!allowed) return true;
  return allowed.includes(script);
}

export function textMatchesTargetLang(text: string, langCode: string): boolean {
  const script = detectScript(text);
  if (script === "Unknown") return true;
  return scriptMatchesLang(script, langCode);
}

const LATIN_LANG_HINTS: Record<string, RegExp[]> = {
  "es": [/ñ/i, /¿/, /¡/],
  "de": [/ä/i, /ö/i, /ü/i, /ß/i],
  "fr": [/ç/i, /œ/i, /æ/i],
  "sv": [/å/i, /ä/i, /ö/i],
  "no": [/å/i, /æ/i, /ø/i],
  "da": [/å/i, /æ/i, /ø/i],
  "is": [/þ/i, /ð/i],
  "cs": [/ř/i, /š/i, /č/i, /ž/i, /ý/i, /á/i, /é/i, /í/i, /ó/i, /ú/i, /ů/i],
  "pl": [/ł/i, /ś/i, /ż/i, /ź/i, /ć/i, /ń/i, /ą/i, /ę/i],
  "hu": [/ő/i, /ű/i],
  "ro": [/ă/i, /â/i, /î/i, /ș/i, /ț/i],
  "tr": [/ı/i, /ş/i, /ğ/i, /ç/i, /ö/i, /ü/i],
  "pt": [/ã/i, /õ/i, /ç/i],
  "vi": [/[\u0103\u00E2\u0111\u00EA\u00F4\u01A1\u01B0\u00E1\u00E0\u1EA3\u00E3\u1EA1\u1EA5\u1EA7\u1EA9\u1EAB\u1EAD\u00E9\u00E8\u1EBB\u1EBD\u1EB9\u00ED\u00EC\u1EC9\u1ECB\u00F3\u00F2\u1ECF\u00F5\u1ECD\u1EDB\u1EDD\u1EDF\u1EE1\u1EE3\u00FA\u00F9\u1EE7\u0169\u1EE5\u00FD\u1EF3\u1EF7\u1EF9\u1EF5]/],
};

export function detectLatinLang(text: string): string | null {
  for (const [lang, patterns] of Object.entries(LATIN_LANG_HINTS)) {
    for (const p of patterns) {
      if (p.test(text)) return lang;
    }
  }
  return null;
}

export function textLangMatchesTarget(text: string, langCode: string): boolean {
  if (!text || !text.trim()) return true;
  const script = detectScript(text);
  if (script === "Unknown") return true;
  if (!scriptMatchesLang(script, langCode)) return false;
  if (script === "Latin") {
    const lang = getLanguageByCode(langCode);
    if (lang && lang.script === "Latin") {
      const detected = detectLatinLang(text);
      if (detected && detected !== langCode) {
        const detectedLang = getLanguageByCode(detected);
        if (detectedLang && detectedLang.script === "Latin") return false;
      }
    }
  }
  return true;
}
