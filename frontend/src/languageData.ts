export interface Language {
  code: string;
  name: string;
  nativeName: string;
  script: string;
  rtl: boolean;
  flag: string;
  region: string;
  popular?: boolean;
}

// legacy bare-code → canonical full locale tag.  projects created before
// the locale-tag refactor stored target_lang as "en", "es", etc.  the
// wizard now emits "en-GB", "es-ES", … so lookups normalise first.
const LEGACY_CODES: Record<string, string> = {
  "en": "en-GB", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
  "it": "it-IT", "pt": "pt-PT", "nl": "nl-NL", "sv": "sv-SE",
  "no": "no-NO", "da": "da-DK", "fi": "fi-FI", "is": "is-IS",
  "pl": "pl-PL", "cs": "cs-CZ", "sk": "sk-SK", "hu": "hu-HU",
  "ro": "ro-RO", "bg": "bg-BG", "sr": "sr-RS",
  "sr-latn": "sr-Latn-RS", "sr-cyrl": "sr-Cyrl-RS",
  "hr": "hr-HR", "sl": "sl-SI", "et": "et-EE", "lv": "lv-LV",
  "lt": "lt-LT", "el": "el-GR", "ru": "ru-RU", "uk": "uk-UA",
  "tr": "tr-TR",
  "ja": "ja-JP", "ko": "ko-KR",
  "zh-cn": "zh-CN", "zh-sg": "zh-SG", "zh-tw": "zh-TW",
  "zh-hk": "zh-HK", "zh-mo": "zh-MO", "mn": "mn-MN",
  "hi": "hi-IN", "bn": "bn-BD", "pa": "pa-IN", "gu": "gu-IN",
  "mr": "mr-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
  "ml": "ml-IN", "si": "si-LK", "am": "am-ET", "ti": "ti-ER",
  "vi": "vi-VN", "th": "th-TH", "my": "my-MM", "km": "km-KH",
  "lo": "lo-LA",
  "ar": "ar-SA", "fa": "fa-IR", "he": "he-IL",
  "hy": "hy-AM", "ka": "ka-GE", "kk": "kk-KZ",
  "uz": "uz-UZ", "az": "az-AZ",
};

export const LANGUAGES: Language[] = [
  // Europe
  { code: "en-GB", name: "English (UK)", nativeName: "English (UK)", script: "Latin", rtl: false, flag: "🇬🇧", region: "Europe", popular: true },
  { code: "es-ES", name: "Spanish (Spain)", nativeName: "Español (España)", script: "Latin", rtl: false, flag: "🇪🇸", region: "Europe", popular: true },
  { code: "fr-FR", name: "French (France)", nativeName: "Français (France)", script: "Latin", rtl: false, flag: "🇫🇷", region: "Europe", popular: true },
  { code: "de-DE", name: "German (Germany)", nativeName: "Deutsch (Deutschland)", script: "Latin", rtl: false, flag: "🇩🇪", region: "Europe", popular: true },
  { code: "it-IT", name: "Italian (Italy)", nativeName: "Italiano (Italia)", script: "Latin", rtl: false, flag: "🇮🇹", region: "Europe" },
  { code: "pt-PT", name: "Portuguese (Portugal)", nativeName: "Português (Portugal)", script: "Latin", rtl: false, flag: "🇵🇹", region: "Europe" },
  { code: "nl-NL", name: "Dutch (Netherlands)", nativeName: "Nederlands (Nederland)", script: "Latin", rtl: false, flag: "🇳🇱", region: "Europe" },
  { code: "sv-SE", name: "Swedish (Sweden)", nativeName: "Svenska (Sverige)", script: "Latin", rtl: false, flag: "🇸🇪", region: "Europe" },
  { code: "no-NO", name: "Norwegian (Norway)", nativeName: "Norsk (Norge)", script: "Latin", rtl: false, flag: "🇳🇴", region: "Europe" },
  { code: "da-DK", name: "Danish (Denmark)", nativeName: "Dansk (Danmark)", script: "Latin", rtl: false, flag: "🇩🇰", region: "Europe" },
  { code: "fi-FI", name: "Finnish (Finland)", nativeName: "Suomi (Suomi)", script: "Latin", rtl: false, flag: "🇫🇮", region: "Europe" },
  { code: "is-IS", name: "Icelandic (Iceland)", nativeName: "Íslenska (Ísland)", script: "Latin", rtl: false, flag: "🇮🇸", region: "Europe" },
  { code: "pl-PL", name: "Polish (Poland)", nativeName: "Polski (Polska)", script: "Latin", rtl: false, flag: "🇵🇱", region: "Europe" },
  { code: "cs-CZ", name: "Czech (Czechia)", nativeName: "Čeština (Česko)", script: "Latin", rtl: false, flag: "🇨🇿", region: "Europe" },
  { code: "sk-SK", name: "Slovak (Slovakia)", nativeName: "Slovenčina (Slovensko)", script: "Latin", rtl: false, flag: "🇸🇰", region: "Europe" },
  { code: "hu-HU", name: "Hungarian (Hungary)", nativeName: "Magyar (Magyarország)", script: "Latin", rtl: false, flag: "🇭🇺", region: "Europe" },
  { code: "ro-RO", name: "Romanian (Romania)", nativeName: "Română (România)", script: "Latin", rtl: false, flag: "🇷🇴", region: "Europe" },
  { code: "bg-BG", name: "Bulgarian (Bulgaria)", nativeName: "Български (България)", script: "Cyrillic", rtl: false, flag: "🇧🇬", region: "Europe" },
  { code: "sr-RS", name: "Serbian (Serbia)", nativeName: "Српски (Србија)", script: "Cyrillic", rtl: false, flag: "🇷🇸", region: "Europe" },
  { code: "sr-Latn-RS", name: "Serbian (Latin)", nativeName: "Srpski (Latinica)", script: "Latin", rtl: false, flag: "🇷🇸", region: "Europe" },
  { code: "sr-Cyrl-RS", name: "Serbian (Cyrillic)", nativeName: "Српски (Ћирилица)", script: "Cyrillic", rtl: false, flag: "🇷🇸", region: "Europe" },
  { code: "hr-HR", name: "Croatian (Croatia)", nativeName: "Hrvatski (Hrvatska)", script: "Latin", rtl: false, flag: "🇭🇷", region: "Europe" },
  { code: "sl-SI", name: "Slovenian (Slovenia)", nativeName: "Slovenščina (Slovenija)", script: "Latin", rtl: false, flag: "🇸🇮", region: "Europe" },
  { code: "et-EE", name: "Estonian (Estonia)", nativeName: "Eesti (Eesti)", script: "Latin", rtl: false, flag: "🇪🇪", region: "Europe" },
  { code: "lv-LV", name: "Latvian (Latvia)", nativeName: "Latviešu (Latvija)", script: "Latin", rtl: false, flag: "🇱🇻", region: "Europe" },
  { code: "lt-LT", name: "Lithuanian (Lithuania)", nativeName: "Lietuvių (Lietuva)", script: "Latin", rtl: false, flag: "🇱🇹", region: "Europe" },
  { code: "el-GR", name: "Greek (Greece)", nativeName: "Ελληνικά (Ελλάδα)", script: "Greek", rtl: false, flag: "🇬🇷", region: "Europe" },
  { code: "ru-RU", name: "Russian (Russia)", nativeName: "Русский (Россия)", script: "Cyrillic", rtl: false, flag: "🇷🇺", region: "Europe", popular: true },
  { code: "uk-UA", name: "Ukrainian (Ukraine)", nativeName: "Українська (Україна)", script: "Cyrillic", rtl: false, flag: "🇺🇦", region: "Europe" },
  { code: "tr-TR", name: "Turkish (Türkiye)", nativeName: "Türkçe (Türkiye)", script: "Latin", rtl: false, flag: "🇹🇷", region: "Europe" },

  // Americas
  { code: "en-US", name: "English (US)", nativeName: "English (US)", script: "Latin", rtl: false, flag: "🇺🇸", region: "Americas", popular: true },
  { code: "es-MX", name: "Spanish (Mexico)", nativeName: "Español (México)", script: "Latin", rtl: false, flag: "🇲🇽", region: "Americas", popular: true },
  { code: "es-US", name: "Spanish (US)", nativeName: "Español (US)", script: "Latin", rtl: false, flag: "🇺🇸", region: "Americas" },
  { code: "pt-BR", name: "Portuguese (Brazil)", nativeName: "Português (Brasil)", script: "Latin", rtl: false, flag: "🇧🇷", region: "Americas", popular: true },
  { code: "fr-CA", name: "French (Canada)", nativeName: "Français (Canada)", script: "Latin", rtl: false, flag: "🇨🇦", region: "Americas" },

  // East Asia
  { code: "ja-JP", name: "Japanese (Japan)", nativeName: "日本語 (日本)", script: "Kanji/Hiragana", rtl: false, flag: "🇯🇵", region: "East Asia", popular: true },
  { code: "ko-KR", name: "Korean (Korea)", nativeName: "한국어 (한국)", script: "Hangul", rtl: false, flag: "🇰🇷", region: "East Asia", popular: true },
  { code: "zh-CN", name: "Chinese (Simplified)", nativeName: "简体中文 (中国)", script: "Han (Simplified)", rtl: false, flag: "🇨🇳", region: "East Asia", popular: true },
  { code: "zh-SG", name: "Chinese (Singapore)", nativeName: "简体中文 (新加坡)", script: "Han (Simplified)", rtl: false, flag: "🇸🇬", region: "East Asia" },
  { code: "zh-TW", name: "Chinese (Traditional)", nativeName: "繁體中文 (台灣)", script: "Han (Traditional)", rtl: false, flag: "🇹🇼", region: "East Asia" },
  { code: "zh-HK", name: "Chinese (Hong Kong)", nativeName: "繁體中文 (香港)", script: "Han (Traditional)", rtl: false, flag: "🇭🇰", region: "East Asia" },
  { code: "zh-MO", name: "Chinese (Macau)", nativeName: "繁體中文 (澳門)", script: "Han (Traditional)", rtl: false, flag: "🇲🇴", region: "East Asia" },
  { code: "mn-MN", name: "Mongolian (Mongolia)", nativeName: "Монгол (Монгол)", script: "Cyrillic", rtl: false, flag: "🇲🇳", region: "East Asia" },

  // South Asia
  { code: "hi-IN", name: "Hindi (India)", nativeName: "हिन्दी (भारत)", script: "Devanagari", rtl: false, flag: "🇮🇳", region: "South Asia", popular: true },
  { code: "bn-BD", name: "Bengali (Bangladesh)", nativeName: "বাংলা (বাংলাদেশ)", script: "Bengali", rtl: false, flag: "🇧🇩", region: "South Asia" },
  { code: "pa-IN", name: "Punjabi (India)", nativeName: "ਪੰਜਾਬੀ (ਭਾਰਤ)", script: "Gurmukhi", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "gu-IN", name: "Gujarati (India)", nativeName: "ગુજરાતી (ભારત)", script: "Gujarati", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "mr-IN", name: "Marathi (India)", nativeName: "मराठी (भारत)", script: "Devanagari", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "ta-IN", name: "Tamil (India)", nativeName: "தமிழ் (இந்தியா)", script: "Tamil", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "te-IN", name: "Telugu (India)", nativeName: "తెలుగు (భారతదేశం)", script: "Telugu", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "kn-IN", name: "Kannada (India)", nativeName: "ಕನ್ನಡ (ಭಾರತ)", script: "Kannada", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "ml-IN", name: "Malayalam (India)", nativeName: "മലയാളം (ഇന്ത്യ)", script: "Malayalam", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "si-LK", name: "Sinhala (Sri Lanka)", nativeName: "සිංහල (ශ්රී ලංකාව)", script: "Sinhala", rtl: false, flag: "🇱🇰", region: "South Asia" },
  { code: "am-ET", name: "Amharic (Ethiopia)", nativeName: "አማርኛ (ኢትዮጵያ)", script: "Ethiopic", rtl: false, flag: "🇪🇹", region: "South Asia" },
  { code: "ti-ER", name: "Tigrinya (Eritrea)", nativeName: "ትግርኛ (ኤርትራ)", script: "Ethiopic", rtl: false, flag: "🇪🇷", region: "South Asia" },

  // Southeast Asia
  { code: "vi-VN", name: "Vietnamese (Vietnam)", nativeName: "Tiếng Việt (Việt Nam)", script: "Latin", rtl: false, flag: "🇻🇳", region: "Southeast Asia" },
  { code: "th-TH", name: "Thai (Thailand)", nativeName: "ไทย (ประเทศไทย)", script: "Thai", rtl: false, flag: "🇹🇭", region: "Southeast Asia" },
  { code: "my-MM", name: "Burmese (Myanmar)", nativeName: "မြန်မာ (မြန်မာ)", script: "Myanmar", rtl: false, flag: "🇲🇲", region: "Southeast Asia" },
  { code: "km-KH", name: "Khmer (Cambodia)", nativeName: "ខ្មែរ (កម្ពុជា)", script: "Khmer", rtl: false, flag: "🇰🇭", region: "Southeast Asia" },
  { code: "lo-LA", name: "Lao (Laos)", nativeName: "ລາວ (ລາວ)", script: "Lao", rtl: false, flag: "🇱🇦", region: "Southeast Asia" },

  // Middle East & Africa
  { code: "ar-SA", name: "Arabic (Saudi Arabia)", nativeName: "العربية (السعودية)", script: "Arabic", rtl: true, flag: "🇸🇦", region: "Middle East & Africa", popular: true },
  { code: "fa-IR", name: "Persian (Iran)", nativeName: "فارسی (ایران)", script: "Arabic", rtl: true, flag: "🇮🇷", region: "Middle East & Africa" },
  { code: "he-IL", name: "Hebrew (Israel)", nativeName: "עברית (ישראל)", script: "Hebrew", rtl: true, flag: "🇮🇱", region: "Middle East & Africa" },
  { code: "hy-AM", name: "Armenian (Armenia)", nativeName: "Հայերեն (Հայաստան)", script: "Armenian", rtl: false, flag: "🇦🇲", region: "Middle East & Africa" },
  { code: "ka-GE", name: "Georgian (Georgia)", nativeName: "ქართული (საქართველო)", script: "Georgian", rtl: false, flag: "🇬🇪", region: "Middle East & Africa" },
  { code: "kk-KZ", name: "Kazakh (Kazakhstan)", nativeName: "Қазақ (Қазақстан)", script: "Cyrillic", rtl: false, flag: "🇰🇿", region: "Middle East & Africa" },
  { code: "uz-UZ", name: "Uzbek (Uzbekistan)", nativeName: "Oʻzbek (Oʻzbekiston)", script: "Latin", rtl: false, flag: "🇺🇿", region: "Middle East & Africa" },
  { code: "az-AZ", name: "Azerbaijani (Azerbaijan)", nativeName: "Azərbaycan (Azərbaycan)", script: "Latin", rtl: false, flag: "🇦🇿", region: "Middle East & Africa" },
];

export const LANGUAGE_REGIONS: Record<string, string[]> = {
  "Europe": ["en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-PT", "nl-NL", "sv-SE", "no-NO", "da-DK", "fi-FI", "is-IS", "pl-PL", "cs-CZ", "sk-SK", "hu-HU", "ro-RO", "bg-BG", "sr-RS", "sr-Latn-RS", "sr-Cyrl-RS", "hr-HR", "sl-SI", "et-EE", "lv-LV", "lt-LT", "el-GR", "ru-RU", "uk-UA", "tr-TR"],
  "Americas": ["en-US", "es-MX", "es-US", "pt-BR", "fr-CA"],
  "East Asia": ["ja-JP", "ko-KR", "zh-CN", "zh-SG", "zh-TW", "zh-HK", "zh-MO", "mn-MN"],
  "South Asia": ["hi-IN", "bn-BD", "pa-IN", "gu-IN", "mr-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "si-LK", "am-ET", "ti-ER"],
  "Southeast Asia": ["vi-VN", "th-TH", "my-MM", "km-KH", "lo-LA"],
  "Middle East & Africa": ["ar-SA", "fa-IR", "he-IL", "hy-AM", "ka-GE", "kk-KZ", "uz-UZ", "az-AZ"],
};

export const REGION_ORDER = ["Europe", "Americas", "East Asia", "South Asia", "Southeast Asia", "Middle East & Africa"];

export const REGION_ABBR: Record<string, string> = {
  "Europe": "EU",
  "Americas": "AM",
  "East Asia": "EA",
  "South Asia": "SA",
  "Southeast Asia": "SEA",
  "Middle East & Africa": "MEA",
};

/** normalise a possibly-legacy bare code to its canonical full locale tag. */
export function normalizeCode(code: string): string {
  return LEGACY_CODES[code] ?? code;
}

export function getLanguageByCode(code: string): Language | undefined {
  return LANGUAGES.find((l) => l.code === code || l.code === normalizeCode(code));
}

export function getPopularLanguages(): Language[] {
  return LANGUAGES.filter((l) => l.popular);
}

export function getLanguagesByRegion(region: string): Language[] {
  return LANGUAGES.filter((l) => l.region === region);
}

export function searchLanguages(query: string, availableCodes?: string[]): Language[] {
  const q = query.toLowerCase();
  const available = availableCodes?.map(normalizeCode);
  return LANGUAGES.filter((l) => {
    if (available && !available.includes(l.code)) return false;
    return l.name.toLowerCase().includes(q) || l.nativeName.toLowerCase().includes(q) || l.code.toLowerCase().includes(q);
  });
}

export function langDisplayName(code: string): string {
  const lang = getLanguageByCode(code);
  return lang ? lang.name : code;
}

export function langNativeName(code: string): string {
  const lang = getLanguageByCode(code);
  return lang ? lang.nativeName : code;
}

export function langWithCode(code: string): string {
  const lang = getLanguageByCode(code);
  return lang ? `${lang.name} (${code})` : code;
}

export function langFlag(code: string): string {
  const lang = getLanguageByCode(code);
  return lang ? lang.flag : "🌐";
}

export function isRtl(code: string): boolean {
  const lang = getLanguageByCode(code);
  return lang ? lang.rtl : false;
}
