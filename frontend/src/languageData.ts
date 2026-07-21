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

export const LANGUAGES: Language[] = [
  // Europe
  { code: "en", name: "English", nativeName: "English", script: "Latin", rtl: false, flag: "🇺🇸", region: "Europe", popular: true },
  { code: "es", name: "Spanish", nativeName: "Español", script: "Latin", rtl: false, flag: "🇪🇸", region: "Europe", popular: true },
  { code: "fr", name: "French", nativeName: "Français", script: "Latin", rtl: false, flag: "🇫🇷", region: "Europe", popular: true },
  { code: "de", name: "German", nativeName: "Deutsch", script: "Latin", rtl: false, flag: "🇩🇪", region: "Europe", popular: true },
  { code: "it", name: "Italian", nativeName: "Italiano", script: "Latin", rtl: false, flag: "🇮🇹", region: "Europe" },
  { code: "pt", name: "Portuguese", nativeName: "Português", script: "Latin", rtl: false, flag: "🇵🇹", region: "Europe" },
  { code: "nl", name: "Dutch", nativeName: "Nederlands", script: "Latin", rtl: false, flag: "🇳🇱", region: "Europe" },
  { code: "sv", name: "Swedish", nativeName: "Svenska", script: "Latin", rtl: false, flag: "🇸🇪", region: "Europe" },
  { code: "no", name: "Norwegian", nativeName: "Norsk", script: "Latin", rtl: false, flag: "🇳🇴", region: "Europe" },
  { code: "da", name: "Danish", nativeName: "Dansk", script: "Latin", rtl: false, flag: "🇩🇰", region: "Europe" },
  { code: "fi", name: "Finnish", nativeName: "Suomi", script: "Latin", rtl: false, flag: "🇫🇮", region: "Europe" },
  { code: "is", name: "Icelandic", nativeName: "Íslenska", script: "Latin", rtl: false, flag: "🇮🇸", region: "Europe" },
  { code: "pl", name: "Polish", nativeName: "Polski", script: "Latin", rtl: false, flag: "🇵🇱", region: "Europe" },
  { code: "cs", name: "Czech", nativeName: "Čeština", script: "Latin", rtl: false, flag: "🇨🇿", region: "Europe" },
  { code: "sk", name: "Slovak", nativeName: "Slovenčina", script: "Latin", rtl: false, flag: "🇸🇰", region: "Europe" },
  { code: "hu", name: "Hungarian", nativeName: "Magyar", script: "Latin", rtl: false, flag: "🇭🇺", region: "Europe" },
  { code: "ro", name: "Romanian", nativeName: "Română", script: "Latin", rtl: false, flag: "🇷🇴", region: "Europe" },
  { code: "bg", name: "Bulgarian", nativeName: "Български", script: "Cyrillic", rtl: false, flag: "🇧🇬", region: "Europe" },
  { code: "sr", name: "Serbian", nativeName: "Српски", script: "Cyrillic", rtl: false, flag: "🇷🇸", region: "Europe" },
  { code: "sr-latn", name: "Serbian (Latin)", nativeName: "Srpski", script: "Latin", rtl: false, flag: "🇷🇸", region: "Europe" },
  { code: "sr-cyrl", name: "Serbian (Cyrillic)", nativeName: "Српски", script: "Cyrillic", rtl: false, flag: "🇷🇸", region: "Europe" },
  { code: "hr", name: "Croatian", nativeName: "Hrvatski", script: "Latin", rtl: false, flag: "🇭🇷", region: "Europe" },
  { code: "sl", name: "Slovenian", nativeName: "Slovenščina", script: "Latin", rtl: false, flag: "🇸🇮", region: "Europe" },
  { code: "et", name: "Estonian", nativeName: "Eesti", script: "Latin", rtl: false, flag: "🇪🇪", region: "Europe" },
  { code: "lv", name: "Latvian", nativeName: "Latviešu", script: "Latin", rtl: false, flag: "🇱🇻", region: "Europe" },
  { code: "lt", name: "Lithuanian", nativeName: "Lietuvių", script: "Latin", rtl: false, flag: "🇱🇹", region: "Europe" },
  { code: "el", name: "Greek", nativeName: "Ελληνικά", script: "Greek", rtl: false, flag: "🇬🇷", region: "Europe" },
  { code: "ru", name: "Russian", nativeName: "Русский", script: "Cyrillic", rtl: false, flag: "🇷🇺", region: "Europe", popular: true },
  { code: "uk", name: "Ukrainian", nativeName: "Українська", script: "Cyrillic", rtl: false, flag: "🇺🇦", region: "Europe" },
  { code: "tr", name: "Turkish", nativeName: "Türkçe", script: "Latin", rtl: false, flag: "🇹🇷", region: "Europe" },

  // East Asia
  { code: "ja", name: "Japanese", nativeName: "日本語", script: "Kanji/Hiragana", rtl: false, flag: "🇯🇵", region: "East Asia", popular: true },
  { code: "ko", name: "Korean", nativeName: "한국어", script: "Hangul", rtl: false, flag: "🇰🇷", region: "East Asia", popular: true },
  { code: "zh-cn", name: "Chinese (Simplified)", nativeName: "简体中文", script: "Han (Simplified)", rtl: false, flag: "🇨🇳", region: "East Asia", popular: true },
  { code: "zh-sg", name: "Chinese (Singapore)", nativeName: "简体中文", script: "Han (Simplified)", rtl: false, flag: "🇸🇬", region: "East Asia" },
  { code: "zh-tw", name: "Chinese (Traditional)", nativeName: "繁體中文", script: "Han (Traditional)", rtl: false, flag: "🇹🇼", region: "East Asia" },
  { code: "zh-hk", name: "Chinese (Hong Kong)", nativeName: "繁體中文", script: "Han (Traditional)", rtl: false, flag: "🇭🇰", region: "East Asia" },
  { code: "zh-mo", name: "Chinese (Macau)", nativeName: "繁體中文", script: "Han (Traditional)", rtl: false, flag: "🇲🇴", region: "East Asia" },
  { code: "mn", name: "Mongolian", nativeName: "Монгол", script: "Cyrillic", rtl: false, flag: "🇲🇳", region: "East Asia" },

  // South Asia
  { code: "hi", name: "Hindi", nativeName: "हिन्दी", script: "Devanagari", rtl: false, flag: "🇮🇳", region: "South Asia", popular: true },
  { code: "bn", name: "Bengali", nativeName: "বাংলা", script: "Bengali", rtl: false, flag: "🇧🇩", region: "South Asia" },
  { code: "pa", name: "Punjabi", nativeName: "ਪੰਜਾਬੀ", script: "Gurmukhi", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "gu", name: "Gujarati", nativeName: "ગુજરાતી", script: "Gujarati", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "mr", name: "Marathi", nativeName: "मराठी", script: "Devanagari", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "ta", name: "Tamil", nativeName: "தமிழ்", script: "Tamil", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "te", name: "Telugu", nativeName: "తెలుగు", script: "Telugu", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "kn", name: "Kannada", nativeName: "ಕನ್ನಡ", script: "Kannada", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "ml", name: "Malayalam", nativeName: "മലയാളം", script: "Malayalam", rtl: false, flag: "🇮🇳", region: "South Asia" },
  { code: "si", name: "Sinhala", nativeName: "සිංහල", script: "Sinhala", rtl: false, flag: "🇱🇰", region: "South Asia" },
  { code: "am", name: "Amharic", nativeName: "አማርኛ", script: "Ethiopic", rtl: false, flag: "🇪🇹", region: "South Asia" },
  { code: "ti", name: "Tigrinya", nativeName: "ትግርኛ", script: "Ethiopic", rtl: false, flag: "🇪🇷", region: "South Asia" },

  // Southeast Asia
  { code: "vi", name: "Vietnamese", nativeName: "Tiếng Việt", script: "Latin", rtl: false, flag: "🇻🇳", region: "Southeast Asia" },
  { code: "th", name: "Thai", nativeName: "ไทย", script: "Thai", rtl: false, flag: "🇹🇭", region: "Southeast Asia" },
  { code: "my", name: "Burmese", nativeName: "မြန်မာ", script: "Myanmar", rtl: false, flag: "🇲🇲", region: "Southeast Asia" },
  { code: "km", name: "Khmer", nativeName: "ខ្មែរ", script: "Khmer", rtl: false, flag: "🇰🇭", region: "Southeast Asia" },
  { code: "lo", name: "Lao", nativeName: "ລາວ", script: "Lao", rtl: false, flag: "🇱🇦", region: "Southeast Asia" },

  // Middle East & Africa
  { code: "ar", name: "Arabic", nativeName: "العربية", script: "Arabic", rtl: true, flag: "🇸🇦", region: "Middle East & Africa", popular: true },
  { code: "fa", name: "Persian", nativeName: "فارسی", script: "Arabic", rtl: true, flag: "🇮🇷", region: "Middle East & Africa" },
  { code: "he", name: "Hebrew", nativeName: "עברית", script: "Hebrew", rtl: true, flag: "🇮🇱", region: "Middle East & Africa" },
  { code: "hy", name: "Armenian", nativeName: "Հայերեն", script: "Armenian", rtl: false, flag: "🇦🇲", region: "Middle East & Africa" },
  { code: "ka", name: "Georgian", nativeName: "ქართული", script: "Georgian", rtl: false, flag: "🇬🇪", region: "Middle East & Africa" },
  { code: "kk", name: "Kazakh", nativeName: "Қазақ", script: "Cyrillic", rtl: false, flag: "🇰🇿", region: "Middle East & Africa" },
  { code: "uz", name: "Uzbek", nativeName: "Oʻzbek", script: "Latin", rtl: false, flag: "🇺🇿", region: "Middle East & Africa" },
  { code: "az", name: "Azerbaijani", nativeName: "Azərbaycan", script: "Latin", rtl: false, flag: "🇦🇿", region: "Middle East & Africa" },
];

export const LANGUAGE_REGIONS: Record<string, string[]> = {
  "Europe": ["en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da", "fi", "is", "pl", "cs", "sk", "hu", "ro", "bg", "sr", "sr-latn", "sr-cyrl", "hr", "sl", "et", "lv", "lt", "el", "ru", "uk", "tr"],
  "East Asia": ["ja", "ko", "zh-cn", "zh-sg", "zh-tw", "zh-hk", "zh-mo", "mn"],
  "South Asia": ["hi", "bn", "pa", "gu", "mr", "ta", "te", "kn", "ml", "si", "am", "ti"],
  "Southeast Asia": ["vi", "th", "my", "km", "lo"],
  "Middle East & Africa": ["ar", "fa", "he", "hy", "ka", "kk", "uz", "az"],
};

export const REGION_ORDER = ["Europe", "East Asia", "South Asia", "Southeast Asia", "Middle East & Africa"];

export const REGION_ABBR: Record<string, string> = {
  "Europe": "EU",
  "East Asia": "EA",
  "South Asia": "SA",
  "Southeast Asia": "SEA",
  "Middle East & Africa": "MEA",
};

export function getLanguageByCode(code: string): Language | undefined {
  return LANGUAGES.find((l) => l.code === code);
}

export function getPopularLanguages(): Language[] {
  return LANGUAGES.filter((l) => l.popular);
}

export function getLanguagesByRegion(region: string): Language[] {
  return LANGUAGES.filter((l) => l.region === region);
}

export function searchLanguages(query: string, availableCodes?: string[]): Language[] {
  const q = query.toLowerCase();
  return LANGUAGES.filter((l) => {
    if (availableCodes && !availableCodes.includes(l.code)) return false;
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
