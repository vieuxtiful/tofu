// 🍢 ToFU — language display names

export const LANGUAGE_NAMES: Record<string, string> = {
  "en": "English",
  "es": "Spanish",
  "fr": "French",
  "de": "German",
  "it": "Italian",
  "pt": "Portuguese",
  "ru": "Russian",
  "uk": "Ukrainian",
  "el": "Greek",
  "ar": "Arabic",
  "fa": "Persian",
  "he": "Hebrew",
  "ja": "Japanese",
  "ko": "Korean",
  "zh-cn": "Chinese (Simplified)",
  "zh-sg": "Chinese (Simplified)",
  "zh-tw": "Chinese (Traditional)",
  "zh-hk": "Chinese (Traditional)",
  "zh-mo": "Chinese (Traditional)",
  "sr-latn": "Serbian (Latin)",
  "sr-cyrl": "Serbian (Cyrillic)",
  "hi": "Hindi",
  "th": "Thai",
  "vi": "Vietnamese",
  "bn": "Bengali",
  "pa": "Punjabi",
  "gu": "Gujarati",
  "mr": "Marathi",
  "ta": "Tamil",
  "te": "Telugu",
  "kn": "Kannada",
  "ml": "Malayalam",
  "si": "Sinhala",
  "my": "Burmese",
  "km": "Khmer",
  "lo": "Lao",
  "am": "Amharic",
  "ti": "Tigrinya",
  "hy": "Armenian",
  "ka": "Georgian",
  "mn": "Mongolian",
  "kk": "Kazakh",
  "uz": "Uzbek",
  "az": "Azerbaijani",
  "tr": "Turkish",
  "nl": "Dutch",
  "sv": "Swedish",
  "no": "Norwegian",
  "da": "Danish",
  "fi": "Finnish",
  "is": "Icelandic",
  "pl": "Polish",
  "cs": "Czech",
  "sk": "Slovak",
  "hu": "Hungarian",
  "ro": "Romanian",
  "bg": "Bulgarian",
  "sr": "Serbian",
  "hr": "Croatian",
  "sl": "Slovenian",
  "et": "Estonian",
  "lv": "Latvian",
  "lt": "Lithuanian",
};

export function langDisplayName(code: string): string {
  return LANGUAGE_NAMES[code] ?? code;
}

export function langWithCode(code: string): string {
  return `${langDisplayName(code)} (${code})`;
}

export const LANGUAGE_REGIONS: Record<string, string[]> = {
  "Europe": ["en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da", "fi", "is", "pl", "cs", "sk", "hu", "ro", "bg", "sr", "sr-latn", "sr-cyrl", "hr", "sl", "et", "lv", "lt", "el", "ru", "uk", "tr"],
  "East Asia": ["ja", "ko", "zh-cn", "zh-sg", "zh-tw", "zh-hk", "zh-mo", "mn"],
  "South Asia": ["hi", "bn", "pa", "gu", "mr", "ta", "te", "kn", "ml", "si", "am", "ti"],
  "Southeast Asia": ["vi", "th", "my", "km", "lo"],
  "Middle East & Africa": ["ar", "fa", "he", "hy", "ka", "kk", "uz", "az"],
};

export const REGION_ORDER = ["Europe", "East Asia", "South Asia", "Southeast Asia", "Middle East & Africa"];
