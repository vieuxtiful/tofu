import { useMemo, useState } from "react";
import { Globe, Search, Star } from "lucide-react";
import { LANGUAGES, LANGUAGE_REGIONS, REGION_ORDER, getPopularLanguages, searchLanguages, type Language } from "./languageData";

interface LanguageSelectorProps {
  selected: string;
  onSelect: (code: string) => void;
  availableCodes?: string[];
  title?: string;
  description?: string;
  lastUsed?: string | null;
}

export default function LanguageSelector({
  selected,
  onSelect,
  availableCodes,
  title = "Select Language",
  description = "Choose a language for translation.",
  lastUsed,
}: LanguageSelectorProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<string>("popular");

  const codes = availableCodes ?? LANGUAGES.map((l) => l.code);

  const popular = useMemo(() => {
    const p = getPopularLanguages();
    return p.filter((l) => codes.includes(l.code));
  }, [codes]);

  const filtered = useMemo<Language[]>(() => {
    if (searchQuery) {
      return searchLanguages(searchQuery, codes);
    }
    if (activeTab === "popular") {
      return popular;
    }
    if (activeTab === "all") {
      return LANGUAGES.filter((l) => codes.includes(l.code));
    }
    const regionCodes = LANGUAGE_REGIONS[activeTab] || [];
    return LANGUAGES.filter((l) => regionCodes.includes(l.code) && codes.includes(l.code));
  }, [searchQuery, activeTab, popular, codes]);

  const regionsWithLanguages = REGION_ORDER.filter((r) => {
    const regionCodes = LANGUAGE_REGIONS[r] || [];
    return regionCodes.some((c) => codes.includes(c));
  });

  return (
    <div className="space-y-4">
      <div className="text-center space-y-2">
        <div className="flex justify-center">
          <div className="p-3 rounded-full bg-cyan-500/10">
            <Globe className="h-8 w-8 text-cyan-400" />
          </div>
        </div>
        <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">{title}</h3>
        <p className="subtext text-sm text-zinc-500">{description}</p>
        {lastUsed && (
          <div className="flex justify-center">
            <span className="inline-flex items-center gap-1 rounded-full bg-zinc-200 px-2 py-0.5 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
              <Star size={10} className="text-yellow-500 fill-yellow-500" />
              Last used: {LANGUAGES.find((l) => l.code === lastUsed)?.name ?? lastUsed}
            </span>
          </div>
        )}
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-600" />
        <input
          type="text"
          placeholder="Search languages..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full rounded-lg border border-zinc-300 bg-white pl-10 pr-3 py-2 text-sm text-zinc-800 outline-hidden focus:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
        />
      </div>

      {/* Region tabs */}
      <div className="flex flex-wrap gap-1">
        <button
          onClick={() => { setActiveTab("popular"); setSearchQuery(""); }}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
            activeTab === "popular" ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
          }`}
        >
          Popular
        </button>
        {regionsWithLanguages.map((region) => (
          <button
            key={region}
            onClick={() => { setActiveTab(region); setSearchQuery(""); }}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              activeTab === region ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
            }`}
          >
            {region}
          </button>
        ))}
        <button
          onClick={() => { setActiveTab("all"); setSearchQuery(""); }}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
            activeTab === "all" ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
          }`}
        >
          All
        </button>
      </div>

      {/* Language grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-[350px] overflow-y-auto pr-1">
        {filtered.map((lang) => {
          const isSel = lang.code === selected;
          const isLast = lang.code === lastUsed;
          return (
            <button
              key={lang.code}
              onClick={() => onSelect(lang.code)}
              className={`relative flex items-center gap-2 rounded-lg border p-3 text-left transition ${
                isSel
                  ? "border-cyan-500 bg-cyan-50 dark:bg-cyan-950/30"
                  : "border-zinc-300 bg-zinc-100 hover:border-cyan-600 hover:bg-zinc-200 dark:border-zinc-700 dark:bg-zinc-800/50 dark:hover:bg-zinc-800"
              }`}
            >
              {isLast && (
                <Star size={10} className="absolute top-1 right-1 text-yellow-500 fill-yellow-500" />
              )}
              <span className="text-xl">{lang.flag}</span>
              <div className="min-w-0">
                <p lang={lang.code} className={`text-sm font-medium truncate ${isSel ? "text-cyan-700 dark:text-cyan-300" : "text-zinc-800 dark:text-zinc-200"}`}>
                  {lang.nativeName}
                </p>
                <p className="subtext text-xs text-zinc-500 truncate">
                  {lang.name} · {lang.script}
                </p>
              </div>
              {lang.rtl && (
                <span className="ml-auto rounded-sm bg-zinc-300 px-1 py-0.5 text-[10px] text-zinc-600 dark:bg-zinc-700 dark:text-zinc-400">RTL</span>
              )}
            </button>
          );
        })}
        {filtered.length === 0 && (
          <div className="col-span-full py-6 text-center text-sm text-zinc-600">
            No languages found matching "{searchQuery}"
          </div>
        )}
      </div>
    </div>
  );
}
