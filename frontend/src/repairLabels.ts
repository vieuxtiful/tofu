/**
 * Provider IDs are durable backend implementation details.  Keep their
 * friendly names in one place so notifications and review controls never
 * expose configuration-oriented identifiers such as `telea_fallback`.
 */
const PROVIDER_LABELS: Record<string, string> = {
  analytic: "Surface reconstruction",
  flat: "Surface reconstruction",
  telea: "Surface reconstruction",
  telea_fallback: "Editable reconstruction",
  lama: "Texture repair",
  lama_local: "Texture repair",
  diffstr_experimental: "Experimental detail repair",
  brushnet_experimental: "Experimental guided repair",
  adobe_fill: "Cloud content-aware fill",
  manual: "Manual repair",
};

export function providerLabel(provider: string | null | undefined): string {
  if (!provider) return "Smart fill";
  return PROVIDER_LABELS[provider]
    ?? provider.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
