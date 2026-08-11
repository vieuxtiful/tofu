declare module "*.png" {
  const url: string;
  export default url;
}

declare module "*.css";

/* Vite's ?raw suffix, used by tests that assert on stylesheet content
   directly -- happy-dom drops `var()` declarations, so a computed-style
   assertion cannot see whether a themed token is wired up. */
declare module "*.css?raw" {
  const source: string;
  export default source;
}

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
