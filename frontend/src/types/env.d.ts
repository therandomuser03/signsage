// Vite env vars used by SignSage (mirrors .env.example). Vite's own
// `vite/client` types merge with this declaration.
interface ImportMetaEnv {
  readonly VITE_SIGNSAGE_API_URL?: string;
  readonly VITE_SIGNSAGE_WS_URL?: string;
  readonly VITE_SUPABASE_URL?: string;
  readonly VITE_SUPABASE_PUBLISHABLE_KEY?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
