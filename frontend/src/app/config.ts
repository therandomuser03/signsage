// Single place that reads environment configuration.
// `import.meta.env` is provided by Vite in browser builds; in test/Node
// environments it may be undefined, so fall back gracefully.
export function deriveWsUrl(apiUrl: string): string {
  return apiUrl.replace(/^http/i, "ws").replace(/\/+$/, "") + "/ws/translate";
}

const env = (import.meta as { env?: Record<string, string> }).env ?? {};
const apiUrl = (env.VITE_SIGNSAGE_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

// Derive WS URL from apiUrl, but never append /api/v1 to the WS path
const derivedWsUrl = deriveWsUrl(apiUrl).replace(/\/api\/v1\/ws\/translate$/, "/ws/translate");

export const config = {
  apiUrl,
  wsUrl: env.VITE_SIGNSAGE_WS_URL ?? derivedWsUrl,
  supabaseUrl: env.VITE_SUPABASE_URL ?? "",
  supabasePublishableKey: env.VITE_SUPABASE_PUBLISHABLE_KEY ?? "",
  // Dev-only: when "true", show the labeled-capture panel under the Word mode
  // card. Any other value (incl. unset) hides it entirely.
  captureMode: env.VITE_CAPTURE_MODE ?? "",
  // Dev-only phrase capture (POST /api/v1/translation/sentence): button shows
  // only when "true" AND capabilities has word mode enabled. Default = hidden.
  phraseMode: env.VITE_PHRASE_MODE ?? "",
  // Legacy six-word video-upload pilot: the panel shows only when exactly
  // "true". Any other value (incl. unset) hides it. Default = hidden.
  legacyUpload: env.VITE_LEGACY_UPLOAD ?? "",
} as const;
