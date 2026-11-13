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
} as const;
