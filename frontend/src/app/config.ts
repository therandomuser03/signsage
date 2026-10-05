// Single place that reads environment configuration.

export function deriveWsUrl(apiUrl: string): string {
  return apiUrl.replace(/^http/i, "ws").replace(/\/+$/, "") + "/ws/translate";
}

const apiUrl = (import.meta.env.VITE_SIGNSAGE_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

// Derive WS URL from apiUrl, but never append /api/v1 to the WS path
const derivedWsUrl = deriveWsUrl(apiUrl).replace(/\/api\/v1\/ws\/translate$/, "/ws/translate");

export const config = {
  apiUrl,
  wsUrl: import.meta.env.VITE_SIGNSAGE_WS_URL ?? derivedWsUrl,
  supabaseUrl: import.meta.env.VITE_SUPABASE_URL ?? "",
  supabasePublishableKey: import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY ?? "",
} as const;
