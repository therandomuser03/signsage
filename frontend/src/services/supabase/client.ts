import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { config } from "../../app/config";

let client: SupabaseClient | null = null;
/** Allows tests to inject a fake Supabase client, bypassing real config. */
let testClient: SupabaseClient | null = null;

export function isSupabaseConfigured(): boolean {
  return Boolean(config.supabaseUrl && config.supabasePublishableKey);
}

/** Throws if Supabase isn't configured -- guard with isSupabaseConfigured() (guest mode has no client). */
export function getSupabase(): SupabaseClient {
  if (testClient) return testClient;
  if (!isSupabaseConfigured()) throw new Error("Supabase is not configured (VITE_SUPABASE_URL / VITE_SUPABASE_PUBLISHABLE_KEY).");
  if (!client) client = createClient(config.supabaseUrl, config.supabasePublishableKey);
  return client;
}

/** Injects a fake Supabase client for unit tests. Call with null to reset. */
export function setSupabaseClientForTesting(c: SupabaseClient | null): void {
  testClient = c;
}
