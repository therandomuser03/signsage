import { getSupabase, isSupabaseConfigured } from "./client";

/** Current access token, or null for guests / unconfigured auth. Supabase refreshes it automatically. */
export async function getAccessToken(): Promise<string | null> {
  if (!isSupabaseConfigured()) return null;
  const { data } = await getSupabase().auth.getSession();
  return data.session?.access_token ?? null;
}

export async function signUp(email: string, password: string) {
  const { data, error } = await getSupabase().auth.signUp({ email, password });
  if (error) throw error;
  return data;
}

export async function signIn(email: string, password: string) {
  const { data, error } = await getSupabase().auth.signInWithPassword({ email, password });
  if (error) throw error;
  return data;
}

export async function signOut(): Promise<void> {
  const { error } = await getSupabase().auth.signOut();
  if (error) throw error;
}

/** Google OAuth via Supabase. Redirects away on success; throws honestly (e.g. provider not enabled) on failure. */
export async function signInWithGoogle() {
  const { data, error } = await getSupabase().auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: `${window.location.origin}/app` },
  });
  if (error) throw error;
  return data;
}

/** Sends a Supabase password-reset email. Throws on failure. */
export async function resetPassword(email: string) {
  const { data, error } = await getSupabase().auth.resetPasswordForEmail(email, {
    redirectTo: `${window.location.origin}/login`,
  });
  if (error) throw error;
  return data;
}

/** Subscribe to sign-in/out; returns an unsubscribe function. No-op when auth isn't configured. */
export function onAuthChange(callback: (signedIn: boolean) => void): () => void {
  if (!isSupabaseConfigured()) return () => undefined;
  const { data } = getSupabase().auth.onAuthStateChange((_event, session) => callback(Boolean(session)));
  return () => data.subscription.unsubscribe();
}
