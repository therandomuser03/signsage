/** OAuth via Supabase is only wired up when explicitly enabled for this deployment.
 *
 * Set VITE_<PROVIDER>_OAUTH_ENABLED=true on your host only after the provider
 * is configured in Supabase. Until then the Login page shows email/password only
 * and never renders a broken OAuth button.
 */
function isProviderOAuthEnabled(value: string | undefined): boolean {
  return value === "true";
}

/** Google OAuth is enabled only when VITE_GOOGLE_OAUTH_ENABLED is exactly 'true'. */
export function isGoogleOAuthEnabled(
  googleOAuthEnabled: string | undefined,
): boolean {
  return isProviderOAuthEnabled(googleOAuthEnabled);
}

/** Microsoft OAuth is enabled only when VITE_MICROSOFT_OAUTH_ENABLED is exactly 'true'. */
export function isMicrosoftOAuthEnabled(
  microsoftOAuthEnabled: string | undefined,
): boolean {
  return isProviderOAuthEnabled(microsoftOAuthEnabled);
}
