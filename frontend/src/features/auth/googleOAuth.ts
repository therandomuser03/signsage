/** Google OAuth is only wired up when explicitly enabled for this deployment.
 *
 * Set VITE_GOOGLE_OAUTH_ENABLED=true on Vercel (or your host) only after the
 * Google provider is configured in Supabase. Until then the Login page shows
 * email/password only and never renders a broken Google button.
 */
export function isGoogleOAuthEnabled(
  googleOAuthEnabled: string | undefined,
): boolean {
  return googleOAuthEnabled === "true";
}
