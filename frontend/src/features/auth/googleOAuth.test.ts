import { test } from "node:test";
import assert from "node:assert/strict";
import { isGoogleOAuthEnabled } from "./googleOAuth";

test("Google OAuth is enabled only when VITE_GOOGLE_OAUTH_ENABLED is exactly 'true'", () => {
  // Enabled case: the provider is configured on the host.
  assert.equal(isGoogleOAuthEnabled("true"), true);
  // Disabled cases: not configured, explicitly false, or empty.
  assert.equal(isGoogleOAuthEnabled(undefined), false);
  assert.equal(isGoogleOAuthEnabled("false"), false);
  assert.equal(isGoogleOAuthEnabled(""), false);
  assert.equal(isGoogleOAuthEnabled("True"), false);
});
