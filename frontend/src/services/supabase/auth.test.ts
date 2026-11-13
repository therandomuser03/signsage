import { test } from "node:test";
import assert from "node:assert/strict";
import type { SupabaseClient } from "@supabase/supabase-js";
import { setSupabaseClientForTesting } from "./client";

// Provide a browser-like `window` used by signInWithMicrosoft for redirectTo.
(globalThis as any).window = { location: { origin: "http://test.local" } };

import { signInWithMicrosoft } from "./auth";

test("signInWithMicrosoft calls signInWithOAuth with provider 'azure' and scopes including 'email'", async () => {
  let captured: any = null;
  setSupabaseClientForTesting({
    auth: {
      signInWithOAuth: async (args: any) => {
        captured = args;
        return { data: null, error: null };
      },
    },
  } as unknown as SupabaseClient);
  await signInWithMicrosoft();
  assert.equal(captured?.provider, "azure");
  assert.ok(
    captured?.options.scopes?.includes("email"),
    `scopes should include 'email', got: ${JSON.stringify(captured?.options.scopes)}`,
  );
  assert.equal(captured?.options.redirectTo, "http://test.local/app");
  setSupabaseClientForTesting(null);
});
