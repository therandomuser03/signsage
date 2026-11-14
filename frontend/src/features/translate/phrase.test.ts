import { test } from "node:test";
import assert from "node:assert/strict";
import { showSignPhraseButton } from "./phrase";
import type { CapabilitiesResponse } from "@/types/api";

function caps(overrides: Partial<CapabilitiesResponse> = {}): CapabilitiesResponse {
  return {
    api_version: "v1",
    websocket_path: "/ws/translate",
    auth_enabled: false,
    auth_required: false,
    history_enabled: false,
    sign_to_text_available: false,
    sign_to_text_unavailable_reason: null,
    sign_to_text_source: "unavailable",
    fallback_vocabulary: [],
    text_to_sign_mode: "avatar_gloss",
    avatar_vocabulary_size: 14,
    landmark_count: 543,
    max_frames_per_sequence: 160,
    word_recognition_enabled: true,
    ...overrides,
  };
}

test("hidden by default (flag unset), even with word mode enabled", () => {
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true })), false);
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true }), ""), false);
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true }), "false"), false);
});

test("hidden with flag true when word mode disabled or null", () => {
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: false }), "true"), false);
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: null }), "true"), false);
  assert.strictEqual(showSignPhraseButton(null, "true"), false);
});

test("visible only with flag true AND word mode enabled", () => {
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true }), "true"), true);
});

test("hidden when word_recognition_enabled absent from payload", () => {
  const without = { ...caps() };
  delete without.word_recognition_enabled;
  assert.strictEqual(showSignPhraseButton(without, "true"), false);
});
