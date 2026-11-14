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

test("visible by default (flag unset), when word mode is enabled", () => {
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true })), true);
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true }), ""), true);
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true }), "true"), true);
});

test("hidden only when flag is exactly 'false'", () => {
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: true }), "false"), false);
});

test("hidden when word mode disabled or null, even if flag is not false", () => {
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: false })), false);
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: false }), "true"), false);
  assert.strictEqual(showSignPhraseButton(caps({ word_recognition_enabled: null })), false);
  assert.strictEqual(showSignPhraseButton(null), false);
});

test("hidden when word_recognition_enabled absent from payload", () => {
  const without = { ...caps() };
  delete without.word_recognition_enabled;
  assert.strictEqual(showSignPhraseButton(without), false);
});
