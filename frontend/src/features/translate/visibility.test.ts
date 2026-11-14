import { test } from "node:test";
import assert from "node:assert/strict";
import { showWordRecognitionCard, showLegacyUpload } from "./visibility";
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
    word_recognition_vocabulary: ["hello", "me", "you", "yes", "no", "help", "water", "where"],
    ...overrides,
  };
}

test("word recognition card: shown only when word_recognition_enabled is true", () => {
  assert.strictEqual(showWordRecognitionCard(caps({ word_recognition_enabled: true })), true);
  assert.strictEqual(showWordRecognitionCard(caps({ word_recognition_enabled: false })), false);
  assert.strictEqual(showWordRecognitionCard(caps({ word_recognition_enabled: null })), false);
  assert.strictEqual(showWordRecognitionCard(null), false);
});

test("word recognition card: hidden when word_recognition_enabled absent from payload", () => {
  const without = { ...caps() };
  delete without.word_recognition_enabled;
  assert.strictEqual(showWordRecognitionCard(without), false);
});

test("legacy upload: hidden by default, even for true-ish values", () => {
  assert.strictEqual(showLegacyUpload(), false);
  assert.strictEqual(showLegacyUpload(""), false);
  assert.strictEqual(showLegacyUpload("false"), false);
  assert.strictEqual(showLegacyUpload("1"), false);
  assert.strictEqual(showLegacyUpload("TRUE"), false);
  assert.strictEqual(showLegacyUpload("True"), false);
});

test("legacy upload: shown only for exactly \"true\"", () => {
  assert.strictEqual(showLegacyUpload("true"), true);
});
