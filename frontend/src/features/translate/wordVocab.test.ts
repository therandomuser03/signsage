import { test } from "node:test";
import assert from "node:assert/strict";
import {
  defaultInterpreterTab,
  splitWordVocab,
  wordDisabledDetail,
  wordVocabSummary,
} from "./wordVocab";
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
    word_recognition_vocabulary: ["hello", "me", "you", "yes", "no", "help", "water", "where", "rest"],
    ...overrides,
  };
}

test("default tab is word when capabilities enable word recognition", () => {
  assert.strictEqual(defaultInterpreterTab(caps({ word_recognition_enabled: true })), "word");
});

test("default tab is rule when word recognition is disabled, null, absent, or capabilities failed to load", () => {
  assert.strictEqual(defaultInterpreterTab(caps({ word_recognition_enabled: false })), "rule");
  assert.strictEqual(defaultInterpreterTab(caps({ word_recognition_enabled: null })), "rule");
  const without = { ...caps() };
  delete without.word_recognition_enabled;
  assert.strictEqual(defaultInterpreterTab(without), "rule");
  assert.strictEqual(defaultInterpreterTab(null), "rule");
});

test("rest class is excluded from the word list, case-insensitively", () => {
  assert.deepStrictEqual(splitWordVocab(["hello", "rest", "water"]), {
    words: ["hello", "water"],
    restClass: true,
  });
  assert.deepStrictEqual(splitWordVocab(["rest", "REST", "Rest"]), {
    words: [],
    restClass: true,
  });
  assert.deepStrictEqual(splitWordVocab(["hello", "water"]), {
    words: ["hello", "water"],
    restClass: false,
  });
  assert.deepStrictEqual(splitWordVocab(null), { words: [], restClass: false });
  assert.deepStrictEqual(splitWordVocab(undefined), { words: [], restClass: false });
});

test("summary counts 8 words plus a rest (no-sign) class for the 9-entry list", () => {
  assert.strictEqual(
    wordVocabSummary(["hello", "me", "you", "yes", "no", "help", "water", "where", "rest"]),
    "8 words plus a rest (no-sign) class",
  );
  assert.strictEqual(wordVocabSummary(["hello"]), "1 word");
  assert.strictEqual(wordVocabSummary(["hello", "water"]), "2 words");
  assert.strictEqual(wordVocabSummary(null), "0 words");
});

test("disabled detail: load failure or the reason from capabilities, else null", () => {
  assert.strictEqual(
    wordDisabledDetail(null),
    "Capabilities could not be loaded — the backend may be offline or still waking up.",
  );
  assert.strictEqual(wordDisabledDetail(caps({ sign_to_text_unavailable_reason: "model_bundle_missing" })), "model_bundle_missing");
  assert.strictEqual(wordDisabledDetail(caps({ sign_to_text_unavailable_reason: null })), null);
});
