// SignSage — tests for useSpeechInput's pure helpers.
// These run in Node (no DOM), so SpeechRecognition is mocked on globalThis.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRecognizer, speechErrorMessage, type Rec } from "./useSpeechInput";

function installSpeechRecognition(Ctor: unknown) {
  const prev = (globalThis as unknown as Record<string, unknown>).window;
  (globalThis as unknown as Record<string, unknown>).window = {
    SpeechRecognition: Ctor,
    webkitSpeechRecognition: undefined,
  };
  return () => {
    (globalThis as unknown as Record<string, unknown>).window = prev;
  };
}

test("speechErrorMessage maps known error codes to user-facing messages", () => {
  assert.equal(speechErrorMessage("not-allowed"), "Microphone blocked. Allow access and try again.");
  assert.equal(speechErrorMessage("network"), "Network failure. Check your connection and try again.");
  assert.equal(speechErrorMessage("audio-capture"), "No microphone found. Connect a microphone and try again.");
  assert.equal(
    speechErrorMessage("service-not-allowed"),
    "Speech service not allowed. Type instead.",
  );
  assert.equal(speechErrorMessage("no-match"), "No speech recognized. Try again.");
  assert.equal(speechErrorMessage("unknown-error"), "Speech recognition failed. Type instead.");
});

test("createRecognizer returns null when SpeechRecognition is unavailable", () => {
  const restore = installSpeechRecognition(undefined);
  try {
    assert.equal(createRecognizer("en-IN"), null);
  } finally {
    restore();
  }
});

test("createRecognizer returns a recognizer configured with the requested lang", () => {
  class FakeRec implements Rec {
    continuous = false;
    interimResults = false;
    lang = "";
    start = () => {};
    stop = () => {};
    abort = () => {};
    onresult: Rec["onresult"] = null;
    onerror: Rec["onerror"] = null;
    onend: Rec["onend"] = null;
  }
  const restore = installSpeechRecognition(FakeRec as unknown as new () => Rec);
  try {
    const rec = createRecognizer("en-IN");
    assert.ok(rec);
    assert.equal(rec.lang, "en-IN");
    assert.equal(rec.continuous, false);
    assert.equal(rec.interimResults, true);
  } finally {
    restore();
  }
});
