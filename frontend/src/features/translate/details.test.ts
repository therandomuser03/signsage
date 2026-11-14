import { test } from "node:test";
import assert from "node:assert/strict";
import {
  type CaptureDetails,
  formatDetailsCompact,
  detailsToJson,
} from "./details";

test("formatDetailsCompact correctly formats a word capture", () => {
  const details: CaptureDetails = {
    kind: "word",
    framesCaptured: 25,
    effectiveFps: 10.02,
    leftHandFraction: 0.12,
    rightHandFraction: 0.95,
    roundTripMs: 145.67,
    predictedLabel: "water",
    confidence: 0.9823,
    threshold: 0.7,
    accepted: true,
    top3: [
      { label: "water", probability: 0.982 },
      { label: "where", probability: 0.015 },
    ],
    modelVersion: "word_v3",
    status: "ok",
  };

  const formatted = formatDetailsCompact(details);
  assert.strictEqual(formatted.kind, "word");
  assert.strictEqual(formatted.frames_captured, 25);
  assert.strictEqual(formatted.effective_fps, 10.0);
  assert.strictEqual(formatted.left_hand_fraction, 0.12);
  assert.strictEqual(formatted.right_hand_fraction, 0.95);
  assert.strictEqual(formatted.request_round_trip_ms, 145.7);
  assert.strictEqual(formatted.predicted_label, "water");
  assert.strictEqual(formatted.confidence, 0.982);
  assert.strictEqual(formatted.threshold, 0.7);
  assert.strictEqual(formatted.accepted, true);
  assert.strictEqual(formatted.model_version, "word_v3");
  assert.strictEqual(formatted.status, "ok");

  const jsonStr = detailsToJson(details);
  const parsed = JSON.parse(jsonStr);
  assert.deepStrictEqual(parsed, formatted);
});

test("formatDetailsCompact correctly formats a phrase capture", () => {
  const details: CaptureDetails = {
    kind: "phrase",
    framesCaptured: 100,
    effectiveFps: 10.0,
    leftHandFraction: 0.0,
    rightHandFraction: 0.99,
    roundTripMs: 250.2,
    glossSequence: [
      { gloss: "where", confidence: 0.95 },
      { gloss: "water", confidence: 0.98 },
    ],
    sentence: "Where is the water?",
    nWindows: 16,
    accepted: true,
    threshold: 0.7,
    modelVersion: "word_v3",
    status: "ok",
  };

  const formatted = formatDetailsCompact(details);
  assert.strictEqual(formatted.kind, "phrase");
  assert.strictEqual(formatted.sentence, "Where is the water?");
  assert.strictEqual(formatted.n_windows, 16);
  assert.deepStrictEqual(formatted.gloss_sequence, [
    { gloss: "where", confidence: 0.95 },
    { gloss: "water", confidence: 0.98 },
  ]);

  const jsonStr = detailsToJson(details);
  assert.strictEqual(typeof jsonStr, "string");
  assert.ok(jsonStr.includes('"sentence":"Where is the water?"'));
});
