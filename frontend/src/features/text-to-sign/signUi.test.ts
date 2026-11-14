import { test } from "node:test";
import assert from "node:assert/strict";
import {
  EXAMPLE_CHIPS,
  glossChipState,
  glossIdToText,
  missingSignLabel,
  sortVocabulary,
} from "./signUi";

test("example chips are exactly the specified six, in order", () => {
  assert.deepStrictEqual([...EXAMPLE_CHIPS], [
    "Hello",
    "Thank you",
    "Where is water?",
    "Help me",
    "Yes",
    "No",
  ]);
});

test("glossChipState: queued when not playing and nothing current", () => {
  assert.strictEqual(glossChipState(0, -1, false), "queued");
  assert.strictEqual(glossChipState(3, -1, false), "queued");
});

test("glossChipState: current only while playing at the current index", () => {
  assert.strictEqual(glossChipState(0, 0, true), "current");
  assert.strictEqual(glossChipState(2, 2, true), "current");
  assert.strictEqual(glossChipState(2, 2, false), "queued"); // paused between steps
  assert.strictEqual(glossChipState(1, 2, true), "past");
  assert.strictEqual(glossChipState(3, 2, true), "queued");
});

test("glossIdToText: underscores become spaces, lowercased", () => {
  assert.strictEqual(glossIdToText("THANK_YOU"), "thank you");
  assert.strictEqual(glossIdToText("HELLO"), "hello");
  assert.strictEqual(glossIdToText("WHERE"), "where");
});

test("sortVocabulary: alphabetical copy, input untouched", () => {
  const input = ["WHERE", "FOOD", "HELLO", "YES"];
  assert.deepStrictEqual(sortVocabulary(input), ["FOOD", "HELLO", "WHERE", "YES"]);
  assert.deepStrictEqual(input, ["WHERE", "FOOD", "HELLO", "YES"]);
});

test("missingSignLabel: word plus the not-in-library annotation", () => {
  assert.strictEqual(missingSignLabel("banana"), "banana — not in sign library");
});
