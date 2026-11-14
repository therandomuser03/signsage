import { test } from "node:test";
import assert from "node:assert/strict";
import { shouldUseVrm } from "./assetExists";

test("shouldUseVrm is true only when both VRM and manifest are available", () => {
  assert.strictEqual(shouldUseVrm(true, true), true);
  assert.strictEqual(shouldUseVrm(true, false), false);
  assert.strictEqual(shouldUseVrm(false, true), false);
  assert.strictEqual(shouldUseVrm(false, false), false);
});
