import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { PROCEDURAL_SIGNS, lerpPose, motionFor, poseAt } from "./proceduralSigns";
import { localGlossMapper } from "./glossMapper";

const manifest = JSON.parse(
  readFileSync(new URL("../../../../public/models/avatar/manifest.json", import.meta.url), "utf-8"),
);

test("every manifest gloss has a procedural motion", () => {
  for (const id of Object.keys(manifest)) {
    assert.ok(PROCEDURAL_SIGNS[id], `missing motion for ${id}`);
    assert.ok(PROCEDURAL_SIGNS[id].durationMs >= 800, `${id} too short`);
    assert.ok(PROCEDURAL_SIGNS[id].keys.length >= 2, `${id} needs keys`);
  }
});

test("mapper output always resolves to a motion (never crashes playback)", () => {
  const seq = localGlossMapper("hello thank you water xyzzy", manifest);
  assert.deepEqual(seq.glossIds, ["HELLO", "THANK_YOU", "WATER"]);
  assert.deepEqual(seq.unmatchedWords, ["xyzzy"]);
  for (const id of seq.glossIds) assert.ok(motionFor(id).durationMs > 0);
  assert.ok(motionFor("NO_SUCH_GLOSS").durationMs > 0, "unknown gloss falls back, not throws");
});

test("poses interpolate from rest and return to rest", () => {
  const start = poseAt(PROCEDURAL_SIGNS.HELLO, 0);
  const end = poseAt(PROCEDURAL_SIGNS.HELLO, 1);
  assert.ok(Math.abs(start.shoulderR[0]) < 0.01, "starts at rest");
  assert.ok(Math.abs(end.shoulderR[0]) < 0.01, "ends at rest");
  const mid = poseAt(PROCEDURAL_SIGNS.HELLO, 0.3);
  assert.ok(mid.shoulderR[0] < -1, "mid-motion raises the arm");
});

test("each gloss visibly differs at mid-motion", () => {
  const seen = new Set<string>();
  for (const id of Object.keys(manifest)) {
    const p = poseAt(motionFor(id), 0.5);
    seen.add(JSON.stringify([p.shoulderR, p.elbowR, p.head]));
  }
  assert.ok(seen.size >= 10, `only ${seen.size} distinct mid-poses for 14 glosses`);
});

test("lerpPose blends joint angles", () => {
  const a = poseAt(PROCEDURAL_SIGNS.YES, 0);
  const b = poseAt(PROCEDURAL_SIGNS.YES, 0.3);
  const m = lerpPose(a, b, 0.5);
  assert.ok(Math.abs(m.head[0] - (a.head[0] + b.head[0]) / 2) < 1e-9);
});
