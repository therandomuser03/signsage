import { test } from "node:test";
import assert from "node:assert/strict";
import { inspectHands, categorizeMorphs, categorizeBones, type BoneLike } from "./rigMap";

// Build a minimal fake skeleton: wrist -> chain, exactly like the real GLB hierarchy.
function fakeBone(name: string, parent: BoneLike | null): BoneLike {
  return { name, parent };
}

function makeHandBones(side: "L" | "R"): BoneLike[] {
  const bones: BoneLike[] = [];
  const wrist = fakeBone(`wrist.${side}`, null);
  bones.push(wrist);
  const add = (name: string, parent: BoneLike) => {
    const b = fakeBone(name, parent);
    bones.push(b);
    return b;
  };
  const t1 = add(`finger1-1.${side}`, wrist);
  const t2 = add(`finger1-2.${side}`, t1);
  add(`finger1-3.${side}`, t2);
  for (const [meta, finger] of [["metacarpal1", "finger2"], ["metacarpal2", "finger3"], ["metacarpal3", "finger4"], ["metacarpal4", "finger5"]] as const) {
    const m = add(`${meta}.${side}`, wrist);
    const j1 = add(`${finger}-1.${side}`, m);
    const j2 = add(`${finger}-2.${side}`, j1);
    add(`${finger}-3.${side}`, j2);
  }
  return bones;
}

test("inspectHands maps a complete hand: all fingers present, connected, no missing bones", () => {
  const bones = [...makeHandBones("L"), ...makeHandBones("R")];
  const hands = inspectHands(bones);
  assert.equal(hands.length, 2);
  for (const hand of hands) {
    assert.equal(hand.wrist, `wrist.${hand.side}`);
    for (const finger of hand.fingers) {
      assert.deepEqual(finger.missing, [], `${hand.side} ${finger.finger} missing bones`);
      assert.equal(finger.connected, true, `${hand.side} ${finger.finger} chain not connected`);
      assert.ok(finger.jointPresent.every(Boolean));
    }
  }
  // finger1 (thumb) must have exactly 3 joints and no metacarpal
  const thumb = hands[0].fingers.find((f) => f.finger === "thumb")!;
  assert.equal(thumb.present.length, 3);
  // other fingers: metacarpal + 3 joints
  const index = hands[0].fingers.find((f) => f.finger === "index")!;
  assert.equal(index.present.length, 4);
});

test("inspectHands reports missing bones and disconnection instead of assuming", () => {
  const bones = makeHandBones("L").filter((b) => b.name !== "finger2-2.L");
  const [hand] = inspectHands(bones);
  const index = hand.fingers.find((f) => f.finger === "index")!;
  assert.deepEqual(index.missing, ["finger2-2.L"]);
  // finger2-3.L is parented to the removed finger2-2.L, which is absent from the
  // skeleton map, so connectivity cannot be verified and must report false.
  assert.equal(index.connected, false);
});

test("categorizeMorphs buckets eyes, brows, mouth/jaw and visemes from real names", () => {
  const names = [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft",
    "browDownLeft", "browInnerUp",
    "jawOpen", "mouthSmile", "noseSneerLeft",
    "viseme_PP", "viseme_aa",
    "somethingElse",
  ];
  const groups = categorizeMorphs(names);
  const byLabel = new Map(groups.map((g) => [g.label, g.targets]));
  assert.deepEqual(byLabel.get("Eyes — blink / wide / squint"), ["eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft"]);
  assert.deepEqual(byLabel.get("Brow"), ["browDownLeft", "browInnerUp"]);
  assert.deepEqual(byLabel.get("Jaw / mouth / nose"), ["jawOpen", "mouthSmile", "noseSneerLeft"]);
  assert.deepEqual(byLabel.get("Speech visemes"), ["viseme_PP", "viseme_aa"]);
  assert.deepEqual(byLabel.get("Other"), ["somethingElse"]);
});

test("categorizeBones covers the whole skeleton exactly once", () => {
  const names = ["spine01", "spine05", "head", "wrist.L", "finger1-1.L", "metacarpal2.R", "toe1-1.L", "WeaponR"];
  const groups = categorizeBones(names);
  const flat = groups.flatMap((g) => g.bones);
  // names come back sanitized (three.js strips reserved chars at import)
  assert.deepEqual([...flat].sort(), [...names.map((n) => n.replace(/[.]/g, ""))].sort());
  const byLabel = new Map(groups.map((g) => [g.label, g.bones]));
  assert.deepEqual(byLabel.get("Spine"), ["spine01", "spine05"]);
  assert.deepEqual(byLabel.get("Fingers (both hands)"), ["finger1-1L", "metacarpal2R"]);
  assert.deepEqual(byLabel.get("Other"), ["WeaponR"]);
});
