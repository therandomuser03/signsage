// Synthetic-sequence tests for AutoSegmenter.
// These verify control-flow behaviour only (segmentation, padding,
// sentence-pause timing). They are NOT accuracy evidence for the model.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ADAPTER_MIN_FRAMES,
  AUTO_SEGMENTER_DEFAULTS,
  AutoSegmenter,
  type AutoEvent,
  type AutoFrame,
} from "./autoSegmenter";
import type { LandmarkPoint } from "./landmarks";

/** 543 zero landmarks; when `hand` is set, one hand landmark is nonzero. */
function lm(hand: "none" | "left" | "right" = "none"): LandmarkPoint[] {
  const out: LandmarkPoint[] = Array.from({ length: 543 }, () => [0, 0, 0]);
  if (hand === "left") out[501] = [0.5, 0.5, 0.1];
  if (hand === "right") out[522] = [0.5, 0.5, 0.1];
  return out;
}

/** A scripted sequence of (durationMs, hand) steps starting at t=0. */
interface Step {
  ms: number;
  hand: "none" | "left" | "right";
}

function buildFrames(steps: Step[], intervalMs = 100): AutoFrame[] {
  const frames: AutoFrame[] = [];
  let t = 0;
  for (const step of steps) {
    const end = t + step.ms;
    while (t < end) {
      frames.push({ tMs: t, landmarks: lm(step.hand) });
      t += intervalMs;
    }
  }
  return frames;
}

function feed(seg: AutoSegmenter, frames: AutoFrame[]): AutoEvent[] {
  const events: AutoEvent[] = [];
  for (const f of frames) events.push(...seg.pushFrame(f));
  return events;
}

const segments = (events: AutoEvent[]) => events.filter((e) => e.type === "segmentReady");
const dues = (events: AutoEvent[]) => events.filter((e) => e.type === "sentenceDue");

// One word sign: hands present for 1000ms after an idle pre-roll.
const WORD_STEPS: Step[] = [
  { ms: 500, hand: "none" },
  { ms: 1000, hand: "right" },
  { ms: 3000, hand: "none" },
];

test("a 500ms dropout inside a sign does not split it", () => {
  const seg = new AutoSegmenter();
  const frames = buildFrames([
    { ms: 500, hand: "none" },
    { ms: 500, hand: "left" },
    { ms: 500, hand: "none" }, // 500ms dropout — shorter than gapMs (700)
    { ms: 500, hand: "left" },
    { ms: 2000, hand: "none" },
  ]);
  const segs = segments(feed(seg, frames));
  assert.equal(segs.length, 1, "sign with a 500ms hand dropout must stay one segment");
  const s = segs[0];
  assert.equal(s.type, "segmentReady");
  // Core: from the first hand frame (t=500) to the last hand frame (t=1900).
  assert.equal(s.stats.durationMs, 1400);
  assert.equal(s.stats.handFrames, 10);
});

test("an 800ms gap splits into two segments", () => {
  const seg = new AutoSegmenter();
  const frames = buildFrames([
    { ms: 500, hand: "none" },
    { ms: 800, hand: "right" },
    { ms: 800, hand: "none" }, // 800ms gap — longer than gapMs (700)
    { ms: 800, hand: "right" },
    { ms: 3000, hand: "none" },
  ]);
  const segs = segments(feed(seg, frames));
  assert.equal(segs.length, 2, "an 800ms gap must produce two segments");
  const [a, b] = segs;
  assert.equal(a.stats.durationMs, 700);
  assert.equal(b.stats.durationMs, 700);
  // Segments must not overlap in their core window.
  assert.ok(a.frames[a.frames.length - 1].tMs < b.frames[0].tMs + AUTO_SEGMENTER_DEFAULTS.padMs);
});

test("the pad frames are included in the emitted segment", () => {
  const seg = new AutoSegmenter();
  const frames = buildFrames(WORD_STEPS);
  const [s] = segments(feed(seg, frames));
  assert.ok(s, "expected one segmentReady");
  const emitted = s.frames;
  const coreStart = 500;
  const coreEnd = 1400; // last hand frame at t=1400
  // Pre-roll: frames before the core window must be present (down to padMs before start).
  assert.ok(emitted[0].tMs >= coreStart - AUTO_SEGMENTER_DEFAULTS.padMs);
  assert.ok(emitted[0].tMs < coreStart, "pre-roll frames missing");
  // Post-roll: frames after the core window must be present (up to padMs after end).
  const last = emitted[emitted.length - 1];
  assert.ok(last.tMs > coreEnd, "post-roll frames missing");
  assert.ok(last.tMs <= coreEnd + AUTO_SEGMENTER_DEFAULTS.padMs);
  // Core frames are contiguous.
  const core = emitted.filter((f) => f.tMs >= coreStart && f.tMs <= coreEnd);
  assert.equal(core.length, 10);
});

test("a segment with fewer than minHandFrames is ignored", () => {
  const seg = new AutoSegmenter();
  const frames = buildFrames([
    { ms: 400, hand: "none" },
    { ms: 300, hand: "right" }, // 3 streak frames — opens, but only 3 hand frames
    { ms: 2000, hand: "none" },
  ]);
  const events = feed(seg, frames);
  assert.equal(segments(events).length, 0, "3 hand frames must not emit a segment");
});

test("maxDurationMs force-closes a long sign", () => {
  const seg = new AutoSegmenter({ maxDurationMs: 4000 });
  const frames = buildFrames([
    { ms: 6000, hand: "right" },
    { ms: 3000, hand: "none" },
  ]);
  const segs = segments(feed(seg, frames));
  assert.ok(segs.length >= 1, "force-close must emit a segment");
  assert.equal(segs[0].stats.durationMs, 4000);
  assert.ok(segs[0].frames.every((f) => f.tMs <= 4000 + AUTO_SEGMENTER_DEFAULTS.padMs));
});

test("sentenceDue fires only after the pause and only with an accepted word, exactly once", () => {
  const mk = () =>
    buildFrames([
      { ms: 500, hand: "none" },
      { ms: 1000, hand: "right" }, // core ends at lastHand t=1400; hands absent from t=1500
      { ms: 10_000, hand: "none" },
    ]);
  // Case A: no accepted word ever -> sentenceDue must never fire.
  const segA = new AutoSegmenter();
  assert.equal(dues(feed(segA, mk())).length, 0);
  // Case B: word accepted after the close -> sentenceDue fires exactly once,
  // no earlier than end + sentencePauseMs (1400 + 2000 = 3400).
  const segB = new AutoSegmenter({ sentencePauseMs: 2000 });
  let accepted = false;
  let dueCount = 0;
  let dueT: number | null = null;
  for (const f of mk()) {
    if (!accepted && f.tMs >= 1500) {
      segB.noteWordAccepted();
      accepted = true;
    }
    for (const ev of segB.pushFrame(f)) {
      if (ev.type === "sentenceDue") {
        dueCount += 1;
        dueT = f.tMs;
      }
    }
  }
  assert.equal(dueCount, 1, "sentenceDue must fire exactly once");
  assert.ok(dueT !== null && dueT >= 3400, `sentenceDue fired too early at t=${dueT}`);
});

test("no events while hands are absent forever", () => {
  const seg = new AutoSegmenter();
  const frames = buildFrames([{ ms: 10_000, hand: "none" }]);
  const events = feed(seg, frames);
  assert.equal(events.length, 0, "absent hands must produce no events");
});

test("two words 0.8s apart give two segments", () => {
  const seg = new AutoSegmenter();
  const frames = buildFrames([
    { ms: 400, hand: "none" },
    { ms: 900, hand: "right" },
    { ms: 800, hand: "none" },
    { ms: 900, hand: "right" },
    { ms: 3000, hand: "none" },
  ]);
  const segs = segments(feed(seg, frames));
  assert.equal(segs.length, 2);
  assert.equal(segs[0].stats.handFrames, 9);
  assert.equal(segs[1].stats.handFrames, 9);
});

test("a segment shorter than the adapter minimum is padded by repeating the edge frame", () => {
  const seg = new AutoSegmenter({
    startFrames: 1,
    minHandFrames: 1,
    minDurationMs: 0,
    padMs: 0,
    gapMs: 100,
  });
  const frames = buildFrames([
    { ms: 100, hand: "right" },
    { ms: 2000, hand: "none" },
  ]);
  const segs = segments(feed(seg, frames));
  assert.equal(segs.length, 1);
  assert.equal(segs[0].frames.length, ADAPTER_MIN_FRAMES);
  assert.deepEqual(segs[0].frames[1], segs[0].frames[0], "edge frame must be repeated, not invented");
});
