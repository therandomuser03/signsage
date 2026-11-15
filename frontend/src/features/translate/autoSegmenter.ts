// Auto mode (hands-free) segmentation for the Word recognition tab.
// Pure module — no React, no DOM, no network. Driven entirely by
// timestamped landmark frames so it is fully testable with synthetic input.
//
// Usage: create an AutoSegmenter, feed every detected camera frame via
// pushFrame({ tMs, landmarks }). It emits:
//   { type: "segmentReady", frames, stats } — a sign segment (pad included)
//   { type: "sentenceDue" }               — pause elapsed after >= 1 accepted word
// The caller must call noteWordAccepted() when a /word response is accepted,
// and resetAcceptedWords() after a sentence is composed.

import type { LandmarkPoint } from "./landmarks";

export interface AutoFrame {
  tMs: number;
  landmarks: ReadonlyArray<LandmarkPoint>;
}

export interface AutoSegmenterConfig {
  /** Pre-roll and post-roll padding around a sign segment, ms. */
  padMs: number;
  /** Consecutive hands-present frames required to open a segment. */
  startFrames: number;
  /** Continuous absence of hands that closes an open segment, ms. */
  gapMs: number;
  /** Minimum hands-present frames inside a segment (rest is ignored). */
  minHandFrames: number;
  /** Minimum core duration of a segment, ms. */
  minDurationMs: number;
  /** Force-close an open segment after this long, ms. */
  maxDurationMs: number;
  /** Pause (hands absent, >= 1 accepted word) after the last segment closed that fires sentenceDue. */
  sentencePauseMs: number;
}

export const AUTO_SEGMENTER_DEFAULTS: AutoSegmenterConfig = {
  padMs: 400,
  startFrames: 3,
  gapMs: 700,
  minHandFrames: 6,
  minDurationMs: 400,
  maxDurationMs: 4000,
  sentencePauseMs: 2000,
};

/** The /word adapter refuses fewer than this many frames; pad by repeating edge frames. */
export const ADAPTER_MIN_FRAMES = 2;

export interface SegmentStats {
  /** Total frames handed to the caller (pre-roll + core + post-roll). */
  frames: number;
  /** Hands-present frames inside the core segment. */
  handFrames: number;
  /** Core duration (end - start), ms. */
  durationMs: number;
}

export type AutoSegmenterPhase = "waiting" | "signing" | "closing";

export type AutoEvent =
  | { type: "segmentReady"; frames: AutoFrame[]; stats: SegmentStats }
  | { type: "sentenceDue" };

// Hand blocks: LEFT_HAND 501-521, RIGHT_HAND 522-542 (inclusive; half-open 501-522 / 522-543).
const LEFT_HAND_START = 501;
const RIGHT_HAND_END = 543;

function handsPresent(landmarks: ReadonlyArray<LandmarkPoint>): boolean {
  for (let i = LEFT_HAND_START; i < RIGHT_HAND_END; i += 1) {
    // Covers both hand blocks (501-521 and 522-542).
    const p = landmarks[i];
    if (!p) continue; // missing landmarks read as zeros (part not detected)
    if (p[0] !== 0 || p[1] !== 0 || p[2] !== 0) return true;
  }
  return false;
}

export class AutoSegmenter {
  private cfg: AutoSegmenterConfig;
  private phase: AutoSegmenterPhase = "waiting";

  /** Frames kept while waiting, used for pre-roll (trimmed to padMs). */
  private recent: AutoFrame[] = [];
  /** Pre-roll snapshot captured when a segment opens. */
  private preRoll: AutoFrame[] = [];
  /** Core frames of the open segment (from segment start to close). */
  private segFrames: AutoFrame[] = [];
  /** Post-roll frames collected after a max-duration force-close. */
  private closingTail: AutoFrame[] = [];
  /** Core segment bookkeeping. */
  private segStartMs = 0;
  private lastHandMs = 0;
  private handFrames = 0;
  private closeEndMs = 0;
  private closeEmitAtMs = 0;
  /** Streak tracking while waiting. */
  private consecHand = 0;
  private runStartMs = 0;
  /** Sentence-pause bookkeeping. */
  private lastSegmentCloseMs: number | null = null;
  private acceptedWords = 0;
  private sentenceDueEmitted = false;
  private handsNow = false;

  constructor(config: Partial<AutoSegmenterConfig> = {}) {
    this.cfg = { ...AUTO_SEGMENTER_DEFAULTS, ...config };
  }

  updateConfig(config: Partial<AutoSegmenterConfig>): void {
    this.cfg = { ...this.cfg, ...config };
  }

  /** Current high-level phase of the segmenter. */
  getPhase(): AutoSegmenterPhase {
    return this.phase;
  }

  /** Call when a /word response for a segment was accepted. */
  noteWordAccepted(): void {
    this.acceptedWords += 1;
  }

  /** Call after a sentence is composed from the accepted words. */
  resetAcceptedWords(): void {
    this.acceptedWords = 0;
    this.sentenceDueEmitted = false;
  }

  /**
   * Milliseconds until sentenceDue would fire, or null when it cannot fire
   * (a segment is open/pending, no accepted word yet, due already emitted,
   * or hands are currently present).
   */
  getSentenceCountdownMs(nowMs: number): number | null {
    if (this.phase !== "waiting") return null;
    if (this.handsNow || this.lastSegmentCloseMs === null) return null;
    if (this.acceptedWords === 0 || this.sentenceDueEmitted) return null;
    const remaining = this.cfg.sentencePauseMs - (nowMs - this.lastSegmentCloseMs);
    return remaining > 0 ? remaining : 0;
  }

  /** Feed one camera frame. Returns any events emitted by this frame. */
  pushFrame(frame: AutoFrame): AutoEvent[] {
    const events: AutoEvent[] = [];
    const { tMs, landmarks } = frame;
    const hands = handsPresent(landmarks);
    this.handsNow = hands;

    if (this.phase === "closing") {
      this.closingTail.push(frame);
      if (tMs >= this.closeEmitAtMs) {
        const ev = this.finalizeSegment();
        if (ev) events.push(ev);
        this.phase = "waiting";
        this.consecHand = 0;
        this.closingTail = [];
      }
    } else if (this.phase === "signing") {
      this.segFrames.push(frame);
      if (hands) {
        this.handFrames += 1;
        this.lastHandMs = tMs;
      }
      if (tMs - this.lastHandMs >= this.cfg.gapMs) {
        // Gap close: post-roll (padMs < gapMs) is already inside segFrames.
        this.beginClose(this.lastHandMs, events);
      } else if (tMs - this.segStartMs >= this.cfg.maxDurationMs) {
        // Force close: post-roll frames arrive after this one.
        this.beginClose(tMs, events);
      }
    } else {
      // waiting
      this.recent.push(frame);
      const cutoff = tMs - this.cfg.padMs;
      while (this.recent.length > 1 && this.recent[0].tMs < cutoff) this.recent.shift();

      if (hands) {
        if (this.consecHand === 0) this.runStartMs = tMs;
        this.consecHand += 1;
      } else {
        this.consecHand = 0;
      }
      if (this.consecHand >= this.cfg.startFrames) {
        // Open a segment.
        this.phase = "signing";
        this.segStartMs = this.runStartMs;
        this.preRoll = this.recent.filter((f) => f.tMs >= this.segStartMs - this.cfg.padMs && f.tMs < this.segStartMs);
        this.segFrames = this.recent.filter((f) => f.tMs >= this.segStartMs);
        this.recent = [];
        this.handFrames = 0;
        this.lastHandMs = -Infinity;
        for (const f of this.segFrames) {
          if (handsPresent(f.landmarks)) {
            this.handFrames += 1;
            this.lastHandMs = f.tMs;
          }
        }
      }
    }

    if (this.phase !== "signing" && !hands && this.lastSegmentCloseMs !== null) {
      const sinceClose = tMs - this.lastSegmentCloseMs;
      if (
        !this.sentenceDueEmitted &&
        this.acceptedWords > 0 &&
        sinceClose >= this.cfg.sentencePauseMs
      ) {
        this.sentenceDueEmitted = true;
        events.push({ type: "sentenceDue" });
      }
    }

    return events;
  }

  /** Switch to closing state; finalize immediately when the post-roll is already buffered. */
  private beginClose(endMs: number, events: AutoEvent[]): void {
    this.phase = "closing";
    this.closeEndMs = endMs;
    this.closeEmitAtMs = endMs + this.cfg.padMs;
    if (this.lastPushedTMs() >= this.closeEmitAtMs) {
      // Post-roll already fully buffered (gap close) — finalize now.
      const ev = this.finalizeSegment();
      if (ev) events.push(ev);
      this.phase = "waiting";
      this.consecHand = 0;
    }
  }

  private lastPushedTMs(): number {
    const frames = this.segFrames;
    return frames.length > 0 ? frames[frames.length - 1].tMs : this.closeEndMs;
  }

  /** Seed the waiting-phase buffer from the tail of the frames we just emitted. */
  private seedRecent(frames: AutoFrame[]): void {
    if (frames.length === 0) {
      this.recent = [];
      return;
    }
    const tailStart = frames[frames.length - 1].tMs - this.cfg.padMs;
    this.recent = frames.filter((f) => f.tMs >= tailStart);
  }

  private finalizeSegment(): AutoEvent | null {
    const coreEnd = this.closeEndMs;
    const inWindow = (f: AutoFrame) =>
      f.tMs >= this.segStartMs - this.cfg.padMs && f.tMs <= coreEnd + this.cfg.padMs;
    const frames = [...this.preRoll, ...this.segFrames, ...this.closingTail].filter(inWindow);

    const handFrames = this.handFrames;
    const durationMs = coreEnd - this.segStartMs;
    this.lastSegmentCloseMs = coreEnd;
    this.sentenceDueEmitted = false;

    this.preRoll = [];
    this.segFrames = [];
    this.closingTail = [];
    this.handFrames = 0;

    if (
      handFrames < this.cfg.minHandFrames ||
      durationMs < this.cfg.minDurationMs
    ) {
      // Not enough signal — ignore the segment entirely, but still seed the
      // pre-roll buffer for the next one.
      this.seedRecent(frames);
      return null;
    }

    // Never hand the adapter fewer than ADAPTER_MIN_FRAMES frames; pad by
    // repeating the edge frame (never by inventing motion).
    while (frames.length < ADAPTER_MIN_FRAMES && frames.length > 0) {
      frames.push({ tMs: frames[frames.length - 1].tMs, landmarks: frames[frames.length - 1].landmarks });
    }

    this.seedRecent(frames);

    return {
      type: "segmentReady",
      frames,
      stats: {
        frames: frames.length,
        handFrames,
        durationMs,
      },
    };
  }
}
