import { test } from "node:test";
import assert from "node:assert/strict";
import {
  activityLabel,
  modelLabel,
  resolveModelState,
  sessionBadgeLabel,
  shouldShowTransport,
  startActionLabel,
  transportLabel,
} from "./sessionStatus";

test("terminal verdict wins: one authoritative unavailable state", () => {
  // The reported glitch: capabilities said "unknown" while the server had
  // already refused — the fused state must still read unavailable.
  assert.equal(
    resolveModelState({ terminalUnavailable: true, available: null }),
    "unavailable",
  );
  assert.equal(
    resolveModelState({ terminalUnavailable: true, available: true }),
    "unavailable",
  );
  assert.equal(
    resolveModelState({ terminalUnavailable: false, available: null }),
    "unknown",
  );
  assert.equal(
    resolveModelState({ terminalUnavailable: false, available: false }),
    "unavailable",
  );
  assert.equal(
    resolveModelState({ terminalUnavailable: false, available: true }),
    "ready",
  );
  assert.equal(modelLabel("ready"), "Ready");
  assert.equal(modelLabel("unavailable"), "Model unavailable");
  assert.equal(modelLabel("unknown"), "Model status unknown");
});

test("transport pill hides only when it would duplicate the verdict", () => {
  // Terminal closed/idle: hidden — showing "Disconnected" there was the
  // misleading second indicator.
  assert.equal(shouldShowTransport("closed", true), false);
  assert.equal(shouldShowTransport("idle", true), false);
  // Genuinely different information: still shown.
  assert.equal(shouldShowTransport("open", true), true);
  assert.equal(shouldShowTransport("connecting", true), true);
  assert.equal(shouldShowTransport("closed", false), true);
  assert.equal(shouldShowTransport("idle", false), true);
  assert.equal(shouldShowTransport("connecting", false), true);
});

test("transport wording never mentions the model", () => {
  assert.equal(transportLabel("open", false), "Connected");
  assert.equal(transportLabel("connecting", false), "Connecting");
  assert.equal(transportLabel("connecting", true), "Reconnecting…");
  assert.equal(transportLabel("closed", true), "Disconnected");
  assert.equal(transportLabel("idle", false), "Disconnected");
});

test("session surfaces read preview-only instead of a failure", () => {
  assert.equal(
    sessionBadgeLabel("closed", { modelUnavailable: true, everOpened: false }),
    "Preview only",
  );
  assert.equal(
    sessionBadgeLabel("idle", { modelUnavailable: true, everOpened: false }),
    "Preview only",
  );
  // No verdict: honest transport wording.
  assert.equal(
    sessionBadgeLabel("closed", { modelUnavailable: false, everOpened: false }),
    "Disconnected",
  );
  assert.equal(
    sessionBadgeLabel("open", { modelUnavailable: false, everOpened: true }),
    "Connected",
  );
  assert.equal(
    sessionBadgeLabel("connecting", { modelUnavailable: false, everOpened: true }),
    "Reconnecting…",
  );
});

test("start action never promises recognition when gated", () => {
  // Camera error path keeps its retry wording.
  assert.equal(
    startActionLabel({ isError: true, modelBlocked: true, modelReady: false }),
    "Retry",
  );
  assert.equal(
    startActionLabel({ isError: false, modelBlocked: true, modelReady: false }),
    "Start camera preview",
  );
  assert.equal(
    startActionLabel({ isError: false, modelBlocked: false, modelReady: true }),
    "Start interpreting",
  );
  assert.equal(
    startActionLabel({ isError: false, modelBlocked: false, modelReady: false }),
    "Start session",
  );
});

test("activity pill never claims listening when recognition is disabled", () => {
  assert.equal(
    activityLabel({ hasDisplay: false, showingPartial: false, running: true, modelUnavailable: true }),
    "unavailable",
  );
  assert.equal(
    activityLabel({ hasDisplay: false, showingPartial: false, running: true, modelUnavailable: false }),
    "listening",
  );
  assert.equal(
    activityLabel({ hasDisplay: false, showingPartial: false, running: false, modelUnavailable: false }),
    "waiting",
  );
  assert.equal(
    activityLabel({ hasDisplay: true, showingPartial: true, running: true, modelUnavailable: false }),
    "live · stabilizing",
  );
  assert.equal(
    activityLabel({ hasDisplay: true, showingPartial: false, running: true, modelUnavailable: false }),
    "final",
  );
});
