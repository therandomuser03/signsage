import type { ConnectionStatus } from "@/services/websocket/connection";

/**
 * Live Interpreter status vocabulary. Four distinct states, one owner each:
 * camera preview (video element), translation session (phase), transport
 * (socket status), model availability (fused below). Nothing here performs
 * I/O; the page supplies the inputs.
 */

export type ModelState = "ready" | "unavailable" | "unknown";

/**
 * Single authoritative model-availability state. An explicit terminal verdict
 * from the server (MODEL_UNAVAILABLE) wins over the capability snapshot, so
 * the indicator stays correct even when the REST fetch failed or is stale.
 */
export function resolveModelState(opts: {
  terminalUnavailable: boolean;
  available: boolean | null;
}): ModelState {
  if (opts.terminalUnavailable) return "unavailable";
  if (opts.available === null) return "unknown";
  return opts.available ? "ready" : "unavailable";
}

export function modelLabel(state: ModelState): string {
  if (state === "ready") return "Ready";
  if (state === "unavailable") return "Model unavailable";
  return "Model status unknown";
}

/** Pure transport wording. Never mentions the model. */
export function transportLabel(
  status: ConnectionStatus,
  everOpened: boolean,
): string {
  if (status === "open") return "Connected";
  if (status === "connecting") return everOpened ? "Reconnecting…" : "Connecting";
  return "Disconnected";
}

/**
 * The transport pill is rendered only when it says something the model
 * indicator does not. A socket closed by an explicit MODEL_UNAVAILABLE
 * verdict carries no information beyond the verdict itself, so it is hidden
 * instead of shown as a misleading second "unavailable".
 */
export function shouldShowTransport(
  status: ConnectionStatus,
  modelUnavailable: boolean,
): boolean {
  if (!modelUnavailable) return true;
  return status === "open" || status === "connecting";
}

/**
 * Session surfaces (preview overlay, session bar). While recognition is
 * terminally unavailable the session is honestly a camera preview, not a
 * broken connection — so it never reads "Disconnected" in that state.
 */
export function sessionBadgeLabel(
  status: ConnectionStatus,
  opts: { modelUnavailable: boolean; everOpened: boolean },
): string {
  if (opts.modelUnavailable) return "Preview only";
  return transportLabel(status, opts.everOpened);
}

/** Idle/empty-state primary action. Never promises recognition when gated. */
export function startActionLabel(opts: {
  isError: boolean;
  modelBlocked: boolean;
  modelReady: boolean;
}): string {
  if (opts.isError) return "Retry";
  if (opts.modelBlocked) return "Start camera preview";
  if (opts.modelReady) return "Start interpreting";
  return "Start session";
}

/** Small "live translation" pill under the panel header. */
export function activityLabel(opts: {
  hasDisplay: boolean;
  showingPartial: boolean;
  running: boolean;
  modelUnavailable: boolean;
}): string {
  if (opts.hasDisplay) return opts.showingPartial ? "live · stabilizing" : "final";
  if (opts.modelUnavailable) return "unavailable";
  return opts.running ? "listening" : "waiting";
}
