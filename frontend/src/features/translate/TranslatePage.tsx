import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CircleAlert,
  Eraser,
  Pause,
  Play,
  PlugZap,
  Square,
} from "lucide-react";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { config } from "@/app/config";
import { apiClient } from "@/services/api/instance";
import { getAccessToken } from "@/services/supabase/auth";
import { TranslationSocket } from "@/services/websocket/client";
import type { ConnectionStatus } from "@/services/websocket/connection";
import type { TranslationTextMessage } from "@/services/websocket/protocol";
import type { CapabilitiesResponse } from "@/types/api";
import {
  activityLabel,
  modelLabel,
  resolveModelState,
  sessionBadgeLabel,
  shouldShowTransport,
  startActionLabel,
  transportLabel,
} from "./sessionStatus";
import { HolisticExtractor, type HolisticFrame } from "./landmarks";
import { loadPrefs } from "@/features/settings/preferences";

type Phase =
  | "idle"
  | "starting-camera"
  | "loading-detector"
  | "live"
  | "paused"
  | "error";

const SEND_INTERVAL_MS = 100; // ~10 landmark frames per second

function drawOverlay(
  canvas: HTMLCanvasElement,
  frame: HolisticFrame,
  color: string
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = color;
  const dot = (x: number, y: number, r: number) => {
    ctx.beginPath();
    ctx.arc(x * width, y * height, r, 0, Math.PI * 2);
    ctx.fill();
  };
  for (const m of frame.pose) dot(m.x, m.y, 3);
  for (const m of frame.face) dot(m.x, m.y, 1.2);
  for (const m of frame.leftHand) dot(m.x, m.y, 2);
  for (const m of frame.rightHand) dot(m.x, m.y, 2);
}

export default function TranslatePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const extractorRef = useRef<HolisticExtractor | null>(null);
  const socketRef = useRef<TranslationSocket | null>(null);
  const rafRef = useRef(0);
  const lastSendRef = useRef(0);
  const framesSentRef = useRef(0);
  const phaseRef = useRef<Phase>("idle");
  const everOpenRef = useRef(false);

  const [phase, setPhaseState] = useState<Phase>("idle");
  const [cameraMessage, setCameraMessage] = useState<string | null>(null);
  const [detectorMessage, setDetectorMessage] = useState<string | null>(null);
  const [capabilities, setCapabilities] =
    useState<CapabilitiesResponse | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [socketStatus, setSocketStatus] = useState<ConnectionStatus>("idle");
  const [socketError, setSocketError] = useState<string | null>(null);
  // Model/backend outage, kept separate from recognition outcomes: a dead
  // model is not an "uncertain prediction".
  const [modelError, setModelError] = useState<string | null>(null);
  const [results, setResults] = useState<TranslationTextMessage[]>([]);
  const [partial, setPartial] = useState<TranslationTextMessage | null>(null);
  const [framesSent, setFramesSent] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [lastSeenAt, setLastSeenAt] = useState<number | null>(null);
  const [mirrorPreview] = useState(() => loadPrefs().mirrorPreview);
  const mirrorClass = mirrorPreview ? "-scale-x-100" : "";

  const setPhase = (p: Phase) => {
    phaseRef.current = p;
    setPhaseState(p);
  };

  useEffect(() => {
    let live = true;
    apiClient
      .health()
      .then(() => live && setBackendOnline(true))
      .catch(() => live && setBackendOnline(false));
    apiClient
      .capabilities()
      .then((caps) => live && setCapabilities(caps))
      .catch(() => live && setCapabilities(null));
    return () => {
      live = false;
    };
  }, []);

  const stopTracks = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const teardown = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    extractorRef.current?.close();
    extractorRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    stopTracks();
  }, [stopTracks]);

  useEffect(() => teardown, [teardown]);

  // Session clock: ticks only while actively interpreting. Local UI state.
  useEffect(() => {
    if (phase !== "live") return;
    const id = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [phase]);

  const loop = useCallback(() => {
    rafRef.current = requestAnimationFrame(loop);
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const extractor = extractorRef.current;
    if (!video || !canvas || !extractor || video.readyState < 2) return;

    if (
      canvas.width !== video.clientWidth ||
      canvas.height !== video.clientHeight
    ) {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    }

    const frame = extractor.detect(video, performance.now());
    if (!frame) return;
    drawOverlay(canvas, frame, "#22c55e");

    if (phaseRef.current !== "live") return;
    if (frame.detected) setLastSeenAt(Date.now());

    const socket = socketRef.current;
    if (!socket || socket.status !== "open") return;
    // Skip empty frames (no body part detected → all-zero 543×3 payload):
    // saves ~30KB JSON per frame with zero information loss. Low-motion but
    // detected frames are still sent; the backend motion gate owns those.
    if (!frame.detected) return;
    if (performance.now() - lastSendRef.current < SEND_INTERVAL_MS) return;
    lastSendRef.current = performance.now();
    try {
      if (socket.sendFrame(frame.landmarks)) {
        framesSentRef.current += 1;
        setFramesSent(framesSentRef.current);
      }
    } catch {
      // Shape guard failed — never send a malformed frame.
    }
  }, []);

  const connectSocket = useCallback(() => {
    socketRef.current?.close();
    setSocketError(null);
    const socket = new TranslationSocket({
      url: config.wsUrl,
      getAccessToken,
      // Bounded auto-reconnect: the backend keeps per-connection state, so a
      // reconnect starts a fresh backend session. close() cancels retries.
      maxReconnectAttempts: 5,
      reconnectBaseDelayMs: 500,
      onStatus: (s) => {
        if (s === "open") {
          everOpenRef.current = true;
          // The server spoke to us: it is reachable even if it then
          // refuses with MODEL_UNAVAILABLE. Never report it offline.
          setBackendOnline(true);
        }
        // A reconnect starts a fresh backend session: drop the stale partial.
        if (s === "connecting" && everOpenRef.current) setPartial(null);
        setSocketStatus(s);
      },
      onTranslation: (msg) => {
        if (msg.final) {
          setPartial(null);
          setResults((prev) => [msg, ...prev]);
        } else {
          setPartial(msg);
        }
      },
      onError: (msg) => {
        setSocketError(msg.message);
        if (msg.code === "MODEL_UNAVAILABLE") {
          // Terminal verdict for this session (auto-retry already
          // suppressed in TranslationSocket): the server answered, so it
          // is online — recognition alone is disabled by the deploy gate.
          setBackendOnline(true);
          setModelError(msg.message);
        }
      },
    });
    socketRef.current = socket;
    socket.connect();
    // User-initiated (Start/Reconnect) only — never polled. Refreshes the
    // model banner in case the deployment changed since mount.
    apiClient.health().then(() => setBackendOnline(true)).catch(() => setBackendOnline(false));
    apiClient.capabilities().then(setCapabilities).catch(() => setCapabilities(null));
  }, []);

  const start = useCallback(async () => {
    if (phaseRef.current === "live" || phaseRef.current === "paused") return;
    setCameraMessage(null);
    setDetectorMessage(null);

    setPhase("starting-camera");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraMessage("This browser has no camera API.");
      setPhase("error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
    } catch {
      setCameraMessage(
        "Camera access is required to interpret signs. Grant permission and try again."
      );
      setPhase("error");
      return;
    }

    setPhase("loading-detector");
    try {
      const extractor = new HolisticExtractor();
      await extractor.init();
      extractorRef.current = extractor;
    } catch {
      extractorRef.current = null;
      setDetectorMessage(
        "Landmark detection could not start. Check your connection (model files load on demand) and try again."
      );
      setPhase("error");
      return;
    }

    framesSentRef.current = 0;
    setFramesSent(0);
    everOpenRef.current = false;
    // A new session must never present the previous session's final as
    // current: drop stale results, partials, and model errors here.
    setResults([]);
    setPartial(null);
    setSocketError(null);
    setModelError(null);
    setElapsed(0);
    lastSendRef.current = 0;
    connectSocket();
    setPhase("live");
    rafRef.current = requestAnimationFrame(loop);
  }, [connectSocket, loop]);

  const pause = useCallback(() => {
    if (phaseRef.current === "live") setPhase("paused");
  }, []);

  const resume = useCallback(() => {
    if (phaseRef.current === "paused") setPhase("live");
  }, []);

  const stop = useCallback(() => {
    const socket = socketRef.current;
    if (socket && socket.status === "open" && framesSentRef.current > 0) {
      socket.sentenceBoundary(loadPrefs().saveHistory);
    }
    cancelAnimationFrame(rafRef.current);
    extractorRef.current?.close();
    extractorRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    stopTracks();
    setPartial(null);
    setElapsed(0);
    setPhase("idle");
  }, [stopTracks]);

  const clearResults = useCallback(() => {
    setResults([]);
    setPartial(null);
    framesSentRef.current = 0;
    setFramesSent(0);
  }, []);

  const terminalUnavailable = modelError !== null;
  const running = phase === "live" || phase === "paused";
  const modelAvailable = capabilities?.sign_to_text_available ?? null;
  // One authoritative model state: the terminal server verdict wins over the
  // capability snapshot, so this stays correct when REST failed or is stale.
  const modelState = resolveModelState({
    terminalUnavailable,
    available: modelAvailable,
  });
  const modelBlocked = modelState === "unavailable";
  const showTransport = shouldShowTransport(socketStatus, terminalUnavailable);
  const transportText = transportLabel(socketStatus, everOpenRef.current);
  const sessionBadge = sessionBadgeLabel(socketStatus, {
    modelUnavailable: terminalUnavailable,
    everOpened: everOpenRef.current,
  });
  const elapsedLabel = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  const busy = phase === "starting-camera" || phase === "loading-detector";
  const latest = results[0] ?? null;
  // A finalized result wins; otherwise show the stabilizing live partial.
  const display = latest ?? partial;
  const showingPartial = latest === null && partial !== null;
  const noSignRecently =
    phase === "live" &&
    framesSent > 0 &&
    (lastSeenAt === null || Date.now() - lastSeenAt > 4000);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Live Interpreter</h1>
          <p className="mt-1 text-sm text-foreground/55">
            Sign naturally. Your movements appear as English in real time.
          </p>
        </div>
        <div className="space-y-0.5 text-right text-xs">
          <p className="flex items-center justify-end gap-1.5 font-medium">
            <span
              aria-hidden="true"
              className={
                modelState === "unknown"
                  ? "inline-block h-2 w-2 rounded-full bg-foreground/30"
                  : modelState === "ready"
                    ? "inline-block h-2 w-2 rounded-full bg-green-500"
                    : "inline-block h-2 w-2 rounded-full bg-red-500"
              }
            />
            {modelLabel(modelState)}
          </p>
          {showTransport && (
          <p className="flex items-center justify-end gap-1.5 text-foreground/60">
            <span
              aria-hidden="true"
              className={
                socketStatus === "open"
                  ? "inline-block h-2 w-2 rounded-full bg-green-500"
                  : socketStatus === "connecting"
                    ? "inline-block h-2 w-2 rounded-full bg-yellow-500 motion-safe:animate-pulse"
                    : "inline-block h-2 w-2 rounded-full bg-foreground/30"
              }
            />
            {transportText}
          </p>
          )}
        </div>
      </div>

      {modelBlocked && (
        <div className="flex items-start gap-2.5 rounded-xl border border-amber-500/25 bg-amber-500/10 px-4 py-3">
          <CircleAlert
            aria-hidden="true"
            className="mt-0.5 h-4 w-4 shrink-0 text-amber-500"
          />
          <div className="text-sm">
            <p className="font-medium">Translation model unavailable</p>
            <p className="mt-0.5 text-xs text-foreground/60">
              {capabilities?.sign_to_text_unavailable_reason ??
                "The backend reports no usable translation model right now."}{" "}
              Camera preview is still available.
            </p>
          </div>
        </div>
      )}

      {capabilities?.sign_to_text_source === "heuristic-fallback" && (
        <div className="rounded-xl border border-foreground/10 bg-card px-4 py-3 text-sm">
          <p className="font-medium">Recognizable signs (limited preview)</p>
          <p className="mt-0.5 text-xs text-foreground/60">
            Rule-based recognition of 5 static one-hand shapes — hold each steady
            toward the camera for ~2 seconds, then press Stop to finalize. Not
            general ISL translation.
          </p>
          <ul className="mt-2 grid gap-1 text-xs text-foreground/75 sm:grid-cols-2">
            <li><span className="font-medium text-foreground">Hello</span> — open palm, fingers spread</li>
            <li><span className="font-medium text-foreground">Yes</span> — thumbs up only</li>
            <li><span className="font-medium text-foreground">No</span> — closed fist</li>
            <li><span className="font-medium text-foreground">You</span> — index finger only</li>
            <li><span className="font-medium text-foreground">Help</span> — index + middle fingers</li>
          </ul>
        </div>
      )}

      {/* Workspace: camera + live translation panel */}
      <div className="grid items-stretch gap-4 lg:grid-cols-3">
      <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-foreground/10 bg-black lg:col-span-2">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`
            absolute inset-0 h-full w-full object-cover ${mirrorClass}
            ${phase === "live" || phase === "paused" ? "" : "hidden"}
          `}
        />
        <canvas
          ref={canvasRef}
          aria-hidden="true"
          className={`
            absolute inset-0 h-full w-full ${mirrorClass}
            ${phase === "live" || phase === "paused" ? "" : "hidden"}
          `}
        />
        {phase !== "live" && phase !== "paused" && (
          <div className="absolute inset-0 grid place-items-center p-6 text-center">
            {busy ? (
              <div className="flex flex-col items-center gap-3 text-white/70">
                <Spinner aria-hidden="true" className="h-8 w-8" />
                <p className="text-sm">
                  {phase === "starting-camera"
                    ? "Requesting camera…"
                    : "Initializing landmark detection…"}
                </p>
              </div>
            ) : (
              <div className="flex max-w-xs flex-col items-center text-white/70">
                <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/5">
                  <Camera aria-hidden="true" className="h-7 w-7" />
                </span>
                <p className="mt-3 text-base font-semibold text-white">
                  {modelBlocked ? "Camera preview only" : "Ready to interpret"}
                </p>
                <p className="mt-1 text-sm">
                  {phase === "error"
                    ? (cameraMessage ??
                      detectorMessage ??
                      "Something went wrong. Try again.")
                    : modelBlocked
                      ? "Translation is unavailable — the recognition model has not passed the deployment gate. Camera preview and landmark overlay will still work."
                      : "Position yourself in frame and start signing."}
                </p>
                <Button
                  onClick={() => void start()}
                  disabled={busy}
                  className="mt-5 rounded-full px-6"
                >
                  <Play aria-hidden="true" />
                  {startActionLabel({
                    isError: phase === "error",
                    modelBlocked,
                    modelReady: modelState === "ready",
                  })}
                </Button>
              </div>
            )}
          </div>
        )}
        {(phase === "live" || phase === "paused") && (
          <>
            <div className="absolute inset-x-0 top-0 flex items-start justify-between gap-2 p-4">
              <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-black/55 px-3 py-1.5 text-[11px] font-medium text-white backdrop-blur-md">
                <span
                  aria-hidden="true"
                  className={
                    phase === "live"
                      ? "inline-block h-1.5 w-1.5 rounded-full bg-green-500 motion-safe:animate-pulse"
                      : "inline-block h-1.5 w-1.5 rounded-full bg-amber-400"
                  }
                />
                {phase === "live" ? "Live" : "Paused"}
              </span>
              <span className="rounded-full border border-white/10 bg-black/55 px-3 py-1.5 text-[11px] font-medium text-white/75 backdrop-blur-md">
                {sessionBadge}
              </span>
            </div>
            {noSignRecently && (
              <div className="absolute inset-x-0 top-16 flex justify-center px-4">
                <p className="rounded-full border border-white/10 bg-black/55 px-3 py-1.5 text-[11px] text-white/75 backdrop-blur-md">
                  No sign detected — position your hands in frame.
                </p>
              </div>
            )}
          </>
        )}
        </div>
        <div className="flex min-h-64 flex-col rounded-xl border border-foreground/10 bg-card lg:min-h-0">
          <div className="border-b border-foreground/10 px-5 py-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
              Live translation
            </p>
            <p className="mt-1.5 flex items-center gap-1.5 text-xs text-foreground/60">
              <span
                aria-hidden="true"
                className={
                  display
                    ? "inline-block h-1.5 w-1.5 rounded-full bg-green-500"
                    : "inline-block h-1.5 w-1.5 rounded-full bg-foreground/30"
                }
              />
              {activityLabel({
                hasDisplay: display !== null,
                showingPartial,
                running,
                modelUnavailable: terminalUnavailable,
              })}
            </p>
          </div>
          <div className="flex flex-1 flex-col justify-center px-5 py-5">
            {modelError ? (
              <>
                <p className="text-2xl font-semibold tracking-tight text-foreground/40">
                  Translation unavailable
                </p>
                <p className="mt-2 text-xs text-foreground/55">
                  {modelError} Camera preview is still available.
                </p>
              </>
            ) : display ? (
              display.accepted ? (
                <>
                  <p className="text-2xl font-semibold tracking-tight">
                    {display.text}
                  </p>
                  <p className="mt-2 text-xs text-foreground/55">
                    {showingPartial
                      ? "Partial result from the live window — keep signing; press Stop to finalize."
                      : `accepted · ${display.result_type} · ${Math.round(display.confidence * 100)}% confidence`}
                  </p>
                </>
              ) : (
                <>
                  <p className="text-2xl font-semibold tracking-tight text-foreground/60">
                    Not confidently recognized
                  </p>
                  <p className="mt-2 text-xs text-foreground/55">
                    {display.reason} · no text returned — keep signing or
                    reposition your hands in frame.
                  </p>
                </>
              )
            ) : (
              <>
                <p className="text-2xl font-semibold tracking-tight text-foreground/40">
                  {running ? "Listening…" : "Waiting for signing…"}
                </p>
                <p className="mt-2 text-xs text-foreground/55">
                  {running
                    ? "Sign toward the camera — interpretation appears here."
                    : "Start a session and sign toward the camera."}
                </p>
              </>
            )}
          </div>
        </div>
      </div>
      {(cameraMessage || detectorMessage) && phase === "error" && (
        <Alert variant="destructive">
          <CircleAlert aria-hidden="true" />
          <AlertTitle>
            {cameraMessage ? "Camera unavailable" : "Detection unavailable"}
          </AlertTitle>
          <AlertDescription>
            {cameraMessage ?? detectorMessage}
          </AlertDescription>
        </Alert>
      )}

      {/* Session bar */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3 rounded-xl border border-foreground/10 bg-card px-4 py-3 text-xs">
        <span className="flex items-center gap-1.5 font-medium">
          <span
            aria-hidden="true"
            className={
              socketStatus === "open"
                ? "inline-block h-2 w-2 rounded-full bg-green-500"
                : socketStatus === "connecting"
                  ? "inline-block h-2 w-2 rounded-full bg-yellow-500 motion-safe:animate-pulse"
                  : "inline-block h-2 w-2 rounded-full bg-foreground/30"
            }
          />
          {sessionBadge}
        </span>
        <span className="font-medium tabular-nums" aria-label="Session time">
          {elapsedLabel}
        </span>
        <span className="text-foreground/60">
          Frames{" "}
          <span className="font-medium text-foreground tabular-nums">
            {framesSent}
          </span>{" "}
          ·{" "}
          <span className="font-medium text-foreground tabular-nums">
            {results.length}
          </span>{" "}
          results · Backend{" "}
          <span className="font-medium text-foreground">
            {backendOnline === null
              ? "Checking…"
              : backendOnline
                ? "Online"
                : "Offline"}
          </span>
        </span>
        {socketError && (
          <span className="text-destructive">{socketError}</span>
        )}
        {noSignRecently && (
          <span className="text-foreground/60">
            No body landmarks seen for a few seconds — move into frame and
            keep signing.
          </span>
        )}
        <span className="ml-auto flex items-center gap-1">
          {!running ? (
            <Button
              onClick={() => void start()}
              disabled={busy}
              size="sm"
              className="rounded-full px-4"
            >
              {busy ? <Spinner aria-hidden="true" /> : <Play aria-hidden="true" />}
              {phase === "error" ? "Retry session" : "Start session"}
            </Button>
          ) : (
            <>
              {phase === "live" ? (
                <Button variant="ghost" size="sm" onClick={pause}>
                  <Pause aria-hidden="true" />
                  Pause
                </Button>
              ) : (
                <Button size="sm" onClick={resume} className="rounded-full px-4">
                  <Play aria-hidden="true" />
                  Resume
                </Button>
              )}
              <Button
                variant="destructive"
                size="sm"
                onClick={stop}
                className="rounded-full px-4"
              >
                <Square aria-hidden="true" />
                Stop session
              </Button>
            </>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={connectSocket}
            disabled={!running || socketStatus === "connecting"}
          >
            <PlugZap aria-hidden="true" />
            Reconnect
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearResults}
            disabled={results.length === 0 && framesSent === 0}
          >
            <Eraser aria-hidden="true" />
            Clear
          </Button>
        </span>
      </div>

      {results.length > 1 && (
        <div>
          <h2 className="text-sm font-medium">Session results</h2>
          <p className="mt-0.5 text-xs text-foreground/50">
            Newest first. Local to this session.
          </p>
          <ul className="mt-2 divide-y divide-foreground/10 border-t border-foreground/10">
            {results.map((r, i) => (
              <li key={`${r.history_id ?? "live"}-${i}`} className="py-3">
                <p className="font-medium">
                  {r.accepted && r.text
                    ? r.text
                    : "Not confidently recognized"}
                </p>
                <p className="mt-0.5 text-xs text-foreground/50">
                  {r.accepted ? "accepted" : r.reason} · {r.result_type} ·{" "}
                  {Math.round(r.confidence * 100)}% confidence
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
