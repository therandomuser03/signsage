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
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { config } from "@/app/config";
import { apiClient } from "@/services/api/instance";
import { getAccessToken } from "@/services/supabase/auth";
import { TranslationSocket } from "@/services/websocket/client";
import type { ConnectionStatus } from "@/services/websocket/connection";
import type { TranslationTextMessage } from "@/services/websocket/protocol";
import type {
  CapabilitiesResponse,
  ImageClassifyResponse,
  VideoTranslationResponse,
} from "@/types/api";
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
import { BrowseSignsSheet } from "./gestureCatalog";
import { loadPrefs } from "@/features/settings/preferences";

type Phase =
  | "idle"
  | "starting-camera"
  | "loading-detector"
  | "live"
  | "paused"
  | "error";

const SEND_INTERVAL_MS = 100; // ~10 landmark frames per second
const DETECT_INTERVAL_MS = 80; // MediaPipe runs far faster than the backend can use; keep preview responsive without re-running all models every animation frame.

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
  const lastDetectRef = useRef(0);
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
  const [browseOpen, setBrowseOpen] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [lastSeenAt, setLastSeenAt] = useState<number | null>(null);
  /** Ordered sequence of accepted sign words from finalized results.
   * Built only from `final:true + accepted` messages; the backend's
   * repeat-agreement stabilization already prevents flooding from a held
   * gesture, so each entry here is a deliberate finalized sign. */
  const [acceptedSigns, setAcceptedSigns] = useState<string[]>([]);
  const [mirrorPreview] = useState(() => loadPrefs().mirrorPreview);
  const mirrorClass = mirrorPreview ? "-scale-x-100" : "";
  const isolatedWordModel = capabilities?.sign_to_text_source === "isolated-word";
  const scoreLabel = capabilities?.sign_to_text_source === "heuristic-fallback"
    ? "frame agreement"
    : "uncalibrated model score";

  // --- Static image classification (REST, alphabet / word) ---
  const [imageClassifyMode, setImageClassifyMode] = useState(false);
  const [imageResult, setImageResult] = useState<ImageClassifyResponse | null>(null);
  const [imageClassifying, setImageClassifying] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);
  const imageIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastClassifyRef = useRef(0);
  const IMAGE_CLASSIFY_INTERVAL_MS = 1500; // classify at most ~1.5s apart

  // --- Video upload for isolated-word recognition ---
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUploadLoading, setVideoUploadLoading] = useState(false);
  const [videoUploadError, setVideoUploadError] = useState<string | null>(null);
  const [videoUploadResult, setVideoUploadResult] = useState<VideoTranslationResponse | null>(null);

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

    const now = performance.now();
    const shouldDetect = now - lastDetectRef.current >= DETECT_INTERVAL_MS;
    let frame: HolisticFrame | null = null;
    if (shouldDetect) {
      frame = extractor.detect(video, now);
      if (frame) {
        lastDetectRef.current = now;
        drawOverlay(canvas, frame, "#22c55e");
      }
    }

    if (!frame) return;

    if (phaseRef.current !== "live") return;
    if (frame.detected) setLastSeenAt(Date.now());

    const socket = socketRef.current;
    if (!socket || socket.status !== "open") return;
    // Skip empty frames (no body part detected → all-zero 543×3 payload):
    // saves ~30KB JSON per frame with zero information loss. Low-motion but
    // detected frames are still sent; the backend motion gate owns those.
    if (!frame.detected) return;
    if (now - lastSendRef.current < SEND_INTERVAL_MS) return;
    lastSendRef.current = now;
    try {
      if (socket.sendFrame(frame.landmarks)) {
        framesSentRef.current += 1;
        setFramesSent(framesSentRef.current);
      }
    } catch {
      // Shape guard failed — never send a malformed frame.
    }
  }, []);

  // --- Static image classification: capture a video frame and classify it ---
  const classifyCurrentFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    setImageClassifying(true);
    setImageError(null);
    try {
      // Draw the current video frame to an offscreen canvas, then base64-encode it.
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Cannot get canvas context");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.85);

      const result = await apiClient.classifyImage({
        image: dataUrl,
        task: "alphabet",
        min_confidence: 0.5,
      });
      setImageResult(result);
    } catch (err: any) {
      setImageError(err?.message ?? "Image classification failed.");
      setImageResult(null);
    } finally {
      setImageClassifying(false);
    }
  }, [apiClient]);

  // Periodically classify frames while image classification mode is active and
  // the camera is live. Stopped automatically when the mode is toggled off.
  useEffect(() => {
    if (!imageClassifyMode || phaseRef.current !== "live") {
      if (imageIntervalRef.current) {
        clearInterval(imageIntervalRef.current);
        imageIntervalRef.current = null;
      }
      return;
    }
    imageIntervalRef.current = setInterval(() => {
      const now = Date.now();
      if (now - lastClassifyRef.current >= IMAGE_CLASSIFY_INTERVAL_MS) {
        lastClassifyRef.current = now;
        void classifyCurrentFrame();
      }
    }, IMAGE_CLASSIFY_INTERVAL_MS);
    return () => {
      if (imageIntervalRef.current) {
        clearInterval(imageIntervalRef.current);
        imageIntervalRef.current = null;
      }
    };
  }, [imageClassifyMode, classifyCurrentFrame, phaseRef.current]);

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
          if (msg.accepted && msg.text) {
            setAcceptedSigns((prev) => [...prev, msg.text]);
          }
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
    lastDetectRef.current = 0;
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
    lastDetectRef.current = 0;
    setPhase("idle");
  }, [stopTracks]);

  const clearResults = useCallback(() => {
    setResults([]);
    setPartial(null);
    setAcceptedSigns([]);
    framesSentRef.current = 0;
    setFramesSent(0);
  }, []);

  const undoLastSign = useCallback(() => {
    setAcceptedSigns((prev) => (prev.length > 0 ? prev.slice(0, -1) : prev));
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
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="rounded-full px-4"
            onClick={() => setBrowseOpen(true)}
          >
            Browse Signs
          </Button>
          <Button
            type="button"
            variant={imageClassifyMode ? "default" : "outline"}
            size="sm"
            className="rounded-full px-4"
            onClick={() => {
              setImageClassifyMode((prev) => {
                const next = !prev;
                if (!next) {
                  setImageResult(null);
                  setImageError(null);
                }
                return next;
              });
            }}
            disabled={!backendOnline}
          >
            {imageClassifyMode ? <Square size={14} /> : <Camera size={14} />}
            {imageClassifyMode ? " Image" : " Image classify"}
          </Button>
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

      {isolatedWordModel && (
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm">
          <p className="font-medium">Limited isolated-word ISL pilot</p>
          <p className="mt-0.5 text-xs text-foreground/60">
            Recognizes a small vocabulary of isolated words; it does not
            translate continuous signing or sentences. Model score is
            uncalibrated and may be wrong.
          </p>
        </div>
      )}

      {imageClassifyMode && (
        <div className="rounded-xl border border-foreground/10 bg-card px-4 py-3 text-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50 mb-2">
            Image classification (alphabet)
          </p>
          {imageError && (
            <div className="text-destructive text-xs mb-2">{imageError}</div>
          )}
          {imageClassifying ? (
            <div className="flex items-center gap-3 text-xs text-foreground/60">
              <Spinner className="h-4 w-4" />
              <span>Classifying…</span>
            </div>
          ) : imageResult ? (
            <div className="flex items-center justify-between">
              <span className={imageResult.accepted ? "font-mono text-2xl font-bold text-foreground" : "font-mono text-2xl font-bold text-foreground/40"}>
                {imageResult.label || "—"}
              </span>
              <div className="text-right text-xs">
                <p className="font-medium">
                  {imageResult.accepted ? "accepted" : imageResult.reason}
                </p>
                <p className="text-foreground/50">
                  confidence {imageResult.confidence.toFixed(2)}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-xs text-foreground/50">
              Position your hand in frame and hold the sign steady.
            </p>
          )}
          <p className="mt-2 text-xs text-foreground/50">
            Static ISL letters (A–Z) only — not continuous signing or sentences.
          </p>
        </div>
      )}

      {/* Video upload for isolated-word recognition */}
      <div className="rounded-xl border border-foreground/10 bg-card px-4 py-3 text-sm">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50 mb-2">
          Video upload (isolated-word pilot)
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="file"
            accept=".mp4,.mov,.avi,.mkv,.webm"
            id="video-upload"
            className="text-xs"
            onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
          />
          <Button
            size="sm"
            variant="outline"
            disabled={!videoFile || videoUploadLoading}
            onClick={async () => {
              if (!videoFile) return;
              setVideoUploadLoading(true);
              setVideoUploadError(null);
              try {
                const result = await apiClient.uploadVideo(videoFile);
                setVideoUploadResult(result);
              } catch (err: any) {
                setVideoUploadError(err?.message ?? "Video upload failed.");
              } finally {
                setVideoUploadLoading(false);
              }
            }}
          >
            {videoUploadLoading ? <Spinner className="h-3 w-3 mr-1" /> : null}
            Process video
          </Button>
        </div>
        {videoUploadError && (
          <p className="mt-2 text-xs text-destructive">{videoUploadError}</p>
        )}
        {videoUploadResult && (
          <div className="mt-2 rounded-md bg-foreground/5 px-3 py-2">
            <p className="text-sm font-medium">
              {videoUploadResult.accepted ? videoUploadResult.text : "Not confidently recognized"}
            </p>
            <p className="mt-1 text-xs text-foreground/60">
              {videoUploadResult.accepted ? "accepted" : videoUploadResult.reason} ·
              confidence {videoUploadResult.confidence.toFixed(2)} ·
              frames {videoUploadResult.frames_processed} ·
              {videoUploadResult.feature_version} ·
              {videoUploadResult.result_type}
            </p>
          </div>
        )}
        <p className="mt-2 text-[10px] text-foreground/40">
          Experimental six-word vocabulary (friend, hello, hospital, market, school, thank you). Uncalibrated scores. Uploads are not retained.
        </p>
      </div>

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
                      : `accepted · ${display.result_type} · ${scoreLabel} ${display.confidence.toFixed(2)}`}
                  </p>
                </>
              ) : (
                <>
                  <p className="text-2xl font-semibold tracking-tight text-foreground/60">
                    Not confidently recognized
                  </p>
                  <p className="mt-2 text-xs text-foreground/55">
                    {display.reason} · {scoreLabel} {display.confidence.toFixed(2)} · no text returned — keep signing or
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
            disabled={results.length === 0 && framesSent === 0 && acceptedSigns.length === 0}
          >
            <Eraser aria-hidden="true" />
            Clear
          </Button>
        </span>
      </div>

      {acceptedSigns.length > 0 && (
        <div className="rounded-xl border border-foreground/10 bg-card px-4 py-3 text-xs">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50 mb-2">
            Accepted signs
          </p>
          <div className="flex flex-wrap gap-2">
            {acceptedSigns.map((sign, i) => (
              <Badge key={i} variant="secondary" className="px-3 py-1.5 text-sm">
                {sign}
              </Badge>
            ))}
            <Button
              variant="ghost"
              size="sm"
              onClick={undoLastSign}
              disabled={acceptedSigns.length === 0}
              title="Remove the last accepted sign"
            >
              Undo last
            </Button>
          </div>
          <p className="mt-2 text-xs text-foreground/50">
            Built from finalized recognition results. Ordered left to right.
            Clear removes the whole sequence.
          </p>
        </div>
      )}

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
                  {scoreLabel} {r.confidence.toFixed(2)}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}

      <BrowseSignsSheet open={browseOpen} onOpenChange={setBrowseOpen} />
    </div>
  );
}
