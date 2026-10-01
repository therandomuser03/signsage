import { useCallback, useEffect, useRef, useState } from "react";
import {
  CameraOff,
  CircleAlert,
  Eraser,
  Languages,
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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { config } from "@/app/config";
import { apiClient } from "@/services/api/instance";
import { getAccessToken } from "@/services/supabase/auth";
import { TranslationSocket } from "@/services/websocket/client";
import type { ConnectionStatus } from "@/services/websocket/connection";
import type {
  CapabilitiesResponse,
  TranslationTextMessage,
} from "@/services/websocket/protocol";
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

  const [phase, setPhaseState] = useState<Phase>("idle");
  const [cameraMessage, setCameraMessage] = useState<string | null>(null);
  const [detectorMessage, setDetectorMessage] = useState<string | null>(null);
  const [capabilities, setCapabilities] =
    useState<CapabilitiesResponse | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [socketStatus, setSocketStatus] = useState<ConnectionStatus>("idle");
  const [socketError, setSocketError] = useState<string | null>(null);
  const [results, setResults] = useState<TranslationTextMessage[]>([]);
  const [framesSent, setFramesSent] = useState(0);
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
      onStatus: setSocketStatus,
      onTranslation: (msg) => setResults((prev) => [msg, ...prev]),
      onError: (msg) => setSocketError(msg.message),
    });
    socketRef.current = socket;
    socket.connect();
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
    setPhase("idle");
  }, [stopTracks]);

  const clearResults = useCallback(() => {
    setResults([]);
    framesSentRef.current = 0;
    setFramesSent(0);
  }, []);

  const running = phase === "live" || phase === "paused";
  const busy = phase === "starting-camera" || phase === "loading-detector";
  const latest = results[0] ?? null;
  const modelAvailable = capabilities?.sign_to_text_available ?? null;
  const noSignRecently =
    phase === "live" &&
    framesSent > 0 &&
    (lastSeenAt === null || Date.now() - lastSeenAt > 4000);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Sign → Text</h1>
          <p className="mt-1 text-sm text-foreground/55">
            Experimental interpretation. Verify before acting on results.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">
            {modelAvailable === null
              ? "Model status unknown"
              : modelAvailable
                ? "Live model"
                : "Model unavailable"}
          </Badge>
          <Badge
            variant={
              socketStatus === "open"
                ? "default"
                : socketStatus === "connecting"
                  ? "secondary"
                  : "outline"
            }
          >
            <span
              aria-hidden="true"
              className={
                socketStatus === "open"
                  ? "mr-1.5 inline-block h-2 w-2 rounded-full bg-green-500"
                  : socketStatus === "connecting"
                    ? "mr-1.5 inline-block h-2 w-2 rounded-full bg-yellow-500 motion-safe:animate-pulse"
                    : "mr-1.5 inline-block h-2 w-2 rounded-full bg-foreground/30"
              }
            />
            {socketStatus === "open"
              ? "Connected"
              : socketStatus === "connecting"
                ? "Connecting"
                : "Disconnected"}
          </Badge>
        </div>
      </div>

      {modelAvailable === false && (
        <Alert>
          <CircleAlert aria-hidden="true" />
          <AlertTitle>Translation model unavailable</AlertTitle>
          <AlertDescription>
            {capabilities?.sign_to_text_unavailable_reason ??
              "The backend reports no usable translation model right now."}{" "}
            You can still run the camera and landmark overlay, but no
            interpretation will arrive.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardContent className="pt-6">
            <div className="relative grid aspect-video place-items-center overflow-hidden rounded-xl bg-muted">
              {(phase === "live" || phase === "paused") && (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`absolute inset-0 h-full w-full object-cover ${mirrorClass}`}
                  />
                  <canvas
                    ref={canvasRef}
                    aria-hidden="true"
                    className={`absolute inset-0 h-full w-full ${mirrorClass}`}
                  />
                </>
              )}
              {phase === "idle" && (
                <video ref={videoRef} className="hidden" />
              )}
              {phase !== "live" && phase !== "paused" && (
                <div className="flex max-w-xs flex-col items-center gap-2 p-6 text-center text-foreground/40">
                  {busy ? (
                    <>
                      <Spinner aria-hidden="true" className="h-8 w-8" />
                      <p className="text-sm">
                        {phase === "starting-camera"
                          ? "Requesting camera…"
                          : "Initializing landmark detection…"}
                      </p>
                    </>
                  ) : (
                    <>
                      <CameraOff aria-hidden="true" className="h-10 w-10" />
                      <p className="text-sm">
                        {cameraMessage ??
                          detectorMessage ??
                          "Press Start to open the camera and begin a session."}
                      </p>
                    </>
                  )}
                </div>
              )}
              {phase === "paused" && (
                <span className="absolute left-3 top-3">
                  <Badge variant="secondary">Paused</Badge>
                </span>
              )}
            </div>
            {(cameraMessage || detectorMessage) && phase === "error" && (
              <Alert variant="destructive" className="mt-4">
                <CircleAlert aria-hidden="true" />
                <AlertTitle>
                  {cameraMessage ? "Camera unavailable" : "Detection unavailable"}
                </AlertTitle>
                <AlertDescription>
                  {cameraMessage ?? detectorMessage}
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Current interpretation</CardTitle>
              <CardDescription>
                Latest result on this connection.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!latest ? (
                <Empty>
                  <EmptyHeader>
                    <EmptyMedia variant="icon">
                      <Languages aria-hidden="true" className="h-6 w-6" />
                    </EmptyMedia>
                    <EmptyTitle>No sign detected</EmptyTitle>
                    <EmptyDescription>
                      {running
                        ? "Sign toward the camera to begin."
                        : "Start a session and interpretation appears here."}
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : (
                <div className="space-y-3">
                  <p className="text-4xl font-semibold tracking-tight">
                    {latest.text}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant={latest.accepted ? "default" : "secondary"}>
                      {latest.accepted ? "accepted" : latest.reason}
                    </Badge>
                    <Badge variant="outline">{latest.result_type}</Badge>
                    <Badge variant="outline">
                      {Math.round(latest.confidence * 100)}% confidence
                    </Badge>
                  </div>
                  <p className="text-xs text-foreground/50">
                    Confidence is reported by the backend for this result —
                    not a guarantee.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Session</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-foreground/60">Frames sent</span>
                <span className="font-medium tabular-nums">{framesSent}</span>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <span className="text-foreground/60">Results</span>
                <span className="font-medium tabular-nums">{results.length}</span>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <span className="text-foreground/60">Backend</span>
                <span className="font-medium">
                  {backendOnline === null
                    ? "Checking…"
                    : backendOnline
                      ? "Online"
                      : "Offline"}
                </span>
              </div>
              {socketError && (
                <p className="text-xs leading-5 text-destructive">{socketError}</p>
              )}
              {noSignRecently && (
                <p className="text-xs leading-5 text-foreground/55">
                  No body landmarks seen for a few seconds — move into
                  frame and keep signing.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-center gap-2 pt-6">
          {!running ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <span>
                  <Button onClick={() => void start()} disabled={busy}>
                    {busy ? <Spinner aria-hidden="true" /> : <Play aria-hidden="true" />}
                    {phase === "error" ? "Retry" : "Start"}
                  </Button>
                </span>
              </TooltipTrigger>
              <TooltipContent>
                Open the camera, load landmark detection, and connect.
              </TooltipContent>
            </Tooltip>
          ) : (
            <>
              {phase === "live" ? (
                <Button variant="outline" onClick={pause}>
                  <Pause aria-hidden="true" />
                  Pause
                </Button>
              ) : (
                <Button onClick={resume}>
                  <Play aria-hidden="true" />
                  Resume
                </Button>
              )}
              <Button variant="outline" onClick={stop}>
                <Square aria-hidden="true" />
                Stop
              </Button>
            </>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Button
                  variant="outline"
                  onClick={clearResults}
                  disabled={results.length === 0 && framesSent === 0}
                >
                  <Eraser aria-hidden="true" />
                  Clear
                </Button>
              </span>
            </TooltipTrigger>
            <TooltipContent>Clears this session’s local results.</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Button
                  variant="ghost"
                  onClick={connectSocket}
                  disabled={!running || socketStatus === "connecting"}
                >
                  <PlugZap aria-hidden="true" />
                  Reconnect
                </Button>
              </span>
            </TooltipTrigger>
            <TooltipContent>Drop and re-open the live connection.</TooltipContent>
          </Tooltip>
        </CardContent>
      </Card>

      {results.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Session results</CardTitle>
            <CardDescription>Newest first. Local to this session.</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="divide-y divide-foreground/10">
              {results.map((r, i) => (
                <li key={`${r.history_id ?? "live"}-${i}`} className="py-3 first:pt-0 last:pb-0">
                  <p className="font-medium">{r.text}</p>
                  <p className="mt-0.5 text-xs text-foreground/50">
                    {r.accepted ? "accepted" : r.reason} · {r.result_type} ·{" "}
                    {Math.round(r.confidence * 100)}% confidence
                  </p>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
