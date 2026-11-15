import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CircleAlert,
  Eraser,
  Info,
  Pause,
  Play,
  LifeBuoy,
  Loader2,
  PlugZap,
  Square,
  Volume2,
  VolumeX,
} from "lucide-react";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { config } from "@/app/config";
import { apiClient, NetworkError } from "@/services/api/instance";
import { ApiError } from "@/services/api/errors";
import { getAccessToken, onAuthChange } from "@/services/supabase/auth";
import { TranslationSocket } from "@/services/websocket/client";
import type { ConnectionStatus } from "@/services/websocket/connection";
import type { TranslationTextMessage } from "@/services/websocket/protocol";
import type {
  CapabilitiesResponse,
  ImageClassifyResponse,
  LandmarkFramePayload,
  VideoTranslationResponse,
  WordRecognitionResponse,
  SentenceRecognitionResponse,
} from "@/types/api";
import { showSignPhraseButton } from "@/features/translate/phrase";
import { showLegacyUpload, showWordRecognitionCard } from "@/features/translate/visibility";
import { type CaptureDetails, detailsToJson, formatDetailsCompact } from "@/features/translate/details";
import {
  AUTO_SEGMENTER_DEFAULTS,
  AutoSegmenter,
  type AutoEvent,
  type SegmentStats,
} from "./autoSegmenter";
import {
  defaultInterpreterTab,
  splitWordVocab,
  wordDisabledDetail,
  wordVocabSummary,
  type InterpreterTab,
} from "@/features/translate/wordVocab";
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
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Switch } from "@/components/ui/switch";
import { useSpeech } from "./speech";
import { InterpretationPanel, type InterpretationStatus } from "./InterpretationPanel";
import { loadPrefs } from "@/features/settings/preferences";

type Phase =
  | "idle"
  | "starting-camera"
  | "loading-detector"
  | "live"
  | "paused"
  | "error";

type WordPhase = "idle" | "countdown" | "recording" | "result" | "error";

const SEND_INTERVAL_MS = 100; // ~10 landmark frames per second
const DETECT_INTERVAL_MS = 80; // MediaPipe runs far faster than the backend can use; keep preview responsive without re-running all models every animation frame.
const WORD_RECORD_DURATION_MS = 2500; // 2.5 s capture window
const PHRASE_RECORD_DURATION_MS = 10_000; // 10 s capture window for the experimental "Sign a phrase" flow
const WORD_COUNTDOWN_SECONDS = 3;

/** One auto-mode segment's detail entry (extends the capture Details shape). */
interface SegmentLogEntry extends CaptureDetails {
  handFrames: number;
  durationMs: number;
}
interface QueuedSegment {
  frames: LandmarkFramePayload[];
  stats: SegmentStats;
}
/** Build a Details-compatible log entry for one auto-mode segment. */
function buildSegmentLogEntry(
  frames: LandmarkFramePayload[],
  stats: SegmentStats,
  overlay: {
    roundTripMs: number;
    predictedLabel?: string | null;
    confidence?: number;
    threshold?: number;
    accepted?: boolean;
    top3?: Array<{ label: string; probability: number }>;
    modelVersion?: string;
    status: string;
  },
): SegmentLogEntry {
  let leftPresent = 0;
  let rightPresent = 0;
  for (const f of frames) {
    const lm = f.landmarks;
    const left = lm.slice(501, 522).some((p) => p[0] !== 0 || p[1] !== 0 || p[2] !== 0);
    const right = lm.slice(522, 543).some((p) => p[0] !== 0 || p[1] !== 0 || p[2] !== 0);
    if (left) leftPresent += 1;
    if (right) rightPresent += 1;
  }
  const n = Math.max(frames.length, 1);
  return {
    kind: "word",
    framesCaptured: stats.frames,
    effectiveFps: stats.durationMs > 0 ? (stats.frames * 1000) / stats.durationMs : 0,
    leftHandFraction: leftPresent / n,
    rightHandFraction: rightPresent / n,
    roundTripMs: overlay.roundTripMs,
    predictedLabel: overlay.predictedLabel ?? null,
    confidence: overlay.confidence ?? 0,
    threshold: overlay.threshold,
    accepted: overlay.accepted ?? false,
    top3: overlay.top3 ?? [],
    modelVersion: overlay.modelVersion,
    status: overlay.status,
    handFrames: stats.handFrames,
    durationMs: stats.durationMs,
  };
}

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
  // New key per translation interaction (start → stop). Grouping key for
  // session-based history: all sentences of one interaction, even across
  // socket reconnects, land on one history session.
  const historySessionIdRef = useRef<string | null>(null);
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
  const [capabilitiesWaking, setCapabilitiesWaking] = useState(false);
  const [socketStatus, setSocketStatus] = useState<ConnectionStatus>("idle");
  const [socketError, setSocketError] = useState<string | null>(null);
  // Model/backend outage, kept separate from recognition outcomes: a dead
  // model is not an "uncertain prediction".
  const [modelError, setModelError] = useState<string | null>(null);
  const [results, setResults] = useState<TranslationTextMessage[]>([]);
  const [partial, setPartial] = useState<TranslationTextMessage | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
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

  // --- Dev-only labeled capture (VITE_CAPTURE_MODE === "true") ---
  // Hidden entirely unless the env var is exactly "true".
  const captureEnabled = config.captureMode === "true";
  const wordVocab = capabilities?.word_recognition_vocabulary ?? [];
  // Rest (no-sign) class is excluded — displayed separately from the words.
  const wordVocabWords = splitWordVocab(capabilities?.word_recognition_vocabulary).words;
  const [tab, setTab] = useState<InterpreterTab | null>(null);
  // Null until the user picks a tab: derive the default from capabilities.
  const activeTab = tab ?? defaultInterpreterTab(capabilities);
  const wordSupported = showWordRecognitionCard(capabilities);

  // --- Word mode (behind SIGNSAGE_WORD_MODE_ENABLED) ---
  // Reuses the same camera + HolisticExtractor as the live interpreter.
  const [wordPhase, setWordPhase] = useState<WordPhase>("idle");
  const [wordResult, setWordResult] = useState<WordRecognitionResponse | null>(null);
  /** True from the moment capture ends until the recognition response lands. */
  const [wordProcessing, setWordProcessing] = useState(false);
  const [wordError, setWordError] = useState<string | null>(null);
  const [wordList, setWordList] = useState<string[]>([]);
  const [wordConfidences, setWordConfidences] = useState<number[]>([]);
  const [wordCountdown, setWordCountdown] = useState(0);
  const [composedSentence, setComposedSentence] = useState<string | null>(null);
  const [composedMethod, setComposedMethod] = useState<string | null>(null);
  const [composedSignCount, setComposedSignCount] = useState<number | null>(null);
  const wordRafRef = useRef(0);
  const wordLastDetectRef = useRef(0);
  const wordFramesRef = useRef<HolisticFrame[]>([]);
  const wordRecStartRef = useRef(0);
  const wordStopTimeRef = useRef(0);
  // Which flow this capture is for: "Record word" -> word REST endpoint,
  // "Sign a phrase" -> /sentence. Also holds the configured capture length.
  const captureKindRef = useRef<"word" | "phrase">("word");
  const captureDurationRef = useRef(WORD_RECORD_DURATION_MS);
  // Label passed from the "Record & save" dev panel into the same capture path.
  const captureTrueLabelRef = useRef<string | null>(null);
  const [captureLabel, setCaptureLabel] = useState<string>(wordVocab[0] ?? "");
  const [captureCounts, setCaptureCounts] = useState<Record<string, number>>({});
  // Result of the experimental 10-second phrase capture (chips + sentence).
  const [phraseResult, setPhraseResult] = useState<SentenceRecognitionResponse | null>(null);
  const [composedHistorySaved, setComposedHistorySaved] = useState<boolean | null>(null);
  const [composeHistoryError, setComposeHistoryError] = useState<string | null>(null);
  const [phraseHistorySaved, setPhraseHistorySaved] = useState<boolean | null>(null);
  /** Selected phrase duration in ms (6 / 10 / 15 s). */
  const [phraseDurationMs, setPhraseDurationMs] = useState(PHRASE_RECORD_DURATION_MS);
  /** Remaining seconds during phrase recording (shown next to End early). */
  const [phraseRemainingSec, setPhraseRemainingSec] = useState(0);
  const phraseCountdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  /** State mirror of captureKindRef so JSX re-renders on change. */
  const [captureKind, setCaptureKind] = useState<"word" | "phrase">("word");
  /** Last completed capture details for the Details disclosure panel. */
  const [lastDetails, setLastDetails] = useState<CaptureDetails | null>(null);

  // --- Auto mode (hands-free) for the Word tab ---
  // Uses the existing camera/HolisticExtractor pipeline and the existing
  // /word + /compose endpoints; the segmenter is a pure timestamped state
  // machine (see autoSegmenter.ts). Auto mode is off by default.
  const [autoMode, setAutoMode] = useState(false);
  const [autoGapMs, setAutoGapMs] = useState(AUTO_SEGMENTER_DEFAULTS.gapMs);
  const [autoSentencePauseMs, setAutoSentencePauseMs] = useState(
    AUTO_SEGMENTER_DEFAULTS.sentencePauseMs,
  );
  const [autoSettingsOpen, setAutoSettingsOpen] = useState(false);
  /** Displayed auto-mode state line — recomputed 4x/s while Auto mode is on. */
  const [autoStateLine, setAutoStateLine] = useState("Waiting for a sign");
  /** Small muted per-segment note (e.g. "Didn't catch that"). */
  const [autoNote, setAutoNote] = useState<string | null>(null);
  /** Set on HTTP 429 / network failure: Auto mode pauses for 5 s. */
  const [autoPausedMessage, setAutoPausedMessage] = useState<string | null>(null);
  /** Last 10 auto segments (includes rejected ones). */
  const [segmentLog, setSegmentLog] = useState<SegmentLogEntry[]>([]);
  const [inflightCount, setInflightCount] = useState(0);
  const [queuedCount, setQueuedCount] = useState(0);
  /** Last 5 composed sentences, newest first — kept while mounted. */
  const [recentSentences, setRecentSentences] = useState<
    Array<{ sentence: string; saved: boolean | null }>
  >([]);
  const autoSegRef = useRef<AutoSegmenter | null>(null);
  const autoModeRef = useRef(autoMode);
  /** Serial /word queue: at most 3 waiting segments (+ 1 in flight). */
  const autoQueueRef = useRef<QueuedSegment[]>([]);
  const autoInflightRef = useRef(0);
  const autoPausedRef = useRef(false);
  /** Bumped on toggle/teardown so in-flight responses are dropped. */
  const autoGenerationRef = useRef(0);
  const gapRef = useRef(autoGapMs);
  const pauseRef = useRef(autoSentencePauseMs);
  /** Callback held in a ref so the RAF loop (stable) can invoke fresh closures. */
  const handleAutoEventRef = useRef<(ev: AutoEvent) => void>(() => {});
  const pumpAutoRef = useRef<() => void>(() => {});
  const makeSentenceRef = useRef<(opts?: { clearWordsAfter?: boolean }) => void>(() => {});

  // --- Speech engine (Groq Orpheus via backend, browser fallback) ---
  const speech = useSpeech();

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
    // One explicit retry on network failure: a cold/sleeping backend gets a
    // second chance while the user sees an honest "waking up" message.
    const fetchCapabilities = async () => {
      try {
        const caps = await apiClient.capabilities({ timeoutMs: 60_000 });
        live && setCapabilities(caps);
      } catch {
        live && setCapabilitiesWaking(true);
        try {
          const caps = await apiClient.capabilities({ timeoutMs: 60_000 });
          live && setCapabilities(caps);
        } catch {
          live && setCapabilities(null);
        } finally {
          live && setCapabilitiesWaking(false);
        }
      }
    };
    void fetchCapabilities();
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

    // Auto mode: feed every detected frame, including hands-absent ones
    // (the gap of absence is what closes a segment).
    if (
      autoModeRef.current &&
      !autoPausedRef.current &&
      phaseRef.current === "live" &&
      autoSegRef.current
    ) {
      const events = autoSegRef.current.pushFrame({
        tMs: now,
        landmarks: frame.landmarks,
      });
      for (const ev of events) handleAutoEventRef.current(ev);
    }

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
      // Sent with the auth message; stable across reconnects of this
      // interaction so history groups all of its sentences into one session.
      getHistorySessionId: () => historySessionIdRef.current,
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
            // Auto-Speak speaks FINAL translations only, deduped by text.
            if (loadPrefs().autoSpeak) {
              void speech.speak(msg.text, { dedupe: true });
            }
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
    // A new translation interaction begins: fresh history-session scope.
    // (Reconnects within this interaction keep the same key.)
    historySessionIdRef.current = crypto.randomUUID();
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

  // The toggle handler owns create/destroy (event handlers may set state);
  // stop() calls disableAutoMode too — clearing buffers and timers.
  const enableAutoMode = useCallback(() => {
    autoGenerationRef.current += 1;
    autoModeRef.current = true;
    autoSegRef.current = new AutoSegmenter({
      gapMs: gapRef.current,
      sentencePauseMs: pauseRef.current,
    });
    setAutoMode(true);
  }, []);

  const disableAutoMode = useCallback(() => {
    autoGenerationRef.current += 1;
    autoModeRef.current = false;
    autoSegRef.current = null;
    autoQueueRef.current = [];
    autoInflightRef.current = 0;
    autoPausedRef.current = false;
    setAutoMode(false);
    setQueuedCount(0);
    setInflightCount(0);
    setAutoNote(null);
    setAutoPausedMessage(null);
    setSegmentLog([]);
    setAutoSettingsOpen(false);
    setAutoStateLine("Waiting for a sign");
  }, []);

  const pause = useCallback(() => {
    if (phaseRef.current === "live") setPhase("paused");
  }, []);

  const resume = useCallback(() => {
    if (phaseRef.current === "paused") setPhase("live");
  }, []);

  const stop = useCallback(() => {
    // Stopping the camera tears down Auto mode entirely (buffers, timers).
    disableAutoMode();
    const socket = socketRef.current;
    if (socket && socket.status === "open" && framesSentRef.current > 0) {
      // Fresh client key: retrying the same stop() never double-saves the sentence.
      socket.sentenceBoundary(loadPrefs().saveHistory, crypto.randomUUID());
    }
    cancelAnimationFrame(rafRef.current);
    extractorRef.current?.close();
    extractorRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    stopTracks();
    historySessionIdRef.current = null;
    setPartial(null);
    setElapsed(0);
    lastDetectRef.current = 0;
    setPhase("idle");
  }, [stopTracks, disableAutoMode]);

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

  // --- Word mode: capture a 2.5 s window of landmarks and classify ---
  const wordLoop = useCallback(() => {
    wordRafRef.current = requestAnimationFrame(wordLoop);
    const video = videoRef.current;
    const extractor = extractorRef.current;
    if (!video || !extractor || video.readyState < 2) return;

    const now = performance.now();
    const shouldDetect = now - wordLastDetectRef.current >= DETECT_INTERVAL_MS;
    let frame: HolisticFrame | null = null;
    if (shouldDetect) {
      frame = extractor.detect(video, now);
      if (frame) {
        wordLastDetectRef.current = now;
        drawOverlay(canvasRef.current!, frame, "#22c55e");
      }
    }

    if (!frame) return;

    if (wordPhaseRef.current !== "recording") return;
    if (frame.detected) {
      // capture the latest detected frame into the recording buffer
      wordFramesRef.current.push(frame);
    }

    if (now >= wordStopTimeRef.current) {
      stopWordRecording();
      return;
    }
  }, []);

  const wordPhaseRef = useRef<WordPhase>("idle");

  const startWordRecording = useCallback(async (trueLabel?: string, opts?: { durationMs?: number; kind?: "word" | "phrase" }) => {
    if (wordPhaseRef.current === "countdown" || wordPhaseRef.current === "recording") return;
    captureTrueLabelRef.current = trueLabel ?? null;
    captureDurationRef.current = opts?.durationMs ?? WORD_RECORD_DURATION_MS;
    captureKindRef.current = opts?.kind ?? "word";
    setCaptureKind(captureKindRef.current);
    const video = videoRef.current;
    const extractor = extractorRef.current;
    if (!video || !extractor) {
      setWordError("Camera or detector not ready.");
      return;
    }
    setWordError(null);
    if (captureKindRef.current === "phrase") {
      setPhraseResult(null);
      setPhraseHistorySaved(null);
    } else {
      setWordResult(null);
    }
    setWordPhase("countdown");
    wordPhaseRef.current = "countdown";

    // 3-2-1 countdown
    console.log("[word-mode] countdown start");
    let remaining = WORD_COUNTDOWN_SECONDS;
    setWordCountdown(remaining);
    const countdownTimer = setInterval(() => {
      remaining -= 1;
      if (remaining >= 1) {
        setWordCountdown(remaining);
      } else {
        setWordCountdown(0);
        clearInterval(countdownTimer);
        console.log("[word-mode] capture start");
        beginWordCapture();
      }
    }, 1000);
  }, []);

  const beginWordCapture = useCallback(() => {
    const video = videoRef.current;
    const extractor = extractorRef.current;
    if (!video || !extractor) {
      setWordPhase("idle");
      wordPhaseRef.current = "idle";
      setWordError("Camera or detector not ready.");
      return;
    }
    wordFramesRef.current = [];
    wordRecStartRef.current = performance.now();
    wordStopTimeRef.current = wordRecStartRef.current + captureDurationRef.current;
    setWordPhase("recording");
    wordPhaseRef.current = "recording";

    // Phrase: show a live remaining-seconds countdown via a 1 s interval.
    if (captureKindRef.current === "phrase") {
      let secs = Math.round(captureDurationRef.current / 1000);
      setPhraseRemainingSec(secs);
      phraseCountdownRef.current = setInterval(() => {
        secs -= 1;
        setPhraseRemainingSec(secs > 0 ? secs : 0);
      }, 1000);
    }

    // start the capture loop if not already running
    if (wordRafRef.current === 0) {
      wordRafRef.current = requestAnimationFrame(wordLoop);
    }
  }, [wordLoop]);

  const stopWordRecording = useCallback(async () => {
    // "Record word" and "Sign a phrase" share this exact capture path.
    const trueLabel = captureTrueLabelRef.current;
    captureTrueLabelRef.current = null;
    const captureKind = captureKindRef.current;
    setWordPhase("result");
    wordPhaseRef.current = "result";

    const frames = wordFramesRef.current;
    const captureEndMs = performance.now();
    const captureDurationMs = captureEndMs - wordRecStartRef.current;
    const effectiveFps =
      frames && frames.length > 0
        ? (frames.length / captureDurationMs) * 1000
        : 0;

    // Hand-present fractions over captured frames.
    let leftPresent = 0;
    let rightPresent = 0;
    if (frames && frames.length > 0) {
      for (const f of frames) {
        const lm = f.landmarks;
        const left = lm.slice(501, 522).some(([x, y, z]) => x !== 0 || y !== 0 || z !== 0);
        const right = lm.slice(522, 543).some(([x, y, z]) => x !== 0 || y !== 0 || z !== 0);
        if (left) leftPresent += 1;
        if (right) rightPresent += 1;
      }
      leftPresent = leftPresent / frames.length;
      rightPresent = rightPresent / frames.length;
    }

    const requestPayload = {
      frames: frames?.map((f) => ({ landmarks: f.landmarks })) ?? [],
      ...(captureKind === "phrase"
        ? {
            save_history: loadPrefs().saveHistory,
            ...(historySessionIdRef.current
              ? { client_session_key: historySessionIdRef.current }
              : {}),
            client_key: crypto.randomUUID(),
          }
        : {}),
      ...(trueLabel ? { true_label: trueLabel } : {}),
    };
    const payloadSize = JSON.stringify(requestPayload).length;

    const t0Request = performance.now();

    if (frames && frames.length > 0) {
      setWordProcessing(true);
      try {
        if (captureKind === "phrase") {
          const result = await apiClient.sentence(requestPayload, {
            retries: 1,
            timeoutMs: 60_000,
          });
          const roundTripMs = performance.now() - t0Request;
          const totalMs = performance.now() - captureEndMs;
          setPhraseResult(result);
          setPhraseHistorySaved(result.history_saved ?? false);
          setLastDetails({
            kind: "phrase",
            framesCaptured: frames.length,
            effectiveFps,
            leftHandFraction: leftPresent,
            rightHandFraction: rightPresent,
            roundTripMs,
            accepted: result.accepted,
            threshold: result.threshold,
            modelVersion: result.model_version,
            status: result.status,
            glossSequence: result.gloss_sequence,
            sentence: result.sentence,
            nWindows: result.gloss_sequence?.length,
          });
          console.log(
            "[phrase-mode] capture end frames=%d effective_fps=%.2f left_present=%.3f right_present=%.3f payload_bytes=%d round_trip_ms=%.1f total_ms=%.1f gloss_sequence=%s",
            frames.length,
            effectiveFps,
            leftPresent,
            rightPresent,
            payloadSize,
            roundTripMs,
            totalMs,
            result.gloss_sequence?.map((g) => g.gloss).join("|") ?? "",
          );
        } else {
          const result = await apiClient.word(requestPayload, {
            retries: 1,
            timeoutMs: 60_000,
          });
          const roundTripMs = performance.now() - t0Request;
          const totalMs = performance.now() - captureEndMs;
          setWordResult(result);
          setLastDetails({
            kind: "word",
            framesCaptured: frames.length,
            effectiveFps,
            leftHandFraction: leftPresent,
            rightHandFraction: rightPresent,
            roundTripMs,
            predictedLabel: result.word,
            confidence: result.confidence,
            threshold: result.threshold,
            accepted: result.accepted,
            top3: result.top3,
            modelVersion: result.model_version,
            status: result.status,
          });
          if (trueLabel) {
            setCaptureCounts((prev) => ({ ...prev, [trueLabel]: (prev[trueLabel] ?? 0) + 1 }));
          }
          if (result.accepted && result.word) {
            setWordList((prev) => [...prev, result.word!]);
            setWordConfidences((prev) => [...prev, result.confidence]);
          }
          console.log(
            "[word-mode] capture end frames=%d effective_fps=%.2f left_present=%.3f right_present=%.3f payload_bytes=%d round_trip_ms=%.1f total_ms=%.1f",
            frames.length,
            effectiveFps,
            leftPresent,
            rightPresent,
            payloadSize,
            roundTripMs,
            totalMs,
          );
        }
      } catch (err: any) {
        const roundTripMs = performance.now() - t0Request;
        const totalMs = performance.now() - captureEndMs;
        console.log(
          "[word-mode] capture end frames=%d effective_fps=%.2f left_present=%.3f right_present=%.3f payload_bytes=%d round_trip_ms=%.1f total_ms=%.1f error",
          frames.length,
          effectiveFps,
          leftPresent,
          rightPresent,
          payloadSize,
          roundTripMs,
          totalMs,
        );
        setWordError(
          err instanceof NetworkError
            ? "Server is waking up, please wait — try again."
            : err?.message ?? "Word recognition failed.",
        );
      } finally {
        setWordProcessing(false);
      }
    } else {
      const totalMs = performance.now() - captureEndMs;
      console.log(
        "[word-mode] capture end frames=0 effective_fps=0 left_present=0 right_present=0 payload_bytes=0 round_trip_ms=0 total_ms=%.1f no_frames",
        totalMs,
      );
      setWordError("No frames captured.");
    }

    cancelAnimationFrame(wordRafRef.current);
    wordRafRef.current = 0;
    wordFramesRef.current = [];
    setWordCountdown(0);
    if (phraseCountdownRef.current) {
      clearInterval(phraseCountdownRef.current);
      phraseCountdownRef.current = null;
    }
    setPhraseRemainingSec(0);
  }, []);

  const removeWordFromList = useCallback((index: number) => {
    setWordList((prev) => prev.filter((_, i) => i !== index));
    setWordConfidences((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const removeLastWord = useCallback(() => {
    setWordList((prev) => (prev.length > 0 ? prev.slice(0, -1) : prev));
    setWordConfidences((prev) => (prev.length > 0 ? prev.slice(0, -1) : prev));
  }, []);

  const clearWordList = useCallback(() => {
    setWordList([]);
    setWordConfidences([]);
    setWordResult(null);
    setWordError(null);
    setComposedSentence(null);
    setComposedMethod(null);
    setComposedSignCount(null);
    setPhraseResult(null);
    setComposedHistorySaved(null);
    setPhraseHistorySaved(null);
    setComposeHistoryError(null);
    setLastDetails(null);
  }, []);

  const makeSentence = useCallback(async (opts?: { clearWordsAfter?: boolean }) => {
    if (wordList.length === 0) {
      // Nothing to compose; disarm the auto-mode pause clock too so a
      // manually emptied list can't re-fire sentenceDue forever.
      autoSegRef.current?.resetAcceptedWords();
      return;
    }
    setComposedSentence(null);
    try {
      const result = await apiClient.compose({
        words: wordList,
        confidences: wordConfidences,
        save_history: loadPrefs().saveHistory,
        client_session_key: historySessionIdRef.current ?? undefined,
        client_key: crypto.randomUUID(),
      }, {
        retries: 1,
        timeoutMs: 60_000,
      });
      setComposedSentence(result.sentence);
      setComposedMethod(result.method ?? null);
      setComposedSignCount(wordList.length);
      setComposedHistorySaved(result.history_saved ?? false);
      setComposeHistoryError(result.history_error ?? null);
      // Keep the last 5 sentences for this session; the auto-mode panel renders them.
      setRecentSentences((prev) =>
        [{ sentence: result.sentence, saved: result.history_saved ?? false }, ...prev].slice(0, 5),
      );
      // The accepted words are consumed into this sentence — the auto-mode
      // pause countdown disarms until the next segment closes.
      autoSegRef.current?.resetAcceptedWords();
      if (opts?.clearWordsAfter) {
        setWordList([]);
        setWordConfidences([]);
      }
    } catch (err: any) {
      setWordError(
        err instanceof NetworkError
          ? "Server is waking up, please wait — try again."
          : err?.message ?? "Failed to compose sentence.",
      );
    }
  }, [wordList, wordConfidences]);

  // Teardown word mode on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(wordRafRef.current);
    };
  }, []);

  // Sign-out must not leave the previous account's recognised words or
  // composed sentence on screen.
  useEffect(() => {
    return onAuthChange((signedIn) => {
      if (!signedIn) {
        setWordList([]);
        setComposedSentence(null);
        setComposedHistorySaved(null);
        setPhraseHistorySaved(null);
        setWordResult(null);
        setWordError(null);
        setWordCountdown(0);
      }
    });
  }, []);

  // --- Auto mode wiring: lifecycle, serial /word queue, countdown line ---
  // Latest values for the segmenter creation (enableAutoMode reads refs).
  useEffect(() => {
    gapRef.current = autoGapMs;
    pauseRef.current = autoSentencePauseMs;
    autoSegRef.current?.updateConfig({ gapMs: autoGapMs, sentencePauseMs: autoSentencePauseMs });
  }, [autoGapMs, autoSentencePauseMs]);

  // 5-second pause after HTTP 429 / network failure, then Auto mode resumes.
  useEffect(() => {
    if (!autoPausedMessage) return undefined;
    autoPausedRef.current = true;
    const id = window.setTimeout(() => {
      setAutoPausedMessage(null);
      autoPausedRef.current = false;
    }, 5000);
    return () => {
      window.clearTimeout(id);
      autoPausedRef.current = false;
    };
  }, [autoPausedMessage]);

  // Re-render the state line 4x per second while Auto mode is on.
  // setState happens only inside the interval (never synchronously).
  useEffect(() => {
    if (!autoMode) return undefined;
    const compute = () => {
      const seg = autoSegRef.current;
      const phase = seg?.getPhase() ?? "waiting";
      const remaining =
        autoPausedMessage || phase !== "waiting" || !seg
          ? null
          : seg.getSentenceCountdownMs(performance.now());
      const line = autoPausedMessage
        ? `Paused — ${autoPausedMessage}`
        : phase !== "waiting"
          ? "Signing..."
          : inflightCount > 0 || queuedCount > 0
            ? "Recognising..."
            : remaining !== null
              ? `Sentence in ${(remaining / 1000).toFixed(1)}s`
              : "Waiting for a sign";
      setAutoStateLine(line);
    };
    const id = window.setInterval(compute, 250);
    return () => window.clearInterval(id);
  }, [autoMode, autoPausedMessage, inflightCount, queuedCount]);

  const appendSegmentLog = useCallback((entry: SegmentLogEntry) => {
    setSegmentLog((prev) => [entry, ...prev].slice(0, 10));
  }, []);

  const pumpAutoQueue = useCallback(async () => {
    if (autoInflightRef.current > 0 || autoPausedRef.current) return;
    const item = autoQueueRef.current.shift();
    if (!item) {
      setQueuedCount(0);
      return;
    }
    autoInflightRef.current += 1;
    setInflightCount(autoInflightRef.current);
    setQueuedCount(autoQueueRef.current.length);
    const generation = autoGenerationRef.current;
    const t0 = performance.now();
    const { frames, stats } = item;
    try {
      const result = await apiClient.word(
        { frames },
        { retries: 1, timeoutMs: 60_000 },
      );
      const roundTripMs = performance.now() - t0;
      if (generation !== autoGenerationRef.current) return; // switched off mid-flight
      appendSegmentLog(
        buildSegmentLogEntry(frames, stats, {
          roundTripMs,
          predictedLabel: result.word,
          confidence: result.confidence,
          threshold: result.threshold,
          accepted: result.accepted,
          top3: result.top3,
          modelVersion: result.model_version,
          status: result.status,
        }),
      );
      if (result.accepted && result.word) {
        setWordList((prev) => [...prev, result.word!]);
        setWordConfidences((prev) => [...prev, result.confidence]);
        autoSegRef.current?.noteWordAccepted();
        setAutoNote(null);
      } else {
        setAutoNote("Didn't catch that");
      }
    } catch (err: unknown) {
      const roundTripMs = performance.now() - t0;
      if (generation !== autoGenerationRef.current) return;
      const isRateLimit = err instanceof ApiError && err.status === 429;
      const message =
        err instanceof NetworkError
          ? "Server is waking up, please wait — try again."
          : isRateLimit
            ? (err as ApiError).message || "Too many requests — pausing Auto mode."
            : err instanceof Error
              ? err.message
              : "Word recognition failed.";
      appendSegmentLog(
        buildSegmentLogEntry(frames, stats, {
          roundTripMs,
          status: isRateLimit
            ? "rate_limited"
            : err instanceof NetworkError
              ? "network_error"
              : "error",
          accepted: false,
        }),
      );
      if (isRateLimit || err instanceof NetworkError) {
        // Rate limited or offline: drop the queued segments and pause.
        autoQueueRef.current = [];
        setQueuedCount(0);
        setAutoPausedMessage(message);
      }
      setAutoNote(message);
    } finally {
      autoInflightRef.current = Math.max(0, autoInflightRef.current - 1);
      setInflightCount(autoInflightRef.current);
      pumpAutoRef.current();
    }
  }, [appendSegmentLog]);

  useEffect(() => {
    pumpAutoRef.current = () => {
      void pumpAutoQueue();
    };
  }, [pumpAutoQueue]);

  const enqueueAutoSegment = useCallback(
    (segment: QueuedSegment) => {
      if (autoPausedRef.current) return; // paused (429 / network): ignore segments
      if (autoQueueRef.current.length >= 3) {
        // Serial queue full (3 waiting + 1 in flight): skip this sign.
        setAutoNote("Too busy — a sign was skipped.");
        appendSegmentLog(
          buildSegmentLogEntry(segment.frames, segment.stats, {
            roundTripMs: 0,
            status: "dropped_queue_full",
            accepted: false,
          }),
        );
        return;
      }
      autoQueueRef.current.push(segment);
      setQueuedCount(autoQueueRef.current.length);
      void pumpAutoQueue();
    },
    [appendSegmentLog, pumpAutoQueue],
  );

  const handleAutoEvent = useCallback(
    (ev: AutoEvent) => {
      if (ev.type === "sentenceDue") {
        // Same function the Make sentence button uses; clears the word list on success.
        void makeSentenceRef.current?.({ clearWordsAfter: true });
        return;
      }
      if (ev.type === "segmentReady") {
        enqueueAutoSegment({
          frames: ev.frames.map((f) => ({
            landmarks: f.landmarks as unknown as number[][],
          })),
          stats: ev.stats,
        });
      }
    },
    [enqueueAutoSegment],
  );

  useEffect(() => {
    handleAutoEventRef.current = handleAutoEvent;
  }, [handleAutoEvent]);
  useEffect(() => {
    makeSentenceRef.current = makeSentence;
  }, [makeSentence]);

  // --- Interpretation derivation (unified panel feed) ---
  const phraseSentence = phraseResult?.accepted ? phraseResult.sentence : null;
  const currentSentence = composedSentence ?? phraseSentence;
  const usingPhrase = !composedSentence && phraseSentence !== null;
  const interpretationStatus: InterpretationStatus = wordProcessing
    ? "composing"
    : currentSentence
      ? "composed"
      : phraseResult && !phraseResult.accepted
        ? "failed"
        : wordList.length > 0
          ? "recognising"
          : "waiting";
  const interpretationProvenance = {
    method: usingPhrase ? (phraseResult?.method ?? null) : composedMethod,
    signs: usingPhrase ? (phraseResult?.gloss_sequence?.length ?? null) : composedSignCount,
    model: usingPhrase ? (phraseResult?.model_version ?? null) : null,
    saved: usingPhrase ? (phraseHistorySaved ?? null) : (composedHistorySaved ?? null),
    historyError: usingPhrase ? (phraseResult?.history_error ?? null) : composeHistoryError,
  };
  const interpretationGlosses = usingPhrase ? (phraseResult?.gloss_sequence ?? null) : null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h1 className="text-3xl font-semibold tracking-tight">Live Interpreter</h1>
          <p className="mt-1 max-w-xl text-sm leading-6 text-foreground/55">
            Sign in front of the camera and SignSage interprets your signs. Use Word recognition for
            the trained model, or Basic preview for the 5-shape demo recognizer.
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="rounded-full px-4"
            onClick={() => setHelpOpen(true)}
          >
            <LifeBuoy aria-hidden="true" />
            Help
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
              Camera mode is still available.
            </p>
          </div>
        </div>
      )}

      {isolatedWordModel && (
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm">
          <p className="font-medium">Limited isolated-word ISL vocabulary</p>
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

      {/* Legacy six-word video-upload pilot: hidden by default, shown only when VITE_LEGACY_UPLOAD === "true". Code intentionally kept. */}
      {showLegacyUpload() && (
      <div className="rounded-xl border border-foreground/10 bg-card px-4 py-3 text-sm">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50 mb-2">
          Video upload (isolated-word recognition)
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
      )}

      {/* Workspace: camera stage (left, ~60%) + tabbed recognition panel (right, ~40%) */}
      <div className="grid items-start gap-4 lg:grid-cols-5">
      <div className="relative mx-auto w-full max-w-[640px] aspect-[4/3] overflow-hidden rounded-xl border border-foreground/10 bg-black lg:col-span-3 self-start">
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
                  {modelError ? "Camera mode is still available." : "Ready to interpret"}
                </p>
                <p className="mt-1 text-sm">
                  {phase === "error"
                    ? (cameraMessage ??
                      detectorMessage ??
                      "Something went wrong. Try again.")
                    : modelBlocked
                      ? "Translation is unavailable — the recognition model has not passed the deployment gate. Camera mode and landmark overlay will still work."
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
        {/* Camera stage Start/Stop — same handlers as the session bar. */}
        {!busy && (
          <button
            type="button"
            onClick={() => (running ? stop() : void start())}
            className="absolute bottom-3 right-3 rounded-full border border-white/10 bg-black/55 px-3 py-1.5 text-[11px] font-medium text-white/85 backdrop-blur-md transition-colors hover:bg-black/70 hover:text-white focus-visible:ring-2 focus-visible:ring-white/60 focus-visible:outline-none"
          >
            {running ? "Stop camera" : "Start camera"}
          </button>
        )}
        </div>
        <div className="flex min-h-64 flex-col rounded-xl border border-foreground/10 bg-card lg:col-span-2 lg:min-h-0">
          <p className="border-b border-foreground/10 px-4 pb-2 pt-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/45">
            Recognition mode
          </p>
          <Tabs
            value={activeTab}
            onValueChange={(v) => setTab(v as InterpreterTab)}
            className="flex flex-1 flex-col"
          >
            <TabsList className="mx-4 mt-3 grid w-[calc(100%-2rem)] grid-cols-2">
              <TabsTrigger value="word">Word recognition</TabsTrigger>
              <TabsTrigger value="rule">Basic preview</TabsTrigger>
            </TabsList>

            <TabsContent value="word" className="flex-1 px-4 pb-4 pt-2">
              {/* Model status chip */}
              {backendOnline !== null && (
                <div className={`mb-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${wordSupported ? "bg-green-500/15 text-green-700" : "bg-amber-500/15 text-amber-700"}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${wordSupported ? "bg-green-500" : "bg-amber-500"}`} aria-hidden="true" />
                  {wordSupported
                    ? `Trained model ready: ${capabilities?.word_recognition_model_version ?? "unknown"}`
                    : `Not available: ${capabilities?.word_recognition_unavailable_reason ?? (backendOnline === false ? "server not reachable" : "word recognition disabled")}`}
                </div>
              )}
              {wordSupported ? (
                <div className="flex flex-col gap-3 text-sm">
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
                        Trained model
                      </p>
                      <span
                        className="max-w-40 truncate text-[10px] text-foreground/40"
                        title={capabilities?.word_recognition_label}
                      >
                        {capabilities?.word_recognition_label}
                      </span>
                    </div>
                    {wordVocabWords.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {wordVocabWords.map((w) => (
                          <Badge key={w} variant="secondary" className="font-normal">
                            {w}
                          </Badge>
                        ))}
                      </div>
                    )}
                    <p className="mt-1 text-[10px] text-foreground/45">
                      {wordVocabSummary(capabilities?.word_recognition_vocabulary)}
                    </p>
                  </div>
                  <p className="text-xs leading-snug text-foreground/50">
                    Sign one word at a time, then press Make sentence
                  </p>

                  {/* Auto mode (hands-free) — visible only while the camera is running */}
                  {running && (
                    <div className="rounded-md border border-foreground/10 bg-foreground/[0.03] px-3 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="text-xs font-semibold text-foreground/80">
                              Auto mode (hands-free)
                            </p>
                            {/* Settings popover: word gap + sentence pause */}
                            <div className="relative">
                              <button
                                type="button"
                                aria-label="Auto mode settings"
                                aria-expanded={autoSettingsOpen}
                                onClick={() => setAutoSettingsOpen((v) => !v)}
                                className="rounded p-1 text-foreground/45 hover:bg-foreground/10 hover:text-foreground/70"
                              >
                                <Info size={14} aria-hidden="true" />
                              </button>
                              {autoSettingsOpen && (
                                <div className="absolute left-0 z-20 mt-1 w-60 rounded-md border border-foreground/15 bg-card p-2.5 shadow-lg">
                                  <label
                                    htmlFor="auto-gap"
                                    className="block text-[11px] font-medium text-foreground/70"
                                  >
                                    Word gap (lower hands between signs)
                                  </label>
                                  <select
                                    id="auto-gap"
                                    className="mt-1 w-full rounded border border-foreground/20 bg-background px-2 py-1 text-xs"
                                    value={autoGapMs}
                                    onChange={(e) => setAutoGapMs(Number(e.target.value))}
                                  >
                                    <option value={500}>0.5 s</option>
                                    <option value={700}>0.7 s</option>
                                    <option value={1000}>1.0 s</option>
                                  </select>
                                  <label
                                    htmlFor="auto-pause"
                                    className="mt-2 block text-[11px] font-medium text-foreground/70"
                                  >
                                    Sentence pause
                                  </label>
                                  <select
                                    id="auto-pause"
                                    className="mt-1 w-full rounded border border-foreground/20 bg-background px-2 py-1 text-xs"
                                    value={autoSentencePauseMs}
                                    onChange={(e) => setAutoSentencePauseMs(Number(e.target.value))}
                                  >
                                    <option value={1500}>1.5 s</option>
                                    <option value={2000}>2 s</option>
                                    <option value={3000}>3 s</option>
                                  </select>
                                </div>
                              )}
                            </div>
                          </div>
                          <p className="mt-0.5 text-[11px] leading-snug text-foreground/55">
                            Sign a word, lower your hands for about a second, then sign the next. After a 2-second pause the sentence appears.
                          </p>
                        </div>
                        <Button
                          type="button"
                          size="sm"
                          variant={autoMode ? "default" : "outline"}
                          aria-pressed={autoMode}
                          onClick={() => (autoMode ? disableAutoMode() : enableAutoMode())}
                          className="shrink-0 rounded-full"
                        >
                          {autoMode ? "On" : "Off"}
                        </Button>
                      </div>
                      {autoMode && (
                        <div className="mt-2 border-t border-foreground/10 pt-2">
                          <p className="text-[11px] font-medium text-foreground/70" role="status" aria-live="polite">
                            {autoStateLine}
                          </p>
                          {autoNote && (
                            <p className="mt-0.5 text-[11px] text-foreground/50">{autoNote}</p>
                          )}
                          {recentSentences.length > 0 && (
                            <div className="mt-1.5">
                              <p className="text-[10px] font-semibold uppercase tracking-wide text-foreground/45">
                                Recent sentences
                              </p>
                              <ul className="mt-0.5 space-y-0.5">
                                {recentSentences.map((s, i) => (
                                  <li key={`${i}-${s.sentence}`} className="text-[11px] leading-snug text-foreground/60">
                                    {s.sentence}
                                    {s.saved !== null && (
                                      <span className="text-foreground/40">
                                        {" "}· {s.saved ? "Saved" : "Not saved"}
                                      </span>
                                    )}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {segmentLog.length > 0 && (
                            <details className="mt-1.5 rounded-md border border-foreground/10 text-[11px]">
                              <summary className="flex cursor-pointer select-none items-center justify-between gap-2 px-3 py-1.5 text-foreground/60 hover:text-foreground">
                                <span>Segments ({segmentLog.length})</span>
                                <button
                                  type="button"
                                  className="ml-auto rounded px-2 py-0.5 text-[10px] text-foreground/50 hover:bg-foreground/10"
                                  onClick={(e) => {
                                    e.preventDefault();
                                    void navigator.clipboard?.writeText(
                                      JSON.stringify(
                                        segmentLog.map((d) => formatDetailsCompact(d)),
                                        null,
                                        2,
                                      ),
                                    );
                                  }}
                                >
                                  Copy
                                </button>
                              </summary>
                              <div className="max-h-36 space-y-0.5 overflow-y-auto px-3 pb-2 pt-1 font-mono text-[10px] text-foreground/60">
                                {segmentLog.map((d, i) => (
                                  <p key={i} className="whitespace-nowrap">
                                    {d.framesCaptured} fr · {d.handFrames} hf · {d.durationMs} ms · rt {d.roundTripMs.toFixed(0)} ms · {d.predictedLabel ?? "—"} · {(d.confidence ?? 0).toFixed(2)} · {d.accepted ? "accepted" : "rejected"}
                                  </p>
                                ))}
                              </div>
                            </details>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Start/stop lives on the camera stage and the session bar — no third button here. */}

                  {/* Sentence construction */}
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
                      Sentence construction
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      type="button"
                      variant="default"
                      size="sm"
                      disabled={
                        wordPhase === "countdown" ||
                        wordPhase === "recording" ||
                        wordProcessing ||
                        phase === "starting-camera" ||
                        phase === "loading-detector" ||
                        phase === "error"
                      }
                      onClick={() => startWordRecording()}
                    >
                      {wordPhase === "countdown" ? (
                        <span className="flex items-center gap-2">
                          <span className="font-mono font-bold text-green-600">{wordCountdown}</span>
                          Recording…
                        </span>
                      ) : wordPhase === "recording" ? (
                        <span className="flex items-center gap-2">
                          <span className="font-mono font-bold text-red-600">●</span>
                          Recording word…
                        </span>
                      ) : (
                        <>
                          <Camera size={14} /> Record word
                        </>
                      )}
                    </Button>
                    {showSignPhraseButton(capabilities) && (
                      <>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          disabled={
                            wordPhase === "countdown" ||
                            wordPhase === "recording" ||
                            wordProcessing ||
                            phase === "starting-camera" ||
                            phase === "loading-detector" ||
                            phase === "error"
                          }
                          onClick={() => startWordRecording(undefined, { durationMs: phraseDurationMs, kind: "phrase" })}
                        >
                          <Camera size={14} /> Sign a phrase <span className="text-foreground/40">(experimental)</span>
                        </Button>
                        {/* End early: visible only while phrase is recording */}
                        {wordPhase === "recording" && captureKind === "phrase" && (
                          <Button
                            type="button"
                            variant="destructive"
                            size="sm"
                            onClick={() => void stopWordRecording()}
                          >
                            End early {phraseRemainingSec > 0 && <span className="ml-1 font-mono text-xs">{phraseRemainingSec}s</span>}
                          </Button>
                        )}
                        {/* Duration selector: hidden while recording */}
                        {wordPhase !== "recording" && wordPhase !== "countdown" && (
                          <select
                            value={phraseDurationMs}
                            onChange={(e) => setPhraseDurationMs(Number(e.target.value))}
                            className="rounded border border-foreground/20 bg-background px-2 py-1 text-xs"
                            aria-label="Phrase duration"
                          >
                            <option value={6_000}>6 s</option>
                            <option value={10_000}>10 s</option>
                            <option value={15_000}>15 s</option>
                          </select>
                        )}
                      </>
                    )}
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={wordList.length === 0}
                      onClick={() => void makeSentence()}
                    >
                      Make sentence
                    </Button>
                  </div>

                  {wordError && <p className="text-xs text-destructive">{wordError}</p>}

                  {/* Shown between the end of capture and the result chip landing. */}
                  {wordProcessing && !wordResult && (
                    <div className="rounded-md bg-foreground/5 px-3 py-2">
                      <div className="flex items-center gap-3 text-xs text-foreground/60">
                        <Spinner className="h-3 w-3" />
                        <span>Processing…</span>
                      </div>
                    </div>
                  )}

                  {wordResult && (
                    <div className="rounded-md bg-foreground/5 px-3 py-2">
                      <p className="text-xs font-semibold text-foreground/60 mb-1">
                        {wordResult.status === "no_hands"
                          ? "No hands detected — please repeat"
                          : wordResult.accepted
                            ? "Recognised word"
                            : "Not confidently recognised"}
                      </p>
                      <div className="flex items-center gap-3">
                        <span className={wordResult.accepted ? "font-mono text-2xl font-bold text-foreground" : "font-mono text-2xl font-bold text-foreground/40"}>
                          {wordResult.word ?? "—"}
                        </span>
                        <div className="text-right text-xs">
                          <p className="font-medium">
                            {wordResult.accepted ? "accepted" : "rejected"}
                          </p>
                          <p className="text-foreground/50">
                            confidence {wordResult.confidence.toFixed(2)} ·
                            threshold {wordResult.threshold.toFixed(2)}
                          </p>
                        </div>
                      </div>
                      {wordResult.top3.length > 0 && !wordResult.accepted && (
                        <p className="mt-1 text-xs text-foreground/50">
                          Top guesses:{" "}
                          {wordResult.top3.map((t) => `${t.label} (${t.probability.toFixed(2)})`).join(", ")}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Details disclosure: last word or phrase capture */}
                  {lastDetails && (
                    <details className="rounded-md border border-foreground/10 text-xs">
                      <summary className="flex cursor-pointer select-none items-center justify-between px-3 py-2 text-foreground/60 hover:text-foreground">
                        <span>Details</span>
                        <button
                          type="button"
                          className="ml-auto rounded px-2 py-0.5 text-[10px] text-foreground/50 hover:bg-foreground/10"
                          onClick={(e) => { e.preventDefault(); navigator.clipboard.writeText(detailsToJson(lastDetails)); }}
                        >
                          Copy
                        </button>
                      </summary>
                      <div className="space-y-0.5 px-3 pb-3 pt-1 font-mono text-[11px] text-foreground/60">
                        <p>frames: {lastDetails.framesCaptured} · fps: {lastDetails.effectiveFps.toFixed(1)} · rt: {lastDetails.roundTripMs.toFixed(0)} ms</p>
                        <p>left hand: {(lastDetails.leftHandFraction * 100).toFixed(0)}% · right hand: {(lastDetails.rightHandFraction * 100).toFixed(0)}%</p>
                        {lastDetails.kind === "word" ? (
                          <>
                            <p>label: {lastDetails.predictedLabel ?? "—"} · confidence: {lastDetails.confidence?.toFixed(3) ?? "—"} · threshold: {lastDetails.threshold?.toFixed(2) ?? "—"}</p>
                            <p>accepted: {String(lastDetails.accepted)} · status: {lastDetails.status ?? "—"}</p>
                            {lastDetails.top3 && lastDetails.top3.length > 0 && (
                              <p>top-3: {lastDetails.top3.map((t) => `${t.label} (${t.probability.toFixed(2)})`).join(", ")}</p>
                            )}
                          </>
                        ) : (
                          <>
                            <p>sentence: {lastDetails.sentence ?? "—"} · windows: {lastDetails.nWindows ?? "—"}</p>
                            <p>accepted: {String(lastDetails.accepted)} · threshold: {lastDetails.threshold?.toFixed(2) ?? "—"}</p>
                            {lastDetails.glossSequence && lastDetails.glossSequence.length > 0 && (
                              <p>glosses: {lastDetails.glossSequence.map((g) => `${g.gloss}(${g.confidence.toFixed(2)})`).join(", ")}</p>
                            )}
                          </>
                        )}
                        <p>model: {lastDetails.modelVersion ?? "—"}</p>
                      </div>
                    </details>
                  )}

                  {/* Accepted words list */}
                  {wordList.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-foreground/60 mb-1">
                        Recognised words
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {wordList.map((w, i) => (
                          <div key={i} className="flex items-center gap-1 bg-muted rounded-full px-3 py-1.5 text-xs">
                            <span className="font-medium">{w}</span>
                            {autoMode && wordConfidences[i] !== undefined && (
                              <span className="font-mono text-[10px] text-foreground/55">
                                {wordConfidences[i].toFixed(2)}
                              </span>
                            )}
                            <Button
                              type="button"
                              variant="ghost"
                              size="xs"
                              className="p-1"
                              onClick={() => removeWordFromList(i)}
                              aria-label={`Remove ${w}`}
                            >
                              <Square size={10} />
                            </Button>
                          </div>
                        ))}
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={removeLastWord}
                        >
                          Remove last
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={clearWordList}
                        >
                          Clear
                        </Button>
                      </div>
                      <p className="mt-1 text-xs text-foreground/50">
                        Tap a word to remove it. Clear removes the whole list.
                      </p>
                    </div>
                  )}

                  {/* Unified interpretation panel: sentence + audio output */}
                  <InterpretationPanel
                    sentence={currentSentence}
                    status={interpretationStatus}
                    provenance={interpretationProvenance}
                    glossSequence={interpretationGlosses}
                    emptyHint={interpretationStatus === "failed" ? "Didn't catch that, please try again." : undefined}
                    speech={speech}
                  />

                  {/* Dev-only labeled capture panel (VITE_CAPTURE_MODE === "true"). Hidden entirely otherwise. */}
                  {captureEnabled && capabilities?.word_recognition_enabled && (
                    <div className="rounded-xl border border-foreground/10 bg-card px-4 py-3 text-sm mt-1">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50 mb-0">
                          Labeled capture (dev)
                        </p>
                        <span className="text-xs text-foreground/40">
                          VITE_CAPTURE_MODE=true
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 mb-2">
                        <label className="text-xs text-foreground/60">Label</label>
                        <select
                          value={captureLabel}
                          onChange={(e) => setCaptureLabel(e.target.value)}
                          className="border border-foreground/20 rounded px-2 py-1.5 text-xs bg-background"
                          disabled={wordProcessing}
                        >
                          {wordVocab.map((v) => (
                            <option key={v} value={v}>
                              {v}
                            </option>
                          ))}
                        </select>
                        <Button
                          type="button"
                          variant="default"
                          size="sm"
                          disabled={
                            wordProcessing ||
                            wordPhase === "countdown" ||
                            wordPhase === "recording" ||
                            phase === "starting-camera" ||
                            phase === "loading-detector" ||
                            phase === "error"
                          }
                          onClick={() => startWordRecording(captureLabel)}
                        >
                          Record &amp; save
                        </Button>
                      </div>
                      <div className="flex flex-wrap gap-2 text-xs text-foreground/50">
                        {wordVocab.map((v) => (
                          <span key={v} className="flex items-center gap-1">
                            {v}: <span className="font-mono">{captureCounts[v] ?? 0}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="mt-1 rounded-xl border border-foreground/10 bg-background/40 p-4 text-sm">
                  <p className="font-medium">Word recognition is not enabled on this server</p>
                  {wordDisabledDetail(capabilities) && (
                    <p className="mt-1 text-xs text-foreground/60">{wordDisabledDetail(capabilities)}</p>
                  )}
                  <p className="mt-2 text-xs text-foreground/50">
                    Use the Basic preview tab instead.
                  </p>
                  <p className="mt-2 text-xs text-foreground/45">
                    Developers: set <code>SIGNSAGE_WORD_MODE_ENABLED=true</code> and restart the backend.
                  </p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="rule" className="flex-1 px-4 pb-4 pt-2">
              <div className="flex h-full flex-col">
                <div className="border-b border-foreground/10 pb-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
                    Live translation
                  </p>
                  <p className="mt-1 text-[10px] text-foreground/50">Basic shape matcher only. Use Word recognition for the trained model.</p>
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
                <div className="flex flex-1 flex-col justify-center py-4">
                  {modelError ? (
                    <>
                      <p className="text-2xl font-semibold tracking-tight text-foreground/40">
                        Translation unavailable
                      </p>
                      <p className="mt-2 text-xs text-foreground/55">
                        {modelError} Camera mode is still available.
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
                        {!showingPartial && (
                          <div className="mt-3 flex flex-wrap items-center gap-2">
                            <Button
                              type="button"
                              size="sm"
                              className="rounded-full"
                              disabled={speech.speechState === "loading"}
                              onClick={() => {
                                if (speech.speechState === "speaking" || speech.speechState === "loading") {
                                  speech.stop();
                                } else {
                                  void speech.speak(display.text);
                                }
                              }}
                              title="Speak this interpretation"
                            >
                              {speech.speechState === "loading" ? (
                                <Loader2 aria-hidden="true" className="animate-spin" />
                              ) : speech.speechState === "speaking" ? (
                                <VolumeX aria-hidden="true" />
                              ) : (
                                <Volume2 aria-hidden="true" />
                              )}
                              {speech.speechState === "speaking" ? "Stop" : "Speak"}
                            </Button>
                            <label className="flex cursor-pointer items-center gap-2 text-xs text-foreground/70">
                              <Switch
                                checked={speech.prefs.autoSpeak}
                                onCheckedChange={(v) => speech.setAutoSpeak(v === true)}
                                aria-label="Auto-Speak final interpretations"
                              />
                              Auto-Speak
                            </label>
                          </div>
                        )}
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
            </TabsContent>
          </Tabs>
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
          {capabilitiesWaking && (
            <span className="text-foreground/60">
              Server is waking up, please wait…
            </span>
          )}
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

      <AlertDialog open={helpOpen} onOpenChange={setHelpOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Indian Sign Language Reference</AlertDialogTitle>
            <AlertDialogDescription>
              Need help learning or checking an Indian Sign Language gesture? Visit the
              Indian Sign Language Portal, which provides a searchable collection of ISL
              signs with reference images and videos.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => window.open("https://indiansignlanguage.org/", "_blank", "noopener,noreferrer")}
            >
              Visit ISL Portal
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
