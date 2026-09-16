import { useCallback, useEffect, useRef, useState } from "react"
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision"
import { flattenHolisticLandmarks, hasReliableSignLandmarks, normalizeLandmarkSequence, poseMotion, TEMPORAL_MIN_FRAMES, TEMPORAL_MOTION_THRESHOLD, TEMPORAL_PAUSE_MS, type HolisticLandmarks } from "../../lib/landmarks"

type CaptureStatus = "off" | "loading" | "ready" | "warning" | "error"
const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
const MODEL_URL = import.meta.env.VITE_MEDIAPIPE_MODEL_URL
  ?? "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

type TranscriptEntry = { id: string; dir: string; text: string; latency?: number }

export function SignModePanel({ caption, captionLatency, captionMeta, error, send, connectionState, transcript, setTranscript }: {
  caption: string; captionLatency: number | null; captionMeta?: { confidence?: number; is_fallback?: boolean; gated_reason?: string | null } | null; error: string | null
  send: (msg: Record<string, unknown>) => boolean
  connectionState: string; transcript: TranscriptEntry[]; setTranscript: React.Dispatch<React.SetStateAction<TranscriptEntry[]>>
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const landmarkerRef = useRef<HolisticLandmarker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameRequestRef = useRef<number | null>(null)
  const bufferedFramesRef = useRef<number[][]>([])
  const previousFrameRef = useRef<number[] | null>(null)
  const pauseSinceRef = useRef<number | null>(null)
  const finishSentenceRef = useRef<(t: "manual" | "motion_pause") => void>(() => undefined)
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus>("off")
  const [captureNote, setCaptureNote] = useState("Camera off")
  const [bufferedFrameCount, setBufferedFrameCount] = useState(0)
  const [pauseMs, setPauseMs] = useState(TEMPORAL_PAUSE_MS)
  const [waitingPause, setWaitingPause] = useState(false)
  const pauseMsRef = useRef(pauseMs)
  useEffect(() => { pauseMsRef.current = pauseMs }, [pauseMs])

  const finishSentence = useCallback((trigger: "manual" | "motion_pause") => {
    const raw = bufferedFramesRef.current
    if (raw.length === 0) { setCaptureNote("No reliable landmarks captured yet."); return }
    bufferedFramesRef.current = []; previousFrameRef.current = null; pauseSinceRef.current = null; setBufferedFrameCount(0); setWaitingPause(false)
    const normalized = normalizeLandmarkSequence(raw)
    const sent = normalized.every(l => send({ type: "landmarks.frame", timestamp_ms: Math.round(performance.now()), landmarks: l })) && send({ type: "sentence.boundary", trigger })
    setCaptureNote(sent ? "Sentence sent. Waiting for caption." : "Backend unavailable.")
    if (!sent) setCaptureStatus("error")
  }, [send])
  useEffect(() => { finishSentenceRef.current = finishSentence }, [finishSentence])

  const stopCamera = useCallback(() => {
    if (frameRequestRef.current !== null) window.cancelAnimationFrame(frameRequestRef.current)
    frameRequestRef.current = null; landmarkerRef.current?.close(); landmarkerRef.current = null
    streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    bufferedFramesRef.current = []; previousFrameRef.current = null; pauseSinceRef.current = null; setBufferedFrameCount(0)
    setCaptureStatus("off"); setCaptureNote("Camera off")
  }, [])

  const startCamera = useCallback(async () => {
    if (connectionState !== "connected") { setCaptureStatus("error"); setCaptureNote("Backend socket is unavailable."); return }
    setCaptureStatus("loading"); setCaptureNote("Loading camera...")
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      const video = videoRef.current; if (!video) throw new Error("Camera preview unavailable.")
      streamRef.current = stream; video.srcObject = stream; await video.play()
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT)
      const landmarker = await HolisticLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL }, runningMode: "VIDEO", minFaceDetectionConfidence: 0.5, minPoseDetectionConfidence: 0.5, minHandLandmarksConfidence: 0.5 })
      landmarkerRef.current = landmarker; setCaptureStatus("ready"); setCaptureNote("Signing. Pause after sentence or press Send.")
      const processFrame = () => {
        const v = videoRef.current, lm = landmarkerRef.current; if (!v || !lm) return
        const result = lm.detectForVideo(v, performance.now())
        const landmarks: HolisticLandmarks = { pose: result.poseLandmarks[0], face: result.faceLandmarks[0], leftHand: result.leftHandLandmarks[0], rightHand: result.rightHandLandmarks[0] }
        if (hasReliableSignLandmarks(landmarks)) {
          if (captureStatus !== "ready") { setCaptureStatus("ready"); setCaptureNote("Signing. Pause after sentence or press Send.") }
          const frame = flattenHolisticLandmarks(landmarks); bufferedFramesRef.current.push(frame); setBufferedFrameCount(bufferedFramesRef.current.length)
          const motion = previousFrameRef.current ? poseMotion(previousFrameRef.current, frame) : Infinity; previousFrameRef.current = frame
          if (motion < TEMPORAL_MOTION_THRESHOLD && bufferedFramesRef.current.length >= TEMPORAL_MIN_FRAMES) {
            pauseSinceRef.current ??= performance.now()
            setWaitingPause(true)
            if (performance.now() - pauseSinceRef.current >= pauseMsRef.current) finishSentenceRef.current("motion_pause")
          } else { pauseSinceRef.current = null; setWaitingPause(false) }
        } else { setCaptureStatus("warning"); setCaptureNote("Tracking needs upper body and at least one hand in frame.") }
        frameRequestRef.current = window.requestAnimationFrame(processFrame)
      }
      frameRequestRef.current = window.requestAnimationFrame(processFrame)
    } catch (caught) { stopCamera(); setCaptureStatus("error"); setCaptureNote(caught instanceof Error ? caught.message : "Camera error.") }
  }, [connectionState, stopCamera, captureStatus])

  useEffect(() => stopCamera, [stopCamera])
  const glossTokens = caption ? caption.split(/\s+/).slice(0, 8) : []

  return (
    <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
      <section className="lg:col-span-7 flex flex-col gap-2">
        <div className="relative w-full aspect-[16/10] bg-surface-container-lowest rounded-xl overflow-hidden shadow-2xl border border-outline-variant/20 flex items-center justify-center">
          <video ref={videoRef} muted playsInline className="absolute inset-0 w-full h-full object-cover -scale-x-100" />
          <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-30" viewBox="0 0 700 460" preserveAspectRatio="none"><defs><pattern id="grid-s" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#253342" strokeWidth="0.5" /></pattern></defs><rect width="100%" height="100%" fill="url(#grid-s)" /></svg>
          <div className="absolute top-3 left-3 flex flex-col gap-1.5">
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-primary"><span className={`w-1.5 h-1.5 rounded-full ${captureStatus==="ready" ? "bg-primary animate-pulse" : captureStatus==="warning" ? "bg-error" : "bg-outline"}`} />{captureStatus==="ready" ? "TRACKING" : captureStatus.toUpperCase()} · {captureStatus==="ready" ? "1080p" : captureNote.slice(0,22)}</span>
            <span className="inline-flex px-2 py-0.5 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-on-surface-variant">TARGET: CONTINUOUS ISL SIGNING</span>
          </div>
          <div className="absolute top-3 right-3 flex items-center gap-2 font-technical-data text-[11px]">
            <span className="px-2 py-1 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 text-secondary flex items-center gap-1"><span className="material-symbols-outlined text-[13px]">gesture</span> {bufferedFrameCount} frames · {waitingPause ? "pause…" : "buffering"}</span>
          </div>
          <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 px-3 py-1.5 rounded bg-surface-container-lowest/90 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-outline">
            <span className="flex items-center gap-1"><span className="material-symbols-outlined text-[14px] text-primary">memory</span> 543 Holistic Coordinates · {connectionState}</span>
            <span className="text-on-surface-variant">{waitingPause ? `Waiting ${pauseMs} ms pause` : "Zero-Cloud · Landmarks only"}</span>
          </div>
          {captureStatus === "off" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-container-lowest/60 backdrop-blur-sm">
              <span className="material-symbols-outlined text-5xl text-on-surface-variant/30 mb-2">videocam_off</span>
              <p className="font-body-sm text-sm text-on-surface-variant">Frame upper body, hands, and face.</p>
            </div>
          )}
        </div>
        <div className="flex flex-col gap-2 bg-surface-container-low rounded-lg px-3 py-2 border border-outline-variant/20">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="font-technical-data text-xs text-on-surface-variant">{captureNote} {captureStatus !== "off" && `(${bufferedFrameCount} frames)`} · Backend {connectionState}</p>
            <div className="flex gap-2">
              {captureStatus === "off" || captureStatus === "error"
                ? <button onClick={() => void startCamera()} className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-primary text-on-primary font-label-md text-sm hover:bg-[#1ABC9C] transition-colors"><span className="material-symbols-outlined text-[16px]">videocam</span> Start</button>
                : <button onClick={stopCamera} className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded bg-surface-container-high text-on-surface font-label-sm text-sm hover:bg-surface-container-highest"><span className="material-symbols-outlined text-[15px]">pause_circle</span> Stop</button>}
              <button onClick={() => finishSentence("manual")} disabled={bufferedFrameCount === 0 || captureStatus === "loading"} className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-surface-container-high text-on-surface font-label-md text-sm border border-outline-variant/40 hover:border-primary disabled:opacity-40"><span className="material-symbols-outlined text-[16px]">send</span> Send</button>
            </div>
          </div>
          <div className="flex items-center gap-2 font-technical-data text-xs text-on-surface-variant">
            <span>Pause {pauseMs} ms</span>
            <input type="range" min={300} max={1200} step={50} value={pauseMs} onChange={e=>setPauseMs(Number(e.target.value))} className="flex-1 accent-primary h-1" />
            <span className="text-outline text-[11px]">short ↔ conversational</span>
          </div>
        </div>
        {error && <div className="rounded bg-error-container/20 border border-error/30 px-3 py-2 font-technical-data text-xs text-error">{error}</div>}
      </section>

      <section className="lg:col-span-5 flex flex-col gap-3 bg-surface-container-low rounded-xl p-4 shadow-lg border border-outline-variant/20">
        <div className="flex items-center justify-between">
          <h2 className="font-headline-md text-headline-md text-on-surface tracking-tight">ENGLISH CAPTION</h2>
          <span className="px-2 py-0.5 rounded bg-primary-container/20 text-primary font-technical-data text-[10px] tracking-wider uppercase font-semibold">Real-Time</span>
        </div>
        <div className="flex items-center gap-1 self-end">
          <button onClick={() => navigator.clipboard.writeText(caption)} disabled={!caption} className="p-1.5 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface disabled:opacity-30"><span className="material-symbols-outlined text-[18px]">content_copy</span></button>
          <button onClick={() => caption && window.speechSynthesis.speak(new SpeechSynthesisUtterance(caption))} disabled={!caption} className="p-1.5 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface disabled:opacity-30"><span className="material-symbols-outlined text-[18px]">volume_up</span></button>
          <button onClick={() => setTranscript([])} className="p-1.5 rounded hover:bg-surface-container text-on-surface-variant"><span className="material-symbols-outlined text-[18px]">backspace</span></button>
        </div>
        <div className="w-full bg-surface-container rounded-lg p-4 flex flex-col gap-3 border border-outline-variant/20">
          {caption ? (
            <>
              <blockquote className={`font-headline-lg text-headline-lg leading-tight tracking-tight ${captionMeta?.is_fallback ? "text-on-surface-variant italic" : "text-on-surface"}`}>“{caption}”</blockquote>
              {captionMeta?.is_fallback && <span className="font-technical-data text-xs text-secondary self-start px-2 py-0.5 rounded bg-secondary/10 border border-secondary/20">Demo/fallback · {captionMeta.gated_reason ?? "low confidence"} — try demo: Hello / Thank you / Yes / No</span>}
            </>
          ) : <p className="font-body-md text-body-md text-on-surface-variant italic flex items-center gap-2"><span className="material-symbols-outlined text-secondary">translate</span> Translated caption will appear here.</p>}
          {captionLatency !== null && <span className="font-technical-data text-xs text-primary self-end">{captionLatency.toFixed(0)} ms · {captionMeta?.is_fallback ? `Fallback ${captionMeta.confidence?.toFixed(2) ?? ""}` : "Transformer beam-5"} {captionMeta?.confidence != null ? `· conf ${captionMeta.confidence.toFixed(2)}` : ""}</span>}
          {glossTokens.length > 0 && (
            <div className="flex flex-col gap-1.5 pt-2 border-t border-outline-variant/20">
              <span className="font-technical-data text-[10px] uppercase tracking-widest text-on-surface-variant">Extracted Gloss Grammar</span>
              <div className="flex flex-wrap gap-1">{glossTokens.map((tok: string, i: number) => (<span key={i} className="px-2.5 py-1 rounded bg-surface-container-high text-primary font-technical-data text-xs tracking-wide uppercase">{tok}</span>))}</div>
            </div>
          )}
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between"><span className="font-technical-data text-xs text-on-surface-variant uppercase tracking-wider">Recent Utterances</span><span className="font-technical-data text-[11px] text-outline-variant">Session Log #{transcript.length}</span></div>
          <div className="flex flex-col gap-2 max-h-52 overflow-auto">
            {transcript.length === 0 ? <p className="font-body-sm text-sm text-on-surface-variant/60 py-2">No utterances yet.</p> : transcript.slice(-4).reverse().map((e: TranscriptEntry) => (
              <div key={e.id} className={`p-3 rounded bg-surface-container-lowest/80 border border-outline-variant/20 flex flex-col gap-1 ${e.dir === "error" ? "border-error/40 bg-error-container/10" : ""}`}>
                <div className="flex items-center justify-between font-technical-data text-[11px] text-on-surface-variant"><span>{e.dir}</span><span className="text-primary">{e.latency ? `${e.latency.toFixed(0)} ms` : ""}</span></div>
                <p className="font-body-md text-sm text-on-surface line-clamp-2">{e.text}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded bg-error-container/10 border border-error/20 px-3 py-2 flex gap-2">
          <span className="material-symbols-outlined text-error text-[16px]">warning</span>
          <p className="font-technical-data text-xs text-error">Research prototype — WER ~1.02, BLEU 0.31 (GRU) / 0.16 (Transformer). Verify every caption. See Research & Benchmarks.</p>
        </div>
      </section>
    </div>
  )
}
