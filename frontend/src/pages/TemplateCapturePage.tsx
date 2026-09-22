import { useCallback, useEffect, useRef, useState } from "react"
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision"
import {
  flattenHolisticLandmarks,
  hasReliableSignLandmarks,
  normalizeLandmarkSequence,
  type HolisticLandmarks,
} from "../lib/mediapipe/landmarks"

const PHRASES = ["Hello", "Thank you", "Yes", "No", "Please", "Sorry", "Help", "Good morning"]
const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
const NOFACE_START = 468 * 3
const NOFACE_END = 543 * 3

function nofaceFeatures(frame: number[]): number[] {
  return frame.slice(NOFACE_START, NOFACE_END)
}

export function TemplateCapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const landmarkerRef = useRef<HolisticLandmarker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameRequestRef = useRef<number | null>(null)
  const framesRef = useRef<number[][]>([])
  const capturingRef = useRef(false)
  const [phrase, setPhrase] = useState(PHRASES[0])
  const [status, setStatus] = useState("Camera off")
  const [frameCount, setFrameCount] = useState(0)
  const [capturing, setCapturing] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)

  const stop = useCallback(() => {
    if (frameRequestRef.current !== null) window.cancelAnimationFrame(frameRequestRef.current)
    frameRequestRef.current = null
    landmarkerRef.current?.close()
    landmarkerRef.current = null
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    setCameraReady(false)
    if (videoRef.current) videoRef.current.srcObject = null
    capturingRef.current = false
    setCapturing(false)
    setStatus("Camera off")
  }, [])

  const start = useCallback(async () => {
    try {
      setStatus("Loading camera and MediaPipe…")
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      if (!videoRef.current) throw new Error("Camera preview unavailable.")
      streamRef.current = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT)
      landmarkerRef.current = await HolisticLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "VIDEO",
        minFaceDetectionConfidence: 0.5,
        minPoseDetectionConfidence: 0.5,
        minHandLandmarksConfidence: 0.5,
      })
      setCameraReady(true)
      setStatus("Camera ready. Select Start recording.")
      const process = () => {
        const video = videoRef.current
        const landmarker = landmarkerRef.current
        if (!video || !landmarker) return
        const result = landmarker.detectForVideo(video, performance.now())
        const landmarks: HolisticLandmarks = {
          pose: result.poseLandmarks[0],
          face: result.faceLandmarks[0],
          leftHand: result.leftHandLandmarks[0],
          rightHand: result.rightHandLandmarks[0],
        }
        if (capturingRef.current && hasReliableSignLandmarks(landmarks)) {
          framesRef.current.push(flattenHolisticLandmarks(landmarks))
          setFrameCount(framesRef.current.length)
        }
        frameRequestRef.current = window.requestAnimationFrame(process)
      }
      frameRequestRef.current = window.requestAnimationFrame(process)
    } catch (error) {
      stop()
      setStatus(error instanceof Error ? error.message : "Camera error.")
    }
  }, [stop])

  useEffect(() => stop, [stop])

  const startRecording = () => {
    framesRef.current = []
    setFrameCount(0)
    capturingRef.current = true
    setCapturing(true)
    setStatus(`Recording ${phrase}. Perform one sign.`)
  }

  const stopRecording = () => {
    capturingRef.current = false
    setCapturing(false)
    const frames = framesRef.current
    if (frames.length < 8) {
      setStatus("Too few reliable frames. Record again with both hands visible.")
      return
    }
    const normalized = normalizeLandmarkSequence(frames).map(nofaceFeatures)
    const blob = new Blob([JSON.stringify({ phrase, frames: normalized }, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    const slug = phrase.toLowerCase().replace(/\s+/g, "_")
    link.href = url
    link.download = `${slug}_${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
    setStatus(`Saved ${frames.length} real landmark frames. Move file into backend/demo_prototypes/.`)
  }

  return (
    <main className="min-h-screen bg-surface p-6 text-on-surface">
      <div className="mx-auto flex max-w-5xl flex-col gap-5">
        <header>
          <h1 className="font-headline-lg">Developer template capture</h1>
          <p className="mt-2 max-w-2xl text-sm text-on-surface-variant">Records real MediaPipe landmarks from this camera. It never fabricates or replays templates.</p>
        </header>
        <div className="grid gap-5 lg:grid-cols-[1.3fr_0.7fr]">
          <div className="relative aspect-video overflow-hidden rounded-xl border border-outline-variant/30 bg-surface-container-lowest">
            <video ref={videoRef} muted playsInline className="h-full w-full object-cover -scale-x-100" />
            <div className="absolute bottom-3 left-3 rounded bg-black/70 px-3 py-2 text-xs">{status} · {frameCount} reliable frames</div>
          </div>
          <section className="flex flex-col gap-4 rounded-xl bg-surface-container p-4">
            <label className="flex flex-col gap-2 text-sm">Sign class<select value={phrase} onChange={event => setPhrase(event.target.value)} disabled={capturing} className="rounded bg-surface-container-lowest p-2 text-on-surface">{PHRASES.map(item => <option key={item}>{item}</option>)}</select></label>
            {!cameraReady ? <button onClick={() => void start()} className="rounded bg-primary px-4 py-2 text-on-primary">Start camera</button> : null}
            {cameraReady && !capturing ? <button onClick={startRecording} className="rounded bg-primary px-4 py-2 text-on-primary">Start recording</button> : null}
            {capturing ? <button onClick={stopRecording} className="rounded bg-primary px-4 py-2 text-on-primary">Stop and save template</button> : null}
            {cameraReady ? <button onClick={stop} className="rounded bg-surface-container-high px-4 py-2">Stop camera</button> : null}
            <p className="text-xs text-on-surface-variant">Record 5–10 separate examples per class. Saved files use normalized 225-feature sequences compatible with backend matcher.</p>
          </section>
        </div>
      </div>
    </main>
  )
}
