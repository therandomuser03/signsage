import { useCallback, useEffect, useRef, useState } from "react"
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision"
import {
  flattenHolisticLandmarks,
  hasReliableSignLandmarks,
  poseMotion,
  TEMPORAL_MIN_FRAMES,
  TEMPORAL_MOTION_THRESHOLD,
  TEMPORAL_PAUSE_MS,
  type HolisticLandmarks,
} from "../../lib/mediapipe/landmarks"

type CaptureStatus = "off" | "loading" | "ready" | "warning" | "error"

const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
  [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
]

const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
]

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"

const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

/*
 * Important:
 * Vite exposes an empty environment variable as an empty string.
 *
 * Therefore:
 *
 *   import.meta.env.VITE_MEDIAPIPE_MODEL_URL ?? DEFAULT_MODEL_URL
 *
 * does NOT fall back when the env variable is "".
 *
 * Using trim() || DEFAULT_MODEL_URL ensures that an empty or whitespace-only
 * environment variable falls back to the known-good default model URL.
 */
const MODEL_URL =
  import.meta.env.VITE_MEDIAPIPE_MODEL_URL?.trim() || DEFAULT_MODEL_URL

function drawHolisticOverlay(
  canvas: HTMLCanvasElement | null,
  landmarks: HolisticLandmarks,
) {
  if (!canvas) return
  const bounds = canvas.getBoundingClientRect()
  const ratio = window.devicePixelRatio || 1
  canvas.width = Math.max(1, Math.round(bounds.width * ratio))
  canvas.height = Math.max(1, Math.round(bounds.height * ratio))
  const context = canvas.getContext("2d")
  if (!context) return
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  context.clearRect(0, 0, bounds.width, bounds.height)

  const project = (point: { x: number; y: number }) => ({
    x: point.x * bounds.width,
    y: point.y * bounds.height,
  })
  const drawComponent = (
    points: HolisticLandmarks["pose"],
    connections: ReadonlyArray<readonly [number, number]>,
    color: string,
    radius: number,
  ) => {
    if (!points?.length) return
    context.strokeStyle = color
    context.fillStyle = color
    context.lineWidth = 1.5
    context.globalAlpha = 0.28
    for (const [start, end] of connections) {
      const first = points[start]
      const second = points[end]
      if (!first || !second) continue
      const from = project(first)
      const to = project(second)
      context.beginPath()
      context.moveTo(from.x, from.y)
      context.lineTo(to.x, to.y)
      context.stroke()
    }
    context.globalAlpha = 0.38
    for (const point of points) {
      const projected = project(point)
      context.beginPath()
      context.arc(projected.x, projected.y, radius, 0, Math.PI * 2)
      context.fill()
    }
  }

  drawComponent(landmarks.pose, POSE_CONNECTIONS, "#61e7cb", 2.5)
  drawComponent(landmarks.leftHand, HAND_CONNECTIONS, "#7dd3fc", 3)
  drawComponent(landmarks.rightHand, HAND_CONNECTIONS, "#7dd3fc", 3)
  drawComponent(landmarks.face, [], "#f0abfc", 0.75)
  context.globalAlpha = 1
}

type TranscriptEntry = {
  id: string
  dir: string
  text: string
  latency?: number
}

export function SignModePanel({
  caption,
  captionLatency,
  captionMeta,
  error,
  send,
  connectionState,
  transcript,
  setTranscript,
}: {
  caption: string
  captionLatency: number | null
  captionMeta?: {
    confidence?: number
    is_fallback?: boolean
    gated_reason?: string | null
  } | null
  error: string | null
  send: (msg: Record<string, unknown>) => boolean
  connectionState: string
  transcript: TranscriptEntry[]
  setTranscript: React.Dispatch<
    React.SetStateAction<TranscriptEntry[]>
  >
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const landmarkerRef = useRef<HolisticLandmarker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameRequestRef = useRef<number | null>(null)

  const bufferedFramesRef = useRef<number[][]>([])
  const previousFrameRef = useRef<number[] | null>(null)
  const pauseSinceRef = useRef<number | null>(null)

  const finishSentenceRef = useRef<
    (t: "manual" | "motion_pause") => void
  >(() => undefined)

  const [captureStatus, setCaptureStatus] =
    useState<CaptureStatus>("off")

  const [captureNote, setCaptureNote] =
    useState("Camera off")

  const [bufferedFrameCount, setBufferedFrameCount] =
    useState(0)

  const [pauseMs, setPauseMs] =
    useState(TEMPORAL_PAUSE_MS)

  const [waitingPause, setWaitingPause] =
    useState(false)

  const pauseMsRef = useRef(pauseMs)

  useEffect(() => {
    pauseMsRef.current = pauseMs
  }, [pauseMs])

  /*
   * Finish the current sentence and send the buffered landmark
   * sequence to the backend.
   */
  const finishSentence = useCallback(
    (trigger: "manual" | "motion_pause") => {
      const raw = bufferedFramesRef.current

      if (raw.length === 0) {
        setCaptureNote(
          "No reliable landmarks captured yet."
        )
        return
      }

      /*
       * Clear the current buffer before sending so that new frames
       * can begin accumulating immediately.
       */
      bufferedFramesRef.current = []
      previousFrameRef.current = null
      pauseSinceRef.current = null

      setBufferedFrameCount(0)
      setWaitingPause(false)

      /*
       * Send every raw frame using the API contract:
       *
       * landmarks.frame
       *   timestamp_ms
       *   landmarks: 1629 values
       *
       * followed by:
       *
       * sentence.boundary
       *
       * IMPORTANT: The backend applies canonical normalization exactly once
       * through prepare_landmarks(..., input_mode="raw"). Do not normalize
       * here to avoid double-normalization / distribution shift.
       */
      const sent =
        raw.every((landmarks) =>
          send({
            type: "landmarks.frame",
            timestamp_ms: Math.round(
              performance.now()
            ),
            landmarks,
          })
        ) &&
        send({
          type: "sentence.boundary",
          trigger,
        })

      setCaptureNote(
        sent
          ? "Sentence sent. Waiting for caption."
          : "Backend unavailable."
      )

      if (!sent) {
        setCaptureStatus("error")
      }
    },
    [send]
  )

  useEffect(() => {
    finishSentenceRef.current = finishSentence
  }, [finishSentence])

  /*
   * Stop camera, MediaPipe and animation loop.
   */
  const stopCamera = useCallback(() => {
    if (frameRequestRef.current !== null) {
      window.cancelAnimationFrame(
        frameRequestRef.current
      )
    }

    frameRequestRef.current = null

    landmarkerRef.current?.close()
    landmarkerRef.current = null

    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop())

    streamRef.current = null

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    const overlay = overlayRef.current
    if (overlay) {
      overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height)
    }

    bufferedFramesRef.current = []
    previousFrameRef.current = null
    pauseSinceRef.current = null

    setBufferedFrameCount(0)
    setWaitingPause(false)

    setCaptureStatus("off")
    setCaptureNote("Camera off")
  }, [])

  /*
   * Start camera and initialize MediaPipe Holistic.
   */
  const startCamera = useCallback(async () => {
    if (connectionState !== "connected") {
      setCaptureStatus("error")
      setCaptureNote(
        "Backend socket is unavailable."
      )
      return
    }

    setCaptureStatus("loading")
    setCaptureNote("Loading camera...")

    try {
      /*
       * Request the user's camera.
       *
       * Raw camera frames remain inside the browser.
       * Only extracted landmark coordinates are sent
       * to the backend.
       */
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: {
              ideal: 1280,
            },
            height: {
              ideal: 720,
            },
          },
          audio: false,
        })

      const video = videoRef.current

      if (!video) {
        throw new Error(
          "Camera preview unavailable."
        )
      }

      streamRef.current = stream
      video.srcObject = stream

      await video.play()

      /*
       * Load the MediaPipe WASM runtime.
       */
      const vision =
        await FilesetResolver.forVisionTasks(
          WASM_ROOT
        )

      /*
       * Create the Holistic Landmarker.
       *
       * MODEL_URL is guaranteed to be non-empty because
       * the constant above uses:
       *
       * env value?.trim() || DEFAULT_MODEL_URL
       */
      const landmarker =
        await HolisticLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath: MODEL_URL,
            },

            runningMode: "VIDEO",

            minFaceDetectionConfidence: 0.5,
            minPoseDetectionConfidence: 0.5,
            minHandLandmarksConfidence: 0.5,
          }
        )

      landmarkerRef.current = landmarker

      setCaptureStatus("ready")
      setCaptureNote(
        "Signing. Pause after sentence or press Send."
      )

      /*
       * Process the camera continuously.
       */
      const processFrame = () => {
        const videoElement = videoRef.current
        const holistic = landmarkerRef.current

        if (!videoElement || !holistic) {
          return
        }

        /*
         * MediaPipe VIDEO mode requires a timestamp.
         */
        const timestamp = performance.now()

        const result =
          holistic.detectForVideo(
            videoElement,
            timestamp
          )

        /*
         * Extract the first detected set of:
         *
         * Pose:       33
         * Face:      468
         * Left hand:  21
         * Right hand: 21
         *
         * Total:     543 landmarks
         */
        const landmarks: HolisticLandmarks = {
          pose: result.poseLandmarks[0],
          face: result.faceLandmarks[0],
          leftHand:
            result.leftHandLandmarks[0],
          rightHand:
            result.rightHandLandmarks[0],
        }

        drawHolisticOverlay(overlayRef.current, landmarks)

        /*
         * Only use frames where the required sign
         * landmarks are reliable enough.
         */
        if (
          hasReliableSignLandmarks(
            landmarks
          )
        ) {
          /*
           * Do not use stale React state inside the
           * animation loop. The callback may run many
           * times per second.
           */
          setCaptureStatus((current) => {
            if (current !== "ready") {
              return "ready"
            }

            return current
          })

          /*
           * Flatten the 543 landmarks into the
           * expected 1629-value XYZ representation.
           */
          const frame =
            flattenHolisticLandmarks(
              landmarks
            )

          /*
           * Store the frame locally.
           *
           * Nothing is transmitted yet. The browser
           * keeps collecting frames until a sentence
           * boundary occurs.
           */
          bufferedFramesRef.current.push(
            frame
          )

          setBufferedFrameCount(
            bufferedFramesRef.current.length
          )

          /*
           * Estimate motion between consecutive
           * landmark frames.
           */
          const motion =
            previousFrameRef.current
              ? poseMotion(
                  previousFrameRef.current,
                  frame
                )
              : Infinity

          previousFrameRef.current = frame

          /*
           * Detect a sufficiently long pause.
           */
          if (
            motion <
              TEMPORAL_MOTION_THRESHOLD &&
            bufferedFramesRef.current.length >=
              TEMPORAL_MIN_FRAMES
          ) {
            pauseSinceRef.current ??=
              performance.now()

            setWaitingPause(true)

            if (
              performance.now() -
                pauseSinceRef.current >=
                pauseMsRef.current
            ) {
              finishSentenceRef.current(
                "motion_pause"
              )
            }
          } else {
            pauseSinceRef.current = null
            setWaitingPause(false)
          }
        } else {
          /*
           * MediaPipe isn't detecting enough
           * reliable landmarks.
           */
          setCaptureStatus("warning")

          setCaptureNote(
            "Move your face, hands, or upper body into view."
          )
        }

        /*
         * Schedule the next frame.
         */
        frameRequestRef.current =
          window.requestAnimationFrame(
            processFrame
          )
      }

      frameRequestRef.current =
        window.requestAnimationFrame(
          processFrame
        )
    } catch (caught) {
      /*
       * Make sure camera/MediaPipe resources are
       * released if initialization fails.
       */
      stopCamera()

      setCaptureStatus("error")

      setCaptureNote(
        caught instanceof Error
          ? caught.message
          : "Camera error."
      )
    }
  }, [
    connectionState,
    stopCamera,
  ])

  /*
   * Clean everything up when this component
   * unmounts.
   */
  useEffect(() => {
    return stopCamera
  }, [stopCamera])

  const glossTokens = caption
    ? caption
        .split(/\s+/)
        .slice(0, 8)
    : []

  return (
    <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
      <section className="lg:col-span-7 flex flex-col gap-2">
        <div className="relative w-full aspect-16/10 bg-surface-container-lowest rounded-xl overflow-hidden shadow-2xl border border-outline-variant/20 flex items-center justify-center">
          <video
            ref={videoRef}
            muted
            playsInline
            className="absolute inset-0 w-full h-full object-cover -scale-x-100"
          />

          <canvas
            ref={overlayRef}
            className="absolute inset-0 w-full h-full pointer-events-none -scale-x-100"
            aria-label="MediaPipe Holistic landmark overlay"
          />

          <div className="absolute top-3 left-3 flex flex-col gap-1.5">
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-primary">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  captureStatus === "ready"
                    ? "bg-primary animate-pulse"
                    : captureStatus ===
                        "warning"
                      ? "bg-error"
                      : "bg-outline"
                }`}
              />

              {captureStatus === "ready"
                ? "TRACKING"
                : captureStatus.toUpperCase()}

              {" · "}

              {captureStatus === "ready"
                ? "1080p"
                : captureNote.slice(0, 22)}
            </span>

            <span className="inline-flex px-2 py-0.5 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-on-surface-variant">
              TARGET: CONTINUOUS ISL SIGNING
            </span>
          </div>

          <div className="absolute top-3 right-3 flex items-center gap-2 font-technical-data text-[11px]">
            <span className="px-2 py-1 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 text-secondary flex items-center gap-1">
              <span className="material-symbols-outlined text-[13px]">
                gesture
              </span>

              {bufferedFrameCount} frames ·{" "}
              {waitingPause
                ? "pause…"
                : "buffering"}
            </span>
          </div>

          <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 px-3 py-1.5 rounded bg-surface-container-lowest/90 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-outline">
            <span className="flex items-center gap-1">
              <span className="material-symbols-outlined text-[14px] text-primary">
                memory
              </span>

              543 Holistic Coordinates ·{" "}
              {connectionState}
            </span>

            <span className="text-on-surface-variant">
              {waitingPause
                ? `Waiting ${pauseMs} ms pause`
                : "Zero-Cloud · Landmarks only"}
            </span>
          </div>

          {captureStatus === "off" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-container-lowest/60 backdrop-blur-sm">
              <span className="material-symbols-outlined text-5xl text-on-surface-variant/30 mb-2">
                videocam_off
              </span>

              <p className="font-body-sm text-sm text-on-surface-variant">
                Frame upper body, hands, and face.
              </p>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2 bg-surface-container-low rounded-lg px-3 py-2 border border-outline-variant/20">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="font-technical-data text-xs text-on-surface-variant">
              {captureNote}{" "}
              {captureStatus !== "off" &&
                `(${bufferedFrameCount} frames)`}{" "}
              · Backend {connectionState}
            </p>

            <div className="flex gap-2">
              {captureStatus === "off" ||
              captureStatus === "error" ? (
                <button
                  onClick={() =>
                    void startCamera()
                  }
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-primary text-on-primary font-label-md text-sm hover:bg-[#1ABC9C] transition-colors"
                >
                  <span className="material-symbols-outlined text-[16px]">
                    videocam
                  </span>

                  Start
                </button>
              ) : (
                <button
                  onClick={stopCamera}
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded bg-surface-container-high text-on-surface font-label-sm text-sm hover:bg-surface-container-highest"
                >
                  <span className="material-symbols-outlined text-[15px]">
                    pause_circle
                  </span>

                  Stop
                </button>
              )}

              <button
                onClick={() =>
                  finishSentence("manual")
                }
                disabled={
                  bufferedFrameCount === 0 ||
                  captureStatus === "loading"
                }
                className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-surface-container-high text-on-surface font-label-md text-sm border border-outline-variant/40 hover:border-primary disabled:opacity-40"
              >
                <span className="material-symbols-outlined text-[16px]">
                  send
                </span>

                Send
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2 font-technical-data text-xs text-on-surface-variant">
            <span>
              Pause {pauseMs} ms
            </span>

            <input
              type="range"
              min={300}
              max={1200}
              step={50}
              value={pauseMs}
              onChange={(e) =>
                setPauseMs(
                  Number(e.target.value)
                )
              }
              className="flex-1 accent-primary h-1"
            />

            <span className="text-outline text-[11px]">
              short ↔ conversational
            </span>
          </div>
        </div>

        {error && (
          <div className="rounded bg-error-container/20 border border-error/30 px-3 py-2 font-technical-data text-xs text-error">
            {error}
          </div>
        )}
      </section>

      <section className="lg:col-span-5 flex flex-col gap-3 bg-surface-container-low rounded-xl p-4 shadow-lg border border-outline-variant/20">
        <div className="flex items-center justify-between">
          <h2 className="font-headline-md text-headline-md text-on-surface tracking-tight">
            ENGLISH CAPTION
          </h2>

          <span className="px-2 py-0.5 rounded bg-primary-container/20 text-primary font-technical-data text-[10px] tracking-wider uppercase font-semibold">
            Real-Time
          </span>
        </div>

        <div className="flex items-center gap-1 self-end">
          <button
            onClick={() =>
              navigator.clipboard.writeText(
                caption
              )
            }
            disabled={!caption}
            className="p-1.5 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface disabled:opacity-30"
          >
            <span className="material-symbols-outlined text-[18px]">
              content_copy
            </span>
          </button>

          <button
            onClick={() =>
              caption &&
              window.speechSynthesis.speak(
                new SpeechSynthesisUtterance(
                  caption
                )
              )
            }
            disabled={!caption}
            className="p-1.5 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface disabled:opacity-30"
          >
            <span className="material-symbols-outlined text-[18px]">
              volume_up
            </span>
          </button>

          <button
            onClick={() =>
              setTranscript([])
            }
            className="p-1.5 rounded hover:bg-surface-container text-on-surface-variant"
          >
            <span className="material-symbols-outlined text-[18px]">
              backspace
            </span>
          </button>
        </div>

        <div className="w-full bg-surface-container rounded-lg p-4 flex flex-col gap-3 border border-outline-variant/20">
          {caption ? (
            <>
              <blockquote
                className={`font-headline-lg text-headline-lg leading-tight tracking-tight ${
                  captionMeta?.is_fallback
                    ? "text-on-surface-variant italic"
                    : "text-on-surface"
                }`}
              >
                “{caption}”
              </blockquote>

              {captionMeta?.is_fallback && (
                <span className="font-technical-data text-xs text-secondary self-start px-2 py-0.5 rounded bg-secondary/10 border border-secondary/20">
                  Demo/fallback ·{" "}
                  {captionMeta.gated_reason ??
                    "low confidence"}{" "}
                  — try demo: Hello / Thank you /
                  Yes / No
                </span>
              )}
            </>
          ) : (
            <p className="font-body-md text-body-md text-on-surface-variant italic flex items-center gap-2">
              <span className="material-symbols-outlined text-secondary">
                translate
              </span>

              Translated caption will appear
              here.
            </p>
          )}

          {captionLatency !== null && (
            <span className="font-technical-data text-xs text-primary self-end">
              {captionLatency.toFixed(0)} ms ·{" "}
              {captionMeta?.is_fallback
                ? `Fallback ${
                    captionMeta.confidence?.toFixed(
                      2
                    ) ?? ""
                  }`
                : "Transformer beam-5"}{" "}
              {captionMeta?.confidence !=
                null &&
                `· conf ${captionMeta.confidence.toFixed(
                  2
                )}`}
            </span>
          )}

          {glossTokens.length > 0 && (
            <div className="flex flex-col gap-1.5 pt-2 border-t border-outline-variant/20">
              <span className="font-technical-data text-[10px] uppercase tracking-widest text-on-surface-variant">
                Extracted Gloss Grammar
              </span>

              <div className="flex flex-wrap gap-1">
                {glossTokens.map(
                  (
                    tok: string,
                    i: number
                  ) => (
                    <span
                      key={i}
                      className="px-2.5 py-1 rounded bg-surface-container-high text-primary font-technical-data text-xs tracking-wide uppercase"
                    >
                      {tok}
                    </span>
                  )
                )}
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="font-technical-data text-xs text-on-surface-variant uppercase tracking-wider">
              Recent Utterances
            </span>

            <span className="font-technical-data text-[11px] text-outline-variant">
              Session Log #{transcript.length}
            </span>
          </div>

          <div className="flex flex-col gap-2 max-h-52 overflow-auto">
            {transcript.length === 0 ? (
              <p className="font-body-sm text-sm text-on-surface-variant/60 py-2">
                No utterances yet.
              </p>
            ) : (
              transcript
                .slice(-4)
                .reverse()
                .map(
                  (
                    e: TranscriptEntry
                  ) => (
                    <div
                      key={e.id}
                      className={`p-3 rounded bg-surface-container-lowest/80 border border-outline-variant/20 flex flex-col gap-1 ${
                        e.dir === "error"
                          ? "border-error/40 bg-error-container/10"
                          : ""
                      }`}
                    >
                      <div className="flex items-center justify-between font-technical-data text-[11px] text-on-surface-variant">
                        <span>
                          {e.dir}
                        </span>

                        <span className="text-primary">
                          {e.latency
                            ? `${e.latency.toFixed(
                                0
                              )} ms`
                            : ""}
                        </span>
                      </div>

                      <p className="font-body-md text-sm text-on-surface line-clamp-2">
                        {e.text}
                      </p>
                    </div>
                  )
                )
            )}
          </div>
        </div>

        <div className="rounded bg-error-container/10 border border-error/20 px-3 py-2 flex gap-2">
          <span className="material-symbols-outlined text-error text-[16px]">
            warning
          </span>

          <p className="font-technical-data text-xs text-error">
            Research prototype — WER ~1.02,
            BLEU 0.31 (GRU) / 0.16
            (Transformer). Verify every caption.
            See Research & Benchmarks.
          </p>
        </div>
      </section>
    </div>
  )
}
