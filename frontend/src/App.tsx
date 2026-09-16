import { useCallback, useEffect, useRef, useState } from "react"
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision"
import { ArrowRight, Camera, CheckCircle2, Keyboard, Languages, Mic, MicOff, Moon, Send, Square, Sun, TriangleAlert, Volume2 } from "lucide-react"

import { PoseCanvas } from "./components/pose-canvas"
import { ThemeProvider, useTheme } from "./components/theme-provider"
import { useSpeechRecognition } from "./hooks/useSpeechRecognition"
import { useTranslationSocket } from "./hooks/use-translation-socket"
import { flattenHolisticLandmarks, hasReliableSignLandmarks, normalizeLandmarkSequence, poseMotion, type HolisticLandmarks } from "./lib/landmarks"

import { Button } from "./components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "./components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert"
import { Textarea } from "./components/ui/textarea"
import { Badge } from "./components/ui/badge"

type ConversationMode = "sign" | "text"
type CaptureStatus = "off" | "loading" | "ready" | "warning" | "error"

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
const MODEL_URL = import.meta.env.VITE_MEDIAPIPE_MODEL_URL
  ?? "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

// ponytail: hash routing with no dependency
function useHashRoute(): [string, (hash: string) => void] {
  const [route, setRoute] = useState(window.location.hash || "#/")
  useEffect(() => {
    const onHash = () => setRoute(window.location.hash || "#/")
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [])
  const navigate = useCallback((hash: string) => { window.location.hash = hash }, [])
  return [route, navigate]
}

function TopBar({ showBack, onBack }: { showBack?: boolean; onBack?: () => void }) {
  const { theme, setTheme } = useTheme()
  return (
    <header className="flex items-center justify-between py-4 mb-6 border-b">
      <a className="flex items-center gap-2 text-lg font-bold hover:opacity-80 transition-opacity" href="#/" aria-label="SignSage home">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Languages size={18} strokeWidth={2.5} />
        </div>
        <span>SignSage</span>
      </a>
      <div className="flex items-center gap-3">
        {showBack && <Button variant="ghost" size="sm" onClick={onBack}>← Back</Button>}
        <Badge variant="secondary" className="hidden sm:inline-flex">Research Prototype</Badge>
        <Button variant="outline" size="icon" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
          {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
        </Button>
      </div>
    </header>
  )
}

function LandingPage({ navigate }: { navigate: (hash: string) => void }) {
  return (
    <div className="mx-auto max-w-5xl px-4 pb-12">
      <TopBar />
      <section className="py-16 md:py-24 text-center">
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6">Bridge the communication gap.</h1>
        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
          SignSage translates between Indian Sign Language and English text in real time. Use your camera to sign, or type text to see the ISL pose animation.
        </p>
        
        <Alert variant="destructive" className="max-w-2xl mx-auto text-left mb-12">
          <TriangleAlert className="h-4 w-4" />
          <AlertTitle>Research Prototype Warning</AlertTitle>
          <AlertDescription>
            Current models can produce incorrect, repetitive, or near-static results. Do not use for medical, legal, or emergency communication.
          </AlertDescription>
        </Alert>

        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto text-left">
          <Card className="hover:border-primary transition-colors cursor-pointer group" onClick={() => navigate("#/translate/sign")}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-secondary flex items-center justify-center mb-4">
                <Camera className="h-6 w-6" />
              </div>
              <CardTitle>Sign to Text</CardTitle>
              <CardDescription>Open your camera, sign in ISL, and get English captions in real time.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center text-primary font-medium group-hover:translate-x-1 transition-transform">
                Open camera <ArrowRight className="ml-2 h-4 w-4" />
              </div>
            </CardContent>
          </Card>

          <Card className="hover:border-primary transition-colors cursor-pointer group" onClick={() => navigate("#/translate/text")}>
            <CardHeader>
              <div className="h-12 w-12 rounded-lg bg-secondary flex items-center justify-center mb-4">
                <Keyboard className="h-6 w-6" />
              </div>
              <CardTitle>Text to Sign</CardTitle>
              <CardDescription>Type English text and see the ISL pose animation generated by the model.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center text-primary font-medium group-hover:translate-x-1 transition-transform">
                Start typing <ArrowRight className="ml-2 h-4 w-4" />
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
      
      <footer className="text-center text-sm text-muted-foreground flex items-center justify-center gap-2 mt-8">
        <CheckCircle2 className="h-4 w-4" /> Camera video stays on this device. Only landmarks are sent.
      </footer>
    </div>
  )
}

type TranscriptEntry = { id: string; dir: "sign→text" | "text→pose" | "error"; text: string; gloss?: string; latency?: number; ts: number }

function SignSageWorkspace({ initialMode, navigate }: { initialMode: ConversationMode; navigate: (hash: string) => void }) {
  const [mode, setMode] = useState<ConversationMode>(initialMode)
  const [caption, setCaption] = useState("")
  const [captionLatency, setCaptionLatency] = useState<number | null>(null)
  const [pose, setPose] = useState<{ frames: number[][]; fps: number; gloss: string; latency: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([])
  const pushEntry = useCallback((e: Omit<TranscriptEntry,"id"|"ts">) => setTranscript((t) => [...t, { ...e, id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, ts: Date.now() }].slice(-20)), [])

  const { connectionState, send } = useTranslationSocket({
    onMessage: (message) => {
      if (message.type === "translation.text") {
        setCaption(message.text)
        setCaptionLatency(message.latency_ms)
        setError(null)
        pushEntry({ dir: "sign→text", text: message.text, latency: message.latency_ms })
      } else if (message.type === "translation.pose") {
        setPose({ frames: message.frames, fps: message.fps, gloss: message.gloss, latency: message.latency_ms })
        setError(null)
        pushEntry({ dir: "text→pose", text: message.gloss, gloss: message.gloss, latency: message.latency_ms })
      } else if (message.type === "error") {
        const msg = `${message.code}: ${message.message}`
        setError(msg)
        pushEntry({ dir: "error", text: msg })
      }
    },
  })

  return (
    <div className="mx-auto max-w-6xl px-4 pb-12 flex flex-col min-h-screen">
      <TopBar showBack onBack={() => navigate("#/")} />
      
      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <Tabs value={mode} onValueChange={(v) => {
            setMode(v as ConversationMode)
            navigate(v === "sign" ? "#/translate/sign" : "#/translate/text")
          }}>
            <TabsList>
              <TabsTrigger value="sign" className="flex items-center gap-2"><Camera className="h-4 w-4" /> Sign to Text</TabsTrigger>
              <TabsTrigger value="text" className="flex items-center gap-2"><Keyboard className="h-4 w-4" /> Text to Sign</TabsTrigger>
            </TabsList>
          </Tabs>
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${connectionState === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-muted-foreground font-medium">
              {connectionState === 'connected' ? 'Connected' : connectionState === 'connecting' ? 'Connecting...' : 'Offline'}
            </span>
          </div>
        </div>

        {mode === "sign" 
          ? <SignToTextPanel caption={caption} captionLatency={captionLatency} send={send} connectionState={connectionState} transcript={transcript} /> 
          : <TextToSignPanel pose={pose} error={error} send={send} connectionState={connectionState} transcript={transcript} />
        }
        {transcript.length > 0 && (
          <div className="mt-6 border rounded-lg bg-card p-3">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold">Conversation</h3>
              <Button variant="ghost" size="sm" onClick={() => setTranscript([])}>Clear</Button>
            </div>
            <div className="space-y-2 max-h-40 overflow-auto text-sm">
              {transcript.map((e) => (
                <div key={e.id} className={`flex justify-between gap-2 ${e.dir==="error" ? "text-destructive" : "text-muted-foreground"}`}>
                  <span><Badge variant="outline" className="mr-2 text-xs">{e.dir}</Badge>{e.text}{e.gloss && e.dir==="text→pose" ? ` — ${e.gloss}` : ""}</span>
                  <span className="shrink-0 text-xs">{e.latency ? `${e.latency.toFixed(0)}ms` : ""}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <footer className="text-center text-sm text-muted-foreground flex items-center justify-center gap-2 mt-6">
        <CheckCircle2 className="h-4 w-4" /> Camera video stays on this device. Only landmarks sent.
      </footer>
    </div>
  )
}

function SignToTextPanel({ caption, captionLatency, send, connectionState }: any) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const landmarkerRef = useRef<HolisticLandmarker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameRequestRef = useRef<number | null>(null)
  const bufferedFramesRef = useRef<number[][]>([])
  const previousFrameRef = useRef<number[] | null>(null)
  const pauseSinceRef = useRef<number | null>(null)
  const finishSentenceRef = useRef<(trigger: "manual" | "motion_pause") => void>(() => undefined)
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus>("off")
  const [captureNote, setCaptureNote] = useState("Camera off")
  const [bufferedFrameCount, setBufferedFrameCount] = useState(0)

  const finishSentence = useCallback((trigger: "manual" | "motion_pause") => {
    const rawFrames = bufferedFramesRef.current
    if (rawFrames.length === 0) {
      setCaptureNote("No reliable landmarks captured yet.")
      return
    }
    bufferedFramesRef.current = []
    previousFrameRef.current = null
    pauseSinceRef.current = null
    setBufferedFrameCount(0)
    const normalized = normalizeLandmarkSequence(rawFrames)
    const sent = normalized.every((landmarks) => send({ type: "landmarks.frame", timestamp_ms: Math.round(performance.now()), landmarks })) && send({ type: "sentence.boundary", trigger })
    setCaptureNote(sent ? "Sentence sent. Waiting for caption." : "Backend unavailable.")
    if (!sent) setCaptureStatus("error")
  }, [send])

  useEffect(() => { finishSentenceRef.current = finishSentence }, [finishSentence])

  const stopCamera = useCallback(() => {
    if (frameRequestRef.current !== null) window.cancelAnimationFrame(frameRequestRef.current)
    frameRequestRef.current = null
    landmarkerRef.current?.close()
    landmarkerRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    bufferedFramesRef.current = []
    previousFrameRef.current = null
    pauseSinceRef.current = null
    setBufferedFrameCount(0)
    setCaptureStatus("off")
    setCaptureNote("Camera off")
  }, [])

  const startCamera = useCallback(async () => {
    if (connectionState !== "connected") {
      setCaptureStatus("error")
      setCaptureNote("Backend socket is unavailable.")
      return
    }
    setCaptureStatus("loading")
    setCaptureNote("Loading camera...")
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      const video = videoRef.current
      if (!video) throw new Error("Camera preview unavailable.")
      streamRef.current = stream
      video.srcObject = stream
      await video.play()
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT)
      const landmarker = await HolisticLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL }, runningMode: "VIDEO", minFaceDetectionConfidence: 0.5, minPoseDetectionConfidence: 0.5, minHandLandmarksConfidence: 0.5,
      })
      landmarkerRef.current = landmarker
      setCaptureStatus("ready")
      setCaptureNote("Signing. Pause after sentence or press Send.")
      const processFrame = () => {
        const activeVideo = videoRef.current
        const activeLandmarker = landmarkerRef.current
        if (!activeVideo || !activeLandmarker) return
        const result = activeLandmarker.detectForVideo(activeVideo, performance.now())
        const landmarks: HolisticLandmarks = { pose: result.poseLandmarks[0], face: result.faceLandmarks[0], leftHand: result.leftHandLandmarks[0], rightHand: result.rightHandLandmarks[0] }
        if (hasReliableSignLandmarks(landmarks)) {
          const frame = flattenHolisticLandmarks(landmarks)
          bufferedFramesRef.current.push(frame)
          setBufferedFrameCount(bufferedFramesRef.current.length)
          const motion = previousFrameRef.current ? poseMotion(previousFrameRef.current, frame) : Infinity
          previousFrameRef.current = frame
          if (motion < 0.012 && bufferedFramesRef.current.length >= 15) {
            pauseSinceRef.current ??= performance.now()
            if (performance.now() - pauseSinceRef.current >= 650) finishSentenceRef.current("motion_pause")
          } else {
            pauseSinceRef.current = null
          }
        } else {
          setCaptureStatus("warning")
          setCaptureNote("Tracking needs upper body and at least one hand in frame.")
        }
        frameRequestRef.current = window.requestAnimationFrame(processFrame)
      }
      frameRequestRef.current = window.requestAnimationFrame(processFrame)
    } catch (caught) {
      stopCamera()
      setCaptureStatus("error")
      setCaptureNote(caught instanceof Error ? caught.message : "Camera error.")
    }
  }, [connectionState, stopCamera])

  useEffect(() => stopCamera, [stopCamera])

  return (
    <div className="grid lg:grid-cols-2 gap-6 flex-1">
      <Card className="flex flex-col overflow-hidden">
        <CardHeader className="pb-4 border-b">
          <div className="flex justify-between items-start">
            <div>
              <CardTitle>Sign Input</CardTitle>
              <CardDescription>Your turn</CardDescription>
            </div>
            <Badge variant={captureStatus === "ready" ? "default" : "secondary"}>
              {captureStatus === "ready" ? "Camera On" : captureStatus}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-4 bg-muted/30">
          <div className="relative flex-1 bg-secondary rounded-lg overflow-hidden border border-border flex items-center justify-center min-h-[300px]">
            <video ref={videoRef} muted playsInline className="absolute inset-0 w-full h-full object-cover -scale-x-100" />
            {captureStatus === "off" && (
              <div className="text-center text-muted-foreground z-10 flex flex-col items-center">
                <Camera className="h-10 w-10 mb-2 opacity-50" />
                <p>Frame upper body, hands, and face.</p>
              </div>
            )}
          </div>
          <div className="mt-4 flex flex-col sm:flex-row gap-4 items-center justify-between">
            <p className="text-sm text-muted-foreground">{captureNote} {captureStatus !== "off" && `(${bufferedFrameCount} frames)`}</p>
            <div className="flex gap-2">
              {captureStatus === "off" || captureStatus === "error" 
                ? <Button onClick={() => void startCamera()}><Camera className="mr-2 h-4 w-4" /> Start</Button> 
                : <Button variant="outline" onClick={stopCamera}><Square className="mr-2 h-4 w-4" /> Stop</Button>
              }
              <Button onClick={() => finishSentence("manual")} disabled={bufferedFrameCount === 0 || captureStatus === "loading"}>
                <Send className="mr-2 h-4 w-4" /> Send
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="flex flex-col overflow-hidden border-primary/20">
        <CardHeader className="pb-4 border-b bg-muted/10">
          <div className="flex justify-between items-start">
            <div>
              <CardTitle>English Translation</CardTitle>
              <CardDescription>Live caption</CardDescription>
            </div>
            <Button variant="ghost" size="icon" disabled={!caption} onClick={() => window.speechSynthesis.speak(new SpeechSynthesisUtterance(caption))}>
              <Volume2 className="h-5 w-5" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-6 items-center justify-center">
          {caption ? (
            <div className="text-center w-full">
              <p className="text-3xl font-bold tracking-tight mb-4 text-foreground">{caption}</p>
              {captionLatency !== null && <p className="text-sm text-muted-foreground">Inference: {captionLatency.toFixed(0)} ms</p>}
            </div>
          ) : (
            <div className="text-center text-muted-foreground flex flex-col items-center">
              <Languages className="h-12 w-12 mb-4 opacity-20" />
              <p>Your translated caption will appear here.</p>
            </div>
          )}
        </CardContent>
        <CardFooter className="bg-muted/30 border-t p-4">
          <Alert variant="destructive" className="w-full mb-0 bg-destructive/5 border-destructive/20">
            <TriangleAlert className="h-4 w-4 text-destructive" />
            <AlertDescription className="text-xs">
              Results must be checked by people communicating. Current model exhibits severe mode collapse.
            </AlertDescription>
          </Alert>
        </CardFooter>
      </Card>
    </div>
  )
}

function TextToSignPanel({ pose, error, send, connectionState }: any) {
  const [text, setText] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const translate = () => { const sent = send({ type: "text.input", text: text.trim() }); setIsSubmitting(sent) }
  const stt = useSpeechRecognition({ lang: "en-IN", onResult: (t) => setText(t) })
  
  useEffect(() => { if (pose || error) setIsSubmitting(false) }, [error, pose])

  return (
    <div className="grid lg:grid-cols-2 gap-6 flex-1">
      <Card className="flex flex-col overflow-hidden">
        <CardHeader className="pb-4 border-b">
          <CardTitle>Write a message</CardTitle>
          <CardDescription>English to ISL — type or speak</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-6 gap-3">
          <Textarea 
            placeholder="Type what you want to communicate…" 
            className="flex-1 min-h-[150px] resize-none text-lg p-4 bg-muted/20"
            value={text} 
            onChange={(e) => setText(e.target.value)} 
          />
          <div className="flex items-center gap-2">
            <Button
              variant={stt.state==="listening" ? "destructive" : "outline"}
              size="sm"
              onClick={() => stt.state==="listening" ? stt.stop() : stt.start()}
              disabled={stt.state==="unsupported"}
              aria-label={stt.state==="listening" ? "Stop listening" : "Start voice input"}
            >
              {stt.state==="listening" ? <MicOff className="mr-2 h-4 w-4" /> : <Mic className="mr-2 h-4 w-4" />}
              {stt.state==="listening" ? "Listening…" : stt.state==="unsupported" ? "Voice unsupported" : "Speak"}
            </Button>
            {stt.error && <span className="text-xs text-destructive">{stt.error}</span>}
            {stt.state==="unsupported" && <span className="text-xs text-muted-foreground">Use Chrome/Edge for voice.</span>}
          </div>
          {error && <Alert variant="destructive"><TriangleAlert className="h-4 w-4" /><AlertDescription className="text-xs break-words">{error}</AlertDescription></Alert>}
        </CardContent>
        <CardFooter className="border-t p-4 bg-muted/10 flex justify-between items-center">
          <p className="text-sm text-muted-foreground hidden sm:block">
            {connectionState === "connected" ? "Ready to translate." : "Socket offline."}
          </p>
          <Button onClick={translate} disabled={!text.trim() || connectionState !== "connected" || isSubmitting}>
            {isSubmitting ? "Generating…" : "Translate"} <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>

      <Card className="flex flex-col overflow-hidden border-primary/20">
        <CardHeader className="pb-4 border-b bg-muted/10">
          <CardTitle>ISL Pose Preview</CardTitle>
          <CardDescription>Sign animation</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-6 items-center justify-center bg-secondary/30 relative">
          {pose ? (
            <div className="w-full h-full flex flex-col items-center justify-center">
              <div className="w-full h-[300px] border bg-background rounded-lg overflow-hidden relative shadow-inner">
                <PoseCanvas frames={pose.frames} fps={pose.fps} />
              </div>
              <div className="flex flex-wrap gap-2 items-center justify-center mt-4">
                <span className="text-sm text-muted-foreground">Gloss: {pose.gloss} • {pose.latency.toFixed(0)} ms</span>
                <Badge variant="destructive" className="text-xs">research — near-static motion (6.7% gt)</Badge>
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground flex flex-col items-center">
              <Send className="h-12 w-12 mb-4 opacity-20" />
              <p>Generated pose animation will appear here.</p>
            </div>
          )}
        </CardContent>
        <CardFooter className="bg-muted/30 border-t p-4">
          <Alert variant="destructive" className="w-full mb-0 bg-destructive/5 border-destructive/20">
            <TriangleAlert className="h-4 w-4 text-destructive" />
            <AlertDescription className="text-xs">
              Current pose model is known to produce near-static limited motion.
            </AlertDescription>
          </Alert>
        </CardFooter>
      </Card>
    </div>
  )
}

function App() {
  const [route, navigate] = useHashRoute()
  return (
    <ThemeProvider defaultTheme="dark" storageKey="signsage-theme">
      {route.startsWith("#/translate") ? (
        <SignSageWorkspace initialMode={route === "#/translate/text" ? "text" : "sign"} navigate={navigate} />
      ) : (
        <LandingPage navigate={navigate} />
      )}
    </ThemeProvider>
  )
}

export default App
