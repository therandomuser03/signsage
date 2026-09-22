import { useCallback, useEffect, useState } from "react"
import { useTranslationSocket } from "../hooks/use-translation-socket"
import { SignModePanel } from "../components/workspace/SignModePanel"
import { TextModePanel } from "../components/workspace/TextModePanel"
import { getSupabase, useAuth } from "../contexts/AuthContext"

type ConversationMode = "sign" | "text"
type TranscriptEntry = { id: string; dir: string; text: string; latency?: number }

export function WorkspacePage({ initialMode, navigate }: { initialMode: ConversationMode; navigate: (hash: string) => void }) {
  const { user } = useAuth()
  const mode = initialMode
  const [caption, setCaption] = useState("")
  const [captionLatency, setCaptionLatency] = useState<number | null>(null)
  const [captionMeta, setCaptionMeta] = useState<{ match_score?: number | null; distance?: number | null; is_fallback?: boolean; gated_reason?: string | null } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [transcript, setTranscript] = useState<TranscriptEntry[]>(() => {
    try { const raw = localStorage.getItem("signsage-transcript"); return raw ? JSON.parse(raw) : [] } catch { return [] }
  })
  const pushEntry = useCallback((e: Omit<TranscriptEntry, "id">) => setTranscript(t => {
    // ponytail: avoid spamming same prediction while holding gesture / mode collapse repeats
    const last = t[t.length - 1]
    if (last && last.text === e.text && last.dir === e.dir) return t
    const next = [...t, { ...e, id: `${Date.now()}-${Math.random().toString(16).slice(2)}` }].slice(-20)
    try { localStorage.setItem("signsage-transcript", JSON.stringify(next)) } catch { /* ignore */ }
    return next
  }), [])
  useEffect(() => { try { localStorage.setItem("signsage-transcript", JSON.stringify(transcript)) } catch { /* ignore */ } }, [transcript])

  const { connectionState, send } = useTranslationSocket({
    onMessage: (message) => {
      if (message.type === "translation.text") {
        setCaption(message.text); setCaptionLatency(message.latency_ms); setCaptionMeta({ match_score: message.match_score, distance: message.distance, is_fallback: message.is_fallback, gated_reason: message.gated_reason }); setError(null); pushEntry({ dir: "sign→text", text: message.text, latency: message.latency_ms })
        // ponytail: best-effort user-associated record — RLS enforces isolation, localStorage stays primary
        if (user?.id) void getSupabase().from("translation_turns").insert({ user_id: user.id, dir: "sign→text", text: message.text, latency_ms: message.latency_ms }).then(()=>{}, ()=>{})
      }
      else if (message.type === "error") { const msg = `${message.code}: ${message.message}`; setError(msg); pushEntry({ dir: "error", text: msg }) }
    },
  })

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      <main className="w-full bg-surface min-h-[calc(100vh-4rem)]">
        <div className="flex flex-col w-full max-w-7xl mx-auto px-4 lg:px-8 py-8">
          <div className="w-full flex flex-col md:flex-row items-center justify-between gap-4 pb-4 mb-4 border-b border-outline-variant/20">
            <div className="flex items-center gap-2 bg-surface-container-lowest p-1 rounded-lg">
              <button onClick={() => navigate("/translate/sign")} className={`flex items-center gap-2 px-4 py-1.5 rounded-md font-label-md text-sm transition-all ${mode === "sign" ? "bg-primary text-on-primary shadow-sm" : "text-on-surface-variant hover:text-on-surface"}`}><span className="material-symbols-outlined text-[16px]">sync_alt</span> ISL → English</button>
              <button onClick={() => navigate("/translate/text")} className={`flex items-center gap-2 px-4 py-1.5 rounded-md font-label-md text-sm transition-all ${mode === "text" ? "bg-primary text-on-primary shadow-sm" : "text-on-surface-variant hover:text-on-surface"}`}><span className="material-symbols-outlined text-[15px]">model_training</span> English → ISL</button>
            </div>
            <div className="flex items-center gap-4 font-technical-data text-xs text-on-surface-variant">
              <span className="flex items-center gap-1.5"><span className={`w-1.5 h-1.5 rounded-full ${connectionState === "connected" ? "bg-primary animate-pulse" : "bg-error"}`} /> Backend: {connectionState}</span>
              <span className="hidden md:inline text-outline-variant">·</span>
              <span className="flex items-center gap-1"><span className="material-symbols-outlined text-[14px] text-primary">memory</span> {captionLatency !== null ? `${captionLatency.toFixed(0)} ms` : "— ms"} · Server-side inference</span>
            </div>
            <span className="hidden lg:inline px-2 py-1 rounded bg-surface-container-high border border-outline-variant/40 font-technical-data text-xs text-on-surface-variant">Camera video stays on-device · only landmark coordinates are sent</span>
          </div>
          {mode === "sign"
            ? <SignModePanel caption={caption} captionLatency={captionLatency} captionMeta={captionMeta} error={error} send={send} connectionState={connectionState} transcript={transcript} setTranscript={setTranscript} />
            : <TextModePanel transcript={transcript} />}
        </div>
      </main>
    </div>
  )
}
