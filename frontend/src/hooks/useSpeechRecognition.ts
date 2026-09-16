import { useCallback, useEffect, useRef, useState } from "react"

// ponytail: browser Web Speech API only — no backend Whisper, no dep. Upgrade path: server Whisper if accuracy matters.
// Web Speech API has no stable TS lib — use `any` for the ctor, keep the file untyped.
type SpeechState = "idle" | "listening" | "unsupported" | "error"

interface STTOptions {
  lang?: string
  onResult?: (text: string) => void
}

export function useSpeechRecognition({ lang = "en-IN", onResult }: STTOptions = {}) {
  const [state, setState] = useState<SpeechState>("idle")
  const [error, setError] = useState<string | null>(null)
  const recRef = useRef<any>(null)

  const supported = typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)

  useEffect(() => {
    if (!supported) setState("unsupported")
  }, [supported])

  const start = useCallback(() => {
    if (!supported) { setState("unsupported"); return }
    try {
      const w = window as unknown as Record<string, any>
      const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition
      if (!Ctor) { setState("unsupported"); return }
      const rec: any = new Ctor()
      rec.lang = lang
      rec.interimResults = false
      rec.continuous = false
      rec.maxAlternatives = 1
      rec.onstart = () => { setState("listening"); setError(null) }
      rec.onresult = (e: any) => {
        const t = e.results[0]?.[0]?.transcript ?? ""
        if (t) onResult?.(t)
        setState("idle")
      }
      rec.onerror = (e: any) => { setError(e.error || "mic error"); setState("error") }
      rec.onend = () => setState((s: SpeechState) => s === "listening" ? "idle" : s)
      recRef.current = rec
      rec.start()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "mic error")
      setState("error")
    }
  }, [lang, onResult, supported])

  const stop = useCallback(() => {
    try { recRef.current?.stop() } catch { /* ignore */ }
    setState("idle")
  }, [])

  useEffect(() => () => { try { recRef.current?.abort() } catch { /* ignore */ } }, [])

  return { state, error, supported, start, stop }
}
