import { useState } from "react"
import { useSpeechRecognition } from "../../hooks/useSpeechRecognition"

const SUPPORTED_PHRASES = [
  "Hello",
  "Thank you",
  "Yes",
  "No",
  "Please",
  "Sorry",
  "Help",
  "Good morning",
]

export function TextModePanel({ transcript }: {
  transcript: { id: string; dir: string; text: string }[]
}) {
  const [text, setText] = useState("")
  const [message, setMessage] = useState<string | null>(null)
  const stt = useSpeechRecognition({ lang: "en-IN", onResult: setText })
  const tokens = text.trim().split(/\s+/).filter(Boolean)

  const translate = () => {
    setMessage(
      "No verified ISL animation resource is available. English-to-ISL is disabled until real sign-specific animation data is supplied.",
    )
  }

  return (
    <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
      <div className="lg:col-span-5 flex flex-col gap-4">
        <div className="rounded-xl bg-surface-container p-4 shadow-md flex flex-col gap-4 border border-outline-variant/20">
          <div className="flex items-center justify-between">
            <h2 className="font-headline-md text-headline-md text-on-surface flex items-center gap-2"><span className="material-symbols-outlined text-primary text-[18px]">notes</span> English Input</h2>
            <span className="font-technical-data text-xs uppercase tracking-widest text-on-surface-variant">SIGN ANIMATION UNAVAILABLE</span>
          </div>
          <textarea value={text} onChange={e => { setText(e.target.value); setMessage(null) }} placeholder="Enter English text…" rows={4} className="w-full rounded-lg bg-surface-container-lowest p-3 text-on-surface font-body-lg text-base focus:outline-none resize-none placeholder:text-on-surface-variant/40" />
          <div className="flex items-center justify-between">
            <button onClick={() => stt.state === "listening" ? stt.stop() : stt.start()} disabled={stt.state === "unsupported"} className="flex items-center gap-1.5 px-2.5 py-1 rounded-full font-technical-data text-xs bg-surface-container-high text-primary">
              <span className="material-symbols-outlined text-[14px]">{stt.state === "listening" ? "mic_off" : "mic"}</span>{stt.state === "listening" ? "Listening…" : "Voice input"}
            </button>
            <span className="font-technical-data text-xs text-on-surface-variant">{text.length} Chars · {tokens.length} Tokens</span>
          </div>
          {stt.error && <p className="font-technical-data text-xs text-error">{stt.error}</p>}
          <button onClick={translate} disabled={!text.trim()} className="w-full py-2.5 rounded-lg bg-primary text-on-primary font-headline-md text-sm disabled:opacity-40">
            Check supported phrase
          </button>
          {message && <p className="rounded bg-error-container/20 border border-error/30 px-3 py-2 font-technical-data text-xs text-error">{message}</p>}
        </div>
        <div className="rounded-xl bg-surface-container p-4 border border-outline-variant/20 flex flex-col gap-2">
          <span className="font-technical-data text-[10px] uppercase tracking-widest text-on-surface-variant">Potential future vocabulary</span>
          <p className="font-technical-data text-sm text-primary">{SUPPORTED_PHRASES.join(" · ")}</p>
          <p className="font-technical-data text-xs text-on-surface-variant">No phrase is shown as translated without verified sign-specific animation data.</p>
        </div>
        {transcript.length > 0 && (
          <div className="rounded-xl bg-surface-container-lowest p-3 border border-outline-variant/20">
            <span className="font-technical-data text-xs uppercase text-on-surface-variant">Recent</span>
            <div className="flex flex-col gap-1 mt-2">{transcript.slice(-3).reverse().map(entry => <div key={entry.id} className="text-sm text-on-surface line-clamp-1"><span className="text-primary font-technical-data text-xs mr-1">{entry.dir}</span>{entry.text}</div>)}</div>
          </div>
        )}
      </div>
      <div className="lg:col-span-7 rounded-xl bg-surface-container-lowest border border-outline-variant/20 min-h-96 flex items-center justify-center p-8 text-center">
        <div className="max-w-md text-on-surface-variant">
          <span className="material-symbols-outlined text-5xl opacity-20">accessibility_new</span>
          <p className="font-body-md mt-3">English-to-ISL output unavailable.</p>
          <p className="font-technical-data text-xs mt-2">Current GLB has generic hand bones but no verified ISL animation sequences. It is not used to fabricate signs.</p>
        </div>
      </div>
    </div>
  )
}
