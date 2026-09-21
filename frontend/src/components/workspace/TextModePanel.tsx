import { useEffect, useState } from "react"
import { PoseCanvas } from "../pose-canvas"
import { useSpeechRecognition } from "../../hooks/useSpeechRecognition"

export function TextModePanel({ pose, error, send, connectionState, transcript }: {
  pose: { frames: number[][]; fps: number; gloss: string; latency: number; isFallback: boolean } | null
  error: string | null; send: (msg: Record<string, unknown>) => boolean
  connectionState: string; transcript: { id: string; dir: string; text: string }[]
}) {
  const [text, setText] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const stt = useSpeechRecognition({ lang: "en-IN", onResult: (t: string) => setText(t) })
  const translate = () => { const sent = send({ type: "text.input", text: text.trim() }); if (sent) setIsSubmitting(true) }
  useEffect(() => { if (pose || error) setIsSubmitting(false) }, [pose, error])
  const tokens = text.trim().split(/\s+/).filter(Boolean)
  const glossPreview = text.trim() ? text.trim().toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).slice(0, 6).join(" ") : "—"

  return (
    <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
      <div className="lg:col-span-5 flex flex-col gap-4">
        <div className="rounded-xl bg-surface-container p-4 shadow-md flex flex-col gap-4 border border-outline-variant/20">
          <div className="flex items-center justify-between">
            <h2 className="font-headline-md text-headline-md text-on-surface flex items-center gap-2"><span className="material-symbols-outlined text-primary text-[18px]">notes</span> Source Text Input</h2>
            <span className="font-technical-data text-xs uppercase tracking-widest text-on-surface-variant">ISL-SYNTH-KERNEL</span>
          </div>
          <div className="relative rounded-lg bg-surface-container-lowest p-3 border border-outline-variant/30 focus-within:border-primary/50">
            <textarea value={text} onChange={e => setText(e.target.value)} placeholder="Enter natural English sentence for ISL translation…" rows={4} className="w-full bg-transparent text-on-surface font-body-lg text-base focus:outline-none resize-none placeholder:text-on-surface-variant/40" />
            <div className="flex items-center justify-between pt-2">
              <button onClick={() => stt.state === "listening" ? stt.stop() : stt.start()} disabled={stt.state === "unsupported"} className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full font-technical-data text-xs transition-colors ${stt.state === "listening" ? "bg-error text-on-error" : "bg-surface-container-high text-primary hover:bg-surface-container-highest"}`}>
                <span className="material-symbols-outlined text-[14px]">{stt.state === "listening" ? "mic_off" : "mic"}</span> {stt.state === "listening" ? "Listening…" : stt.state === "unsupported" ? "Voice unsupported" : "Mic Armed"}
              </button>
              <span className="font-technical-data text-xs text-on-surface-variant">{text.length} Chars · {tokens.length} Tokens</span>
            </div>
            {stt.error && <p className="font-technical-data text-xs text-error mt-1">{stt.error}</p>}
          </div>
          <div className="flex flex-wrap gap-1 items-center">
            <span className="font-label-sm text-xs text-on-surface-variant">Try a supported demo phrase:</span>
            {['"Hello"', '"Thank you"', '"How are you"'].map(p => (
              <button key={p} onClick={() => setText(p.replace(/"/g, ""))} className="px-2 py-0.5 rounded-full bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-technical-data text-xs">{p}</button>
            ))}
          </div>
          <button onClick={translate} disabled={!text.trim() || connectionState !== "connected" || isSubmitting} className="w-full py-2.5 rounded-lg bg-primary hover:bg-[#1ABC9C] text-on-primary font-headline-md text-sm flex items-center justify-center gap-2 disabled:opacity-40 shadow-md">
            <span className="material-symbols-outlined text-[18px]">smart_toy</span> {isSubmitting ? "Synthesizing…" : "Synthesize ISL Pose Sequence"}
          </button>
          {error && <div className="rounded bg-error-container/20 border border-error/30 px-3 py-2 font-technical-data text-xs text-error wrap-break-word">{error}</div>}
        </div>
        <div className="rounded-xl bg-surface-container p-4 border border-outline-variant/20 flex flex-col gap-2">
          <span className="font-technical-data text-[10px] uppercase tracking-widest text-on-surface-variant">Linguistic Decomposition — Gloss</span>
          <p className="font-technical-data text-sm text-primary uppercase">{glossPreview}</p>
          <div className="flex flex-wrap gap-1">{tokens.slice(0, 6).map((t, i) => (<span key={i} className="px-2 py-1 rounded bg-surface-container-high text-primary font-technical-data text-xs uppercase">{t}</span>))}</div>
          {pose && <span className="font-technical-data text-xs text-on-surface-variant">Gloss: {pose.gloss} · {pose.latency.toFixed(0)} ms · {pose.fps} FPS · {pose.isFallback ? "Recorded demo clip (hands)" : "Generated pose (research model, near-static)"}</span>}
        </div>
        {transcript.length > 0 && (
          <div className="rounded-xl bg-surface-container-lowest p-3 border border-outline-variant/20">
            <div className="flex items-center justify-between mb-2"><span className="font-technical-data text-xs uppercase text-on-surface-variant">Recent</span><span className="text-xs text-outline-variant font-technical-data">#{transcript.length}</span></div>
            <div className="flex flex-col gap-1">{transcript.slice(-3).reverse().map((e: { id: string; dir: string; text: string }) => (<div key={e.id} className="text-sm text-on-surface font-body-sm line-clamp-1"><span className="text-primary font-technical-data text-xs mr-1">{e.dir}</span>{e.text}</div>))}</div>
          </div>
        )}
      </div>
      <div className="lg:col-span-7 flex flex-col gap-3">
        <div className="rounded-xl bg-surface-container-lowest border border-outline-variant/20 overflow-hidden shadow-xl">
          <div className="px-3 py-2 bg-surface-container-low border-b border-outline-variant/20 flex items-center justify-between">
            <span className="font-technical-data text-xs text-on-surface flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-primary animate-pulse" /> POSE VIEWPORT{pose ? ` · ${pose.fps} FPS` : ""}</span>
            <span className="font-technical-data text-[11px] text-outline">{pose ? `${pose.frames.length} frames` : "Idle"}</span>
          </div>
          <div className="relative w-full aspect-16/10 bg-surface-container-lowest flex items-center justify-center overflow-hidden">
            {pose ? <PoseCanvas frames={pose.frames} fps={pose.fps} handsOnly={pose.isFallback} /> : <div className="text-center text-on-surface-variant flex flex-col items-center"><span className="material-symbols-outlined text-5xl opacity-20 mb-2">smart_toy</span><p className="font-body-sm text-sm">Pose synthesis will appear here.</p></div>}
            {pose && (
              <div className="absolute top-3 left-3 px-2 py-1 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-on-surface-variant">
                {pose.isFallback ? "Recorded demo clip · hands only" : "Generated by research model"}
              </div>
            )}
            <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between px-2 py-1 rounded bg-surface-container-lowest/80 backdrop-blur border border-outline-variant/40 font-technical-data text-[11px] text-outline">
              <span>Landmarks computed on the SignSage server</span>
              <span className="text-primary">GLB avatar + landmarks</span>
            </div>
          </div>
        </div>
        {pose?.isFallback ? (
          <div className="rounded bg-primary/10 border border-primary/30 px-3 py-2 flex gap-2 items-center">
            <span className="material-symbols-outlined text-primary text-[16px]">check_circle</span>
            <p className="font-technical-data text-xs text-on-surface-variant">Recorded demo phrase. GLB avatar hand bones are retargeted from captured hand landmarks.</p>
          </div>
        ) : (
          <div className="rounded bg-secondary/20 border border-secondary/30 px-3 py-2 flex gap-2 items-center">
            <span className="material-symbols-outlined text-secondary text-[16px]">info</span>
            <p className="font-technical-data text-xs text-on-surface-variant">This phrase isn't in the bounded demo set, so it fell back to the research pose-generation model, which is near-static (~7.9% of ground-truth motion) — see Research &amp; Benchmarks. Try: Hello / Thank you / Yes / No / Help / Please / Sorry / Good morning / How are you / I need help.</p>
          </div>
        )}
      </div>
    </div>
  )
}
