import { useEffect, useState } from "react"

type Entry = { id: string; dir: string; text: string; latency?: number; ts: number }

export function ArchivePage({ navigate }: { navigate: (hash: string) => void }) {
  const [entries, setEntries] = useState<Entry[]>([])
  useEffect(() => {
    try { const raw = localStorage.getItem("signsage-transcript"); if (raw) setEntries(JSON.parse(raw)) } catch { /* ignore */ }
  }, [])
  const clear = () => { localStorage.removeItem("signsage-transcript"); setEntries([]) }

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      <main className="w-full max-w-7xl mx-auto px-4 lg:px-8 py-8 flex flex-col gap-8">
        <div className="flex items-center justify-between">
          <div><h1 className="font-headline-lg text-headline-lg text-on-surface">Session Archive</h1><p className="font-body-sm text-sm text-on-surface-variant">Local-only — stored in <span className="font-mono text-xs text-primary">localStorage</span> (ponytail: ephemeral, per-tab transcript if cleared; upgrade to IndexedDB/persistence if needed).</p></div>
          <button onClick={clear} disabled={entries.length === 0} className="px-3 py-1.5 rounded border border-outline-variant text-on-surface-variant text-label-sm hover:border-error hover:text-error disabled:opacity-40">Clear Archive</button>
        </div>
        {entries.length === 0 ? (
          <div className="rounded-xl bg-surface-container border border-outline-variant/30 p-8 text-center flex flex-col items-center gap-3">
            <span className="material-symbols-outlined text-5xl text-on-surface-variant/20">history</span>
            <p className="font-body-md text-body-md text-on-surface-variant">No archived sessions yet. Translate something and it will appear here.</p>
            <button onClick={() => navigate("/translate/sign")} className="px-4 py-2 rounded-lg bg-primary text-on-primary font-label-md text-sm">Start Translating</button>
          </div>
        ) : (
          <div className="grid gap-2">
            {entries.slice().reverse().map(e => (
              <div key={e.id} className="rounded-lg bg-surface-container border border-outline-variant/20 p-3 flex items-start justify-between gap-3">
                <div><span className="inline-flex px-1.5 py-0.5 rounded bg-surface-container-high text-primary font-technical-data text-xs mr-2">{e.dir}</span><span className="font-body-md text-sm text-on-surface">{e.text}</span></div>
                <span className="font-technical-data text-xs text-outline whitespace-nowrap">{e.latency ? `${e.latency.toFixed(0)} ms` : ""} · {new Date(e.ts).toLocaleTimeString()}</span>
              </div>
            ))}
          </div>
        )}
        <div className="rounded bg-surface-container-high border border-outline-variant/30 p-3 font-technical-data text-xs text-on-surface-variant flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-[16px]">lock</span> Archive is on-device only — never sent to server. Close tab → stays in localStorage until cleared.
        </div>
      </main>
    </div>
  )
}
