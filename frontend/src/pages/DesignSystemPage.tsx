export function DesignSystemPage(_: { navigate: (hash: string) => void }) {
  const colors = [
    ["Surface", "#0d141d", "bg-surface"], ["Surface-low", "#151c25", "bg-surface-container-low"],
    ["Surface", "#192029", "bg-surface-container"], ["Surface-high", "#242a34", "bg-surface-container-high"],
    ["Surface-highest", "#2e353f", "bg-surface-container-highest"], ["Outline", "#86948e", "border-outline"],
    ["Primary", "#65dabc", "bg-primary"], ["On-primary", "#00382d", "bg-on-primary"],
    ["Secondary", "#cabeff", "bg-secondary"], ["Error", "#ffb4ab", "bg-error"],
  ]
  return (
    <div className="min-h-screen bg-surface flex flex-col">
      <main className="max-w-7xl mx-auto w-full px-4 lg:px-8 py-8 flex flex-col gap-8">
        <div className="flex flex-col gap-3">
          <span className="font-technical-data text-xs uppercase tracking-widest text-primary">Precision Linguistic Instrument • Design System</span>
          <h1 className="font-headline-xl text-headline-xl text-on-surface">SignSage — Research Pipeline & Benchmarks</h1>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-3xl">Dark instrument palette for extended optical monitoring. All pages share these tokens — built from <span className="text-primary font-mono text-sm">DESIGN.md</span> + Stitch <span className="font-mono text-xs">8207772843765018051</span>.</p>
        </div>

        <section className="grid md:grid-cols-2 gap-4">
          <div className="rounded-xl bg-surface-container border border-outline-variant/30 p-4">
            <h2 className="font-headline-md text-headline-md text-on-surface mb-3">Colors</h2>
            <div className="grid grid-cols-2 gap-2">
              {colors.map(([name, hex]) => (
                <div key={name} className="flex items-center gap-2 p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
                  <span className="h-6 w-6 rounded border border-outline-variant/30" style={{ background: hex }} />
                  <span className="font-technical-data text-xs text-on-surface">{name}</span>
                  <span className="font-mono text-[11px] text-on-surface-variant ml-auto">{hex}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-xl bg-surface-container border border-outline-variant/30 p-4 flex flex-col gap-4">
            <h2 className="font-headline-md text-headline-md text-on-surface">Typography</h2>
            <div className="flex flex-col gap-2">
              <div><div className="font-technical-data text-[11px] uppercase text-on-surface-variant">headline-xl — Space Grotesk 3rem/600</div><div className="font-headline-xl text-headline-xl text-on-surface">Real-Time ISL ↔ English</div></div>
              <div><div className="font-technical-data text-[11px] uppercase text-on-surface-variant">body-lg — Inter 1.125rem</div><div className="font-body-lg text-body-lg text-on-surface-variant">Communication without the barrier.</div></div>
              <div><div className="font-technical-data text-[11px] uppercase text-on-surface-variant">technical-data — JetBrains Mono</div><div className="font-technical-data text-technical-data text-primary">37.8 ms · 60 FPS · 543 Holistic Coordinates</div></div>
            </div>
          </div>
        </section>

        <section className="rounded-xl bg-surface-container border border-outline-variant/30 p-4">
          <h2 className="font-headline-md text-headline-md text-on-surface mb-3">Elevation & Components</h2>
          <div className="grid md:grid-cols-3 gap-3">
            <div className="rounded-lg bg-surface-container-lowest border border-outline-variant/30 p-3"><div className="font-label-md text-label-md text-on-surface">Level 0 — Canvas</div><div className="font-technical-data text-xs text-on-surface-variant">#090E13 · video feed</div></div>
            <div className="rounded-lg bg-surface-container border border-outline-variant/30 p-3"><div className="font-label-md text-label-md text-on-surface">Level 1 — Module</div><div className="font-technical-data text-xs text-on-surface-variant">#101720 · 1px #253342</div></div>
            <div className="rounded-lg bg-surface-container-low border border-outline-variant/30 p-3 shadow-[0_4px_16px_rgba(0,0,0,0.6)]"><div className="font-label-md text-label-md text-on-surface">Level 2 — Floating</div><div className="font-technical-data text-xs text-on-surface-variant">#151E28 · popover</div></div>
          </div>
          <div className="flex flex-wrap gap-2 mt-4">
            <button className="px-4 py-2 rounded-lg bg-primary text-on-primary font-label-md text-sm">Primary — Start Translation</button>
            <button className="px-4 py-2 rounded-lg bg-surface-container-low border border-outline-variant text-on-surface-variant font-label-sm text-sm">Secondary — Re-calibrate</button>
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-surface-container-low border border-outline-variant/40 font-technical-data text-xs text-primary"><span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />Latency 37.8 ms</span>
          </div>
        </section>

        <section className="rounded-xl bg-surface-container-lowest border border-outline-variant/30 p-4 flex items-center gap-4">
          <img src="/signsage-logo.svg" alt="logo" className="h-16 w-16 rounded-xl border border-outline-variant/30" />
          <div><div className="font-headline-md text-headline-md text-on-surface">Precision Instrument Logo</div><div className="font-body-sm text-body-sm text-on-surface-variant">120×120 · #101720 · #16A085 / #7C5CFC · Stitch e6bd6bfc</div></div>
          <a href="/signsage-logo.svg" download className="ml-auto px-3 py-1.5 rounded bg-surface-container border border-outline-variant text-primary font-technical-data text-xs">Download SVG</a>
        </section>
      </main>
    </div>
  )
}
