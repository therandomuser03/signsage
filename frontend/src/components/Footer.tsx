export function Footer() {
  return (
    <footer className="w-full bg-surface-container-lowest border-t border-outline-variant/30 py-8">
      <div className="w-full max-w-7xl mx-auto px-4 lg:px-8 flex flex-col gap-6">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 pb-6 border-b border-outline-variant/20">
          <div className="md:col-span-5 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <img src="/signsage-logo.svg" alt="SignSage" className="h-7 w-7 rounded object-contain" />
              <span className="font-headline-md text-headline-md tracking-tight text-on-surface">SignSage</span>
              <span className="px-2 py-0.5 rounded border border-outline-variant/30 bg-surface-container text-on-surface-variant font-technical-data text-xs">v1.2-rc</span>
            </div>
            <p className="font-body-sm text-body-sm text-on-surface-variant max-w-md">
              Bidirectional Indian Sign Language ↔ English translation. Landmarks only, zero-cloud — see Privacy &amp; Research for honest limits.
            </p>
            <span className="inline-flex items-center gap-1.5 font-technical-data text-xs text-outline">
              <span className="w-1.5 h-1.5 rounded-full bg-primary" /> Zero-Cloud Processing Guarantee
            </span>
          </div>
          <div className="md:col-span-4 flex flex-col gap-1">
            <span className="font-label-sm text-label-sm uppercase tracking-wider text-outline">Research</span>
            <p className="font-technical-data text-xs text-on-surface-variant leading-relaxed bg-surface-container-low p-3 rounded border border-outline-variant/20">
              SignSage Consortium. (2024). High-Fidelity Bidirectional ISL Neural Synthesis. M.Tech Research Monograph. arXiv:2409.ISL01 — see Research &amp; Benchmarks for NO-GO status.
            </p>
          </div>
          <div className="md:col-span-3 flex flex-col gap-3">
            <span className="font-label-sm text-label-sm uppercase tracking-wider text-outline">Explore</span>
            <nav className="flex flex-col gap-1 font-body-sm text-sm">
              <a href="#/research" onClick={e=>{e.preventDefault();window.location.hash="#/research"}} className="text-on-surface-variant hover:text-primary">Research &amp; Benchmarks</a>
              <a href="#/technology" onClick={e=>{e.preventDefault();window.location.hash="#/technology"}} className="text-on-surface-variant hover:text-primary">Technology</a>
              <a href="#/privacy" onClick={e=>{e.preventDefault();window.location.hash="#/privacy"}} className="text-on-surface-variant hover:text-primary">Privacy</a>
              <a href="#/archive" onClick={e=>{e.preventDefault();window.location.hash="#/archive"}} className="text-on-surface-variant hover:text-primary">Archive</a>
            </nav>
          </div>
        </div>
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2 font-technical-data text-xs text-on-surface-variant">
          <span>© 2024 SignSage · CC BY-NC-SA 4.0 (iSign) · Not certified for medical/legal use</span>
          <span>ISL-ENG DUAL KERNEL · BUILT 2024.11.08</span>
        </div>
      </div>
    </footer>
  )
}
