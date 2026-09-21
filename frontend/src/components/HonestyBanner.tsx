export function HonestyBanner() {
  return (
    <div className="w-full bg-error-container/15 border-b border-error/30 px-4 lg:px-8 py-2">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center gap-2 font-technical-data text-xs">
        <span className="inline-flex items-center gap-1.5 text-error font-semibold"><span className="w-2 h-2 rounded-full bg-error animate-pulse" /> NO-GO · Run 20260916T144755Z</span>
        <span className="text-on-surface-variant">GRU BLEU 0.31 &gt; Trivial 0.26 &gt; Transformer 0.16 · Transformer unique 3.9% · Pose motion 7.9% GT (static-pose artifact)</span>
        <span className="text-outline ml-auto">See research/outputs/phase0/RESULTS.md in this project</span>
      </div>
    </div>
  )
}
