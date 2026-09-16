import { useCallback } from "react"
import { stitchBenchmarksHtml } from "../lib/stitchResearchHtml"
import { HonestyBanner } from "../components/HonestyBanner"

const researchHtml = stitchBenchmarksHtml
  .replace(/<header[\s\S]*?<\/header>/, "")
  .replace(/<footer[\s\S]*?<\/footer>/g, "")
  // standardize global container to homepage reference: max-w-7xl + px-4 lg:px-8 (homepage uses px-margin/lg:px-margin-desktop = same)
  .replaceAll("max-w-6xl", "max-w-7xl")
  .replaceAll("px-margin lg:px-margin-desktop", "px-4 lg:px-8")
  .replaceAll("px-margin-desktop", "px-4 lg:px-8")
  .replaceAll("px-margin", "px-4")

export function ResearchPage({ navigate }: { navigate: (h: string) => void }) {
  const onClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const a = (e.target as HTMLElement).closest("a[data-path], a[href^='#']") as HTMLAnchorElement | null
    if (!a) return
    const dp = a.getAttribute("data-path"); const href = a.getAttribute("href") ?? ""
    if (dp === "translator" || href === "#" && dp === "translator") { e.preventDefault(); navigate("#/translate/sign"); return }
    if (href.startsWith("#") && href.length > 1) { e.preventDefault(); document.getElementById(href.slice(1))?.scrollIntoView({ behavior: "smooth" }); return }
    if (dp?.includes("research")) { e.preventDefault(); navigate("#/research"); return }
    if (dp?.includes("session")) { e.preventDefault(); navigate("#/translate/sign"); return }
    if (dp?.includes("architecture") || dp === "technology") { e.preventDefault(); navigate("#/technology"); return }
  }, [navigate])
  return (
    <div className="min-h-screen bg-surface flex flex-col">
      <div><HonestyBanner /><div onClick={onClick} dangerouslySetInnerHTML={{ __html: researchHtml }} /></div>
      <div className="w-full max-w-7xl mx-auto px-4 lg:px-8 py-8">
        <div className="rounded-xl bg-surface-container border border-outline-variant/30 p-4 mt-4">
          <h3 className="font-headline-md text-on-surface text-base mb-2">Authoritative numbers — run 20260916T144755Z (filtered 122,925 rows, 10k/1.5k/1.5k)</h3>
          <div className="grid md:grid-cols-2 gap-4 font-technical-data text-xs">
            <div className="space-y-1 text-on-surface-variant">
              <div><span className="text-on-surface">Trivial</span> WER 1.018 · BLEU 0.264 · ROUGE-L 0.025 — “Let me tell you about it.”</div>
              <div><span className="text-primary font-semibold">GRU</span> WER 1.015 · BLEU 0.311 · ROUGE-L 0.094 · 0.6% unique</div>
              <div>Transformer-full WER 1.019 · BLEU 0.158 · ROUGE 0.044 · 3.9% unique</div>
              <div>Transformer-no-face WER 1.034 · BLEU 0.207 · val loss 8.504 &lt; full 8.680</div>
            </div>
            <div className="space-y-1 text-on-surface-variant">
              <div>Text→Pose retrieval DTW 614 · generative 453 · MSE 0.177→0.122</div>
              <div>Motion energy GT 0.0417 · retrieval 0.034 · <span className="text-error">generative 0.00328 (7.9% GT) — collapsed</span></div>
              <div>Latency sign→text 183 ms · text→pose 13 ms</div>
              <div className="text-outline">See <span className="text-primary">research/outputs/phase0/RESULTS.md</span> for full findings.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
