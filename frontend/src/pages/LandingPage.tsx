import { useCallback } from "react"
import { stitchLandingHtml } from "../lib/stitchLandingHtml"

const landingHtml = stitchLandingHtml
  .replace(/<header[\s\S]*?<\/header>/, "")
  .replace(/<footer[\s\S]*?<\/footer>/, "")
  .replace(/<script[\s\S]*?<\/script>/g, "")
  .replace(/M\.Tech Research Initiative/g, "SignSage Platform")
  .replace(/M\.Tech/g, "SignSage")
  // keep global container in sync with AppNavbar/Footer: homepage is the reference (max-w-7xl + px-4 lg:px-8)
  .replaceAll("max-w-6xl", "max-w-7xl")
  .replaceAll("px-margin lg:px-margin-desktop", "px-4 lg:px-8")
  .replaceAll("px-margin-desktop", "px-4 lg:px-8")
  .replaceAll("px-margin", "px-4")

export function LandingPage({ navigate }: { navigate: (hash: string) => void }) {
  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const a = (e.target as HTMLElement).closest("a[data-path], a[href^='#']") as HTMLAnchorElement | null
    if (!a) return
    const dp = a.getAttribute("data-path")
    const href = a.getAttribute("href") ?? ""
    if (dp === "translator") { e.preventDefault(); navigate("#/translate/sign"); return }
    if (dp === "landing") { e.preventDefault(); navigate("#/"); window.scrollTo({ top: 0, behavior: "smooth" }); return }
    if (href.startsWith("#") && href.length > 1) { e.preventDefault(); document.getElementById(href.slice(1))?.scrollIntoView({ behavior: "smooth" }); return }
    if (dp === "research-benchmarks" || dp === "research-and-benchmarks") { e.preventDefault(); navigate("#/research"); return }
    if (dp === "technology" || dp?.includes("architecture")) { e.preventDefault(); navigate("#/technology"); return }
    if (dp === "privacy") { e.preventDefault(); navigate("#/privacy"); return }
    if (dp?.includes("archive") || dp === "session-archive") { e.preventDefault(); navigate("#/archive"); return }
  }, [navigate])

  return (
    <div onClick={handleClick} className="min-h-screen bg-surface flex flex-col">
      <div dangerouslySetInnerHTML={{ __html: landingHtml }} />
    </div>
  )
}
