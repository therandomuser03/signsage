import { useCallback } from "react"
import { stitchMonographHtml } from "../lib/stitchResearchHtml"
import { HonestyBanner } from "../components/HonestyBanner"

const techHtml = stitchMonographHtml
  .replace(/<header[\s\S]*?<\/header>/, "")
  .replace(/<footer[\s\S]*?<\/footer>/g, "")
  .replaceAll("max-w-6xl", "max-w-7xl")
  .replaceAll("px-margin lg:px-margin-desktop", "px-4 lg:px-8")
  .replaceAll("px-margin-desktop", "px-4 lg:px-8")
  .replaceAll("px-margin", "px-4")

export function TechnologyPage({ navigate }: { navigate: (h: string) => void }) {
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
      <div><HonestyBanner /><div onClick={onClick} dangerouslySetInnerHTML={{ __html: techHtml }} /></div>
    </div>
  )
}
