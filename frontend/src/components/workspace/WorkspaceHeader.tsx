import { ThemeToggle } from "../ThemeToggle"

export function WorkspaceHeader({ connectionState, navigate }: { connectionState: string; navigate: (h: string) => void }) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-surface-container-lowest/95 backdrop-blur-md border-b border-outline-variant/40">
      <div className="h-16 w-full max-w-7xl mx-auto px-4 lg:px-8 flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <a href="/" onClick={e => { e.preventDefault(); navigate("/") }} className="flex items-center gap-2">
            <img src="/signsage-logo.svg" alt="SignSage" className="h-8 w-8 rounded object-contain" />
            <span className="font-headline-md text-headline-md tracking-tight text-on-surface hidden sm:inline">SignSage</span>
          </a>
        </div>
        <nav className="hidden md:flex items-center gap-2">
          <button onClick={() => navigate("/")} className="px-3 py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface text-label-sm">Home</button>
          <button onClick={() => navigate("/research")} className="px-3 py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface text-label-sm">Research</button>
          <button onClick={() => navigate("/archive")} className="px-3 py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface text-label-sm">Archive</button>
          <button onClick={() => navigate("/privacy")} className="px-3 py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface text-label-sm">Privacy</button>
          <button onClick={() => navigate("/translate/sign")} className="px-3 py-2 rounded-lg bg-primary text-on-primary text-label-sm font-medium">Open Translator</button>
        </nav>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <span className={`h-2 w-2 rounded-full ${connectionState === "connected" ? "bg-primary animate-pulse" : "bg-error"}`} />
          <span className="text-technical-data text-xs text-on-surface-variant hidden sm:inline">{connectionState === "connected" ? "Live" : "Offline"}</span>
        </div>
      </div>
    </header>
  )
}
