import { ThemeToggle } from "./ThemeToggle"
import { useAuth } from "../contexts/AuthContext"

export function AppNavbar({ route, navigate }: { route: string; navigate: (h: string) => void }) {
  const { user, signOut } = useAuth()
  const is = (p: string) =>
    p === "#/" ? route === "#/" || route === "" : route.startsWith(p)
  const link = (p: string, label: string) => (
    <button
      onClick={() => navigate(p)}
      className={`px-3 py-1.5 rounded text-label-sm transition-colors ${
        is(p) ? "bg-surface-container text-on-surface" : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface"
      }`}
    >
      {label}
    </button>
  )
  return (
    <header className="sticky top-0 z-50 bg-surface-container-lowest/95 backdrop-blur-md border-b border-outline-variant/40">
      <div className="h-16 w-full max-w-7xl mx-auto px-4 lg:px-8 flex items-center justify-between gap-4">
        <button onClick={() => navigate("#/")} className="flex items-center gap-2 shrink-0">
          <img src="/signsage-logo.svg" alt="SignSage" className="h-8 w-8 rounded object-contain shrink-0" />
          <span className="font-headline-md text-headline-md tracking-tight text-on-surface hidden sm:inline">SignSage</span>
        </button>
        <nav className="hidden md:flex items-center gap-1">
          {link("#/", "Home")}
          {link("#/research", "Research")}
          {link("#/technology", "Technology")}
          {link("#/privacy", "Privacy")}
          {link("#/archive", "Archive")}
        </nav>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => navigate("#/translate/sign")}
            className="hidden sm:inline-flex px-3 py-2 rounded-lg bg-primary text-on-primary text-label-sm font-medium hover:bg-primary/90"
          >
            Open Translator
          </button>
          {user ? (
            <>
              <span className="hidden lg:inline-flex max-w-[160px] truncate font-technical-data text-xs text-on-surface-variant px-2">{user.email}</span>
              <button onClick={() => void signOut()} className="hidden sm:inline-flex px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface text-label-sm hover:bg-surface-container-high">Sign out</button>
            </>
          ) : (
            <button onClick={() => navigate(`#/auth?next=${encodeURIComponent("#/translate/sign")}`)} className="hidden sm:inline-flex px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface text-label-sm hover:bg-surface-container-high">Sign in</button>
          )}
          <ThemeToggle />
        </div>
      </div>
      {/* ponytail: single-row header stays 64px; mobile scroll nav inside same bar via css, no extra height */}
      <nav className="md:hidden flex items-center gap-1 px-4 pb-2 overflow-x-auto">
        {link("#/", "Home")}
        {link("#/research", "Research")}
        {link("#/technology", "Tech")}
        {link("#/privacy", "Privacy")}
        {link("#/archive", "Archive")}
        <button onClick={() => navigate("#/translate/sign")} className="px-3 py-1.5 rounded bg-primary text-on-primary text-label-sm shrink-0">Open Translator</button>
        {user ? (
          <button onClick={() => void signOut()} className="px-3 py-1.5 rounded bg-surface-container border border-outline-variant/40 text-on-surface text-label-sm shrink-0">Sign out</button>
        ) : (
          <button onClick={() => navigate(`#/auth?next=${encodeURIComponent("#/translate/sign")}`)} className="px-3 py-1.5 rounded bg-surface-container border border-outline-variant/40 text-on-surface text-label-sm shrink-0">Sign in</button>
        )}
      </nav>
    </header>
  )
}
