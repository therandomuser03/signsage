import { useCallback, useEffect, useState } from "react"
import { ThemeProvider } from "./components/theme-provider"
import { AuthProvider, useAuth } from "./contexts/AuthContext"
import { AppNavbar } from "./components/AppNavbar"
import { Footer } from "./components/Footer"
import { LandingPage } from "./pages/LandingPage"
import { WorkspacePage } from "./pages/WorkspacePage"
import { ResearchPage } from "./pages/ResearchPage"
import { TechnologyPage } from "./pages/TechnologyPage"
import { DesignSystemPage } from "./pages/DesignSystemPage"
import { PrivacyPage } from "./pages/PrivacyPage"
import { ArchivePage } from "./pages/ArchivePage"
import { AuthPage } from "./pages/AuthPage"
import { NotFoundPage } from "./pages/NotFoundPage"

function useHashRoute(): [string, (hash: string) => void] {
  const getRoute = () => window.location.pathname + window.location.search || "/"
  const [route, setRoute] = useState(() => {
    // migrate legacy #/ URLs → clean URLs once
    const h = window.location.hash.replace(/^#/, "")
    if (h.startsWith("/")) {
      history.replaceState(null, "", h)
      return h
    }
    return getRoute()
  })
  useEffect(() => {
    const onPop = () => setRoute(getRoute())
    const onHash = () => {
      const h = window.location.hash.replace(/^#/, "")
      if (h.startsWith("/")) {
        history.replaceState(null, "", h)
        setRoute(h)
      }
    }
    window.addEventListener("popstate", onPop)
    window.addEventListener("hashchange", onHash)
    return () => {
      window.removeEventListener("popstate", onPop)
      window.removeEventListener("hashchange", onHash)
    }
  }, [])
  const navigate = useCallback((hash: string) => {
    const clean = hash.replace(/^#/, "") || "/"
    if (clean !== getRoute()) {
      history.pushState(null, "", clean)
      setRoute(clean)
    }
    window.scrollTo(0, 0)
  }, [])
  return [route, navigate]
}

const AUTH_DISABLED = true // ponytail: testing/build — flip to false to re-enable protected translator

function AppShell() {
  const [route, navigate] = useHashRoute()
  const { user, loading } = useAuth()

  // Protected translator: unauth → auth with next (disabled for testing)
  useEffect(() => {
    if (AUTH_DISABLED || loading) return
    if (route.startsWith("/translate") && !user) {
      const next = encodeURIComponent(route)
      if (!route.startsWith("/auth")) navigate("/auth?next=" + next)
    }
  }, [route, user, loading, navigate])

  return (
    <>
      <AppNavbar route={route} navigate={navigate} />
      <div className="flex flex-col min-h-[calc(100vh-4rem)]">
        {route.startsWith("/auth") ? (
          <AuthPage navigate={navigate} />
        ) : route.startsWith("/translate") ? (
          AUTH_DISABLED ? (
            <WorkspacePage initialMode={route === "/translate/text" ? "text" : "sign"} navigate={navigate} />
          ) : loading ? (
            <div className="flex-1 flex items-center justify-center font-technical-data text-sm text-on-surface-variant">Checking authentication…</div>
          ) : user ? (
            <WorkspacePage initialMode={route === "/translate/text" ? "text" : "sign"} navigate={navigate} />
          ) : (
            <div className="flex-1 flex items-center justify-center font-technical-data text-sm text-on-surface-variant">Redirecting to sign in…</div>
          )
        ) : route.startsWith("/research") ? (
          <ResearchPage navigate={navigate} />
        ) : route.startsWith("/technology") ? (
          <TechnologyPage navigate={navigate} />
        ) : route.startsWith("/design") ? (
          <DesignSystemPage navigate={navigate} />
        ) : route.startsWith("/privacy") ? (
          <PrivacyPage navigate={navigate} />
        ) : route.startsWith("/archive") ? (
          <ArchivePage navigate={navigate} />
        ) : route === "/" || route === "" ? (
          <LandingPage navigate={navigate} />
        ) : (
          <NotFoundPage navigate={navigate} />
        )}
        <Footer />
      </div>
    </>
  )
}

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="signsage-theme">
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </ThemeProvider>
  )
}

export default App
