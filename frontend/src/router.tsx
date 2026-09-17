import { BrowserRouter, Routes, Route, NavLink, Link } from "react-router-dom"
import { useEffect } from "react"
import { LandingPage } from "./pages/LandingPage"
import { ResearchPage } from "./pages/ResearchPage"
import { TechnologyPage } from "./pages/TechnologyPage"
import { DesignSystemPage } from "./pages/DesignSystemPage"
import { PrivacyPage } from "./pages/PrivacyPage"
import { ArchivePage } from "./pages/ArchivePage"
import { AuthPage } from "./pages/AuthPage"
import { WorkspacePage } from "./pages/WorkspacePage"
import { NotFoundPage } from "./pages/NotFoundPage"
import { useNavigate } from "react-router-dom"

export default function Router() {
  const navigate = useNavigate()
  useEffect(() => {}, [])

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage navigate={navigate} />} />
        <Route path="/research" element={<ResearchPage navigate={navigate} />} />
        <Route path="/technology" element={<TechnologyPage navigate={navigate} />} />
        <Route path="/privacy" element={<PrivacyPage navigate={navigate} />} />
        <Route path="/archive" element={<ArchivePage navigate={navigate} />} />
        <Route path="/translate/sign" element={<WorkspacePage initialMode="sign" navigate={navigate} />} />
        <Route path="/translate/text" element={<WorkspacePage initialMode="text" navigate={navigate} />} />
        <Route path="/auth" element={<AuthPage navigate={navigate} />} />
        <Route element={<NotFoundPage navigate={navigate} />}>
          <Route path="*" />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
