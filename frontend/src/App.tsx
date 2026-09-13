import { useState } from "react"
import {
  ArrowRight,
  Camera,
  CheckCircle2,
  Keyboard,
  Languages,
  Moon,
  Send,
  Sun,
  TriangleAlert,
  Volume2,
} from "lucide-react"

import { ThemeProvider, useTheme } from "./components/theme-provider"

type ConversationMode = "sign" | "text"

function SignSageWorkspace() {
  const [mode, setMode] = useState<ConversationMode>("sign")
  const { theme, setTheme } = useTheme()
  const isSignMode = mode === "sign"

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#workspace" aria-label="SignSage home">
          <span className="brand-mark" aria-hidden="true">
            <Languages size={21} strokeWidth={2.4} />
          </span>
          <span>SignSage</span>
        </a>

        <div className="topbar-actions">
          <span className="prototype-badge">Research prototype</span>
          <button
            className="icon-button"
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? <Sun size={19} /> : <Moon size={19} />}
          </button>
        </div>
      </header>

      <section className="hero-copy" aria-labelledby="page-title">
        <p className="eyebrow">Indian Sign Language communication workspace</p>
        <h1 id="page-title">Choose how to communicate.</h1>
        <p>
          SignSage is a thesis prototype. Current models are under evaluation and can
          produce incorrect results. Do not use it for medical, legal, or emergency communication.
        </p>
      </section>

      <section id="workspace" className="workspace" aria-label="Translation workspace">
        <div className="mode-switcher" role="tablist" aria-label="Communication direction">
          <button
            className={isSignMode ? "mode-tab is-active" : "mode-tab"}
            type="button"
            role="tab"
            aria-selected={isSignMode}
            onClick={() => setMode("sign")}
          >
            <Camera size={18} />
            Sign to text
          </button>
          <button
            className={!isSignMode ? "mode-tab is-active" : "mode-tab"}
            type="button"
            role="tab"
            aria-selected={!isSignMode}
            onClick={() => setMode("text")}
          >
            <Keyboard size={18} />
            Text to sign
          </button>
        </div>

        {isSignMode ? <SignToTextPanel /> : <TextToSignPanel />}
      </section>

      <footer className="privacy-note">
        <CheckCircle2 size={18} aria-hidden="true" />
        Camera video stays on this device. Only extracted landmarks will be sent for inference.
      </footer>
    </main>
  )
}

function SignToTextPanel() {
  return (
    <div className="translation-grid">
      <section className="capture-card" aria-labelledby="camera-heading">
        <div className="card-heading">
          <div>
            <p className="section-kicker">Your turn</p>
            <h2 id="camera-heading">Sign to translate</h2>
          </div>
          <span className="status-pill">
            <span className="status-dot" />
            Camera off
          </span>
        </div>

        <div className="camera-placeholder">
          <div className="camera-icon" aria-hidden="true">
            <Camera size={34} />
          </div>
          <h3>Camera is ready when you are.</h3>
          <p>Frame your upper body, hands, and face. Pause after a sentence or use Send.</p>
          <button className="primary-button" type="button" disabled>
            <Camera size={18} />
            Start camera
          </button>
          <span className="button-note">Camera connection will be added next.</span>
        </div>
      </section>

      <section className="result-card" aria-labelledby="caption-heading" aria-live="polite">
        <div className="card-heading">
          <div>
            <p className="section-kicker">Live caption</p>
            <h2 id="caption-heading">English translation</h2>
          </div>
          <button className="icon-button subtle" type="button" disabled aria-label="Play translation aloud">
            <Volume2 size={19} />
          </button>
        </div>

        <div className="empty-result">
          <Languages size={30} aria-hidden="true" />
          <p>Your translated caption will appear here.</p>
        </div>

        <div className="model-warning" role="note">
          <TriangleAlert size={18} aria-hidden="true" />
          <span>Results must be checked by the people communicating.</span>
        </div>
      </section>
    </div>
  )
}

function TextToSignPanel() {
  return (
    <div className="translation-grid">
      <section className="capture-card" aria-labelledby="text-heading">
        <div className="card-heading">
          <div>
            <p className="section-kicker">Your turn</p>
            <h2 id="text-heading">Write a message</h2>
          </div>
        </div>

        <label className="text-field-label" htmlFor="message">
          English text
        </label>
        <textarea
          id="message"
          className="message-field"
          placeholder="Type what you want to communicate…"
          rows={7}
          disabled
        />
        <div className="text-actions">
          <span>Text-to-pose connection will be added next.</span>
          <button className="primary-button" type="button" disabled>
            Translate
            <ArrowRight size={18} />
          </button>
        </div>
      </section>

      <section className="result-card" aria-labelledby="animation-heading" aria-live="polite">
        <div className="card-heading">
          <div>
            <p className="section-kicker">Sign animation</p>
            <h2 id="animation-heading">ISL pose preview</h2>
          </div>
        </div>

        <div className="empty-result pose-placeholder">
          <Send size={30} aria-hidden="true" />
          <p>Generated pose animation will appear here.</p>
        </div>

        <div className="model-warning" role="note">
          <TriangleAlert size={18} aria-hidden="true" />
          <span>Current pose model is known to produce limited motion.</span>
        </div>
      </section>
    </div>
  )
}

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="signsage-theme">
      <SignSageWorkspace />
    </ThemeProvider>
  )
}

export default App
