import { useEffect, useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import { BlurFade } from "../components/ui/blur-fade"
import { TextAnimate } from "../components/ui/text-animate"
import { FlickeringGrid } from "../components/ui/flickering-grid"
import { useAuth } from "../contexts/AuthContext"

function getNextHash(): string {
  const hash = window.location.hash || ""
  const qIndex = hash.indexOf("?")
  if (qIndex === -1) return "#/translate/sign"
  const qs = new URLSearchParams(hash.slice(qIndex + 1))
  const next = qs.get("next")
  if (!next) return "#/translate/sign"
  try { return decodeURIComponent(next) } catch { return next }
}

export function AuthPage({ navigate }: { navigate: (h: string) => void }) {
  const { user, loading, signIn, signUp, signOut } = useAuth()
  const [mode, setMode] = useState<"signin" | "signup">("signin")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [info, setInfo] = useState<string | null>(null)

  const nextHash = getNextHash()

  useEffect(() => {
    if (!loading && user) navigate(nextHash)
  }, [user, loading, navigate, nextHash])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null); setInfo(null)
    if (!email.trim() || !password) { setError("Email and password are required."); return }
    if (password.length < 6) { setError("Password must be at least 6 characters."); return }
    setSubmitting(true)
    const res = mode === "signin" ? await signIn(email.trim(), password) : await signUp(email.trim(), password)
    setSubmitting(false)
    if (res.error) { setError(res.error); return }
    if (mode === "signup") {
      setInfo("Account created — check email to confirm, then sign in.")
      setMode("signin")
    } else {
      navigate(nextHash)
    }
  }

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-4rem)] w-full flex items-center justify-center bg-surface">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center gap-3">
          <span className="h-8 w-8 rounded-full border-2 border-primary/20 border-t-primary animate-spin" />
          <span className="font-technical-data text-sm text-on-surface-variant">Checking session…</span>
        </motion.div>
      </div>
    )
  }

  if (user) {
    return (
      <div className="min-h-[calc(100vh-4rem)] w-full max-w-7xl mx-auto px-4 lg:px-8 flex items-center justify-center py-12">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="flex flex-col items-center gap-5 rounded-2xl bg-surface-container border border-outline-variant/30 p-8 shadow-xl">
          <img src="/signsage-logo.svg" alt="SignSage" className="h-12 w-12 rounded" />
          <div className="text-center">
            <p className="font-headline-md text-on-surface">Signed in as {user.email}</p>
            <p className="font-body-sm text-sm text-on-surface-variant">You can now open the translator.</p>
          </div>
          <div className="flex gap-3">
            <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={() => navigate(nextHash)} className="px-6 py-2.5 rounded-lg bg-primary text-on-primary text-sm font-medium">Continue to Translator</motion.button>
            <button onClick={() => void signOut()} className="px-6 py-2.5 rounded-lg bg-surface-container-high border border-outline-variant text-on-surface text-sm">Sign out</button>
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="relative min-h-[calc(100vh-4rem)] w-full overflow-hidden bg-surface flex items-center justify-center">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8 }} className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-secondary/5" />
        <FlickeringGrid className="absolute inset-0 opacity-30" squareSize={4} gridGap={6} color="#16A085" maxOpacity={0.08} flickerChance={0.1} />
      </motion.div>

      <div className="relative w-full max-w-7xl mx-auto px-4 lg:px-8 py-8 flex justify-center">
        <div className="w-full grid lg:grid-cols-[1.05fr_0.95fr] gap-8 lg:gap-12 items-center">
          {/* Left — branding / info */}
          <div className="hidden lg:flex flex-col gap-6 pr-6">
            <BlurFade delay={0.05}>
              <div className="flex items-center gap-3">
                <img src="/signsage-logo.svg" alt="SignSage" className="h-10 w-10 rounded-lg shadow-sm" />
                <span className="font-headline-lg text-headline-lg tracking-tight text-on-surface">SignSage</span>
                <span className="px-2 py-0.5 rounded border border-outline-variant/30 bg-surface-container text-on-surface-variant font-technical-data text-xs">v1.2-rc</span>
              </div>
            </BlurFade>

            <BlurFade delay={0.12}>
              <TextAnimate as="h1" by="word" animation="blurInUp" className="font-headline-xl text-headline-xl tracking-tight text-on-surface leading-[1.05] text-4xl lg:text-[42px]" once>
                One account. Every translation, linked.
              </TextAnimate>
            </BlurFade>

            <BlurFade delay={0.18}>
              <p className="font-body-lg text-body-lg text-on-surface-variant leading-relaxed max-w-xl">
                Sign in to use the protected translator. ISL ↔ English runs on 543 holistic landmarks — zero-cloud, landmarks only. Your session keeps translations associated with you.
              </p>
            </BlurFade>

            <BlurFade delay={0.24} className="flex flex-col gap-3 pt-2">
              <div className="flex gap-3">
                {[
                  { icon: "lock", title: "Zero-Cloud", desc: "Video never leaves your device" },
                  { icon: "badge", title: "Linked Sessions", desc: "History tied to your account" },
                  { icon: "verified_user", title: "Research Honest", desc: "NO-GO limits disclosed" },
                ].map(f => (
                  <div key={f.title} className="flex-1 rounded-xl bg-surface-container border border-outline-variant/30 p-3 flex flex-col gap-1">
                    <span className="material-symbols-outlined text-primary text-[18px]">{f.icon}</span>
                    <span className="font-label-sm text-sm text-on-surface">{f.title}</span>
                    <span className="font-technical-data text-xs text-on-surface-variant leading-tight">{f.desc}</span>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2 font-technical-data text-xs text-outline">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" /> 543 coordinates · 60 FPS · WebGPU
                <span className="mx-1">·</span>
                <a href="#/privacy" onClick={e=>{e.preventDefault();navigate("#/privacy")}} className="text-primary hover:underline">Privacy</a>
                <span>·</span>
                <a href="#/research" onClick={e=>{e.preventDefault();navigate("#/research")}} className="text-primary hover:underline">Research</a>
              </div>
            </BlurFade>
          </div>

          {/* Right — auth card */}
          <BlurFade delay={0.1} className="w-full flex justify-center lg:justify-end">
            <motion.div
              initial={{ opacity: 0, y: 16, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              className="w-full max-w-[440px] rounded-2xl bg-surface-container-lowest/90 backdrop-blur border border-outline-variant/30 shadow-2xl overflow-hidden"
            >
              <div className="p-6 sm:p-8 flex flex-col gap-5">
                {/* mobile logo */}
                <div className="lg:hidden flex items-center gap-2">
                  <img src="/signsage-logo.svg" alt="SignSage" className="h-8 w-8 rounded" />
                  <span className="font-headline-md text-on-surface">SignSage</span>
                </div>

                <div className="flex flex-col gap-1.5 text-center lg:text-left">
                  <h2 className="font-headline-lg text-xl text-on-surface tracking-tight">{mode === "signin" ? "Welcome back" : "Create your account"}</h2>
                  <p className="font-body-sm text-sm text-on-surface-variant">
                    {mode === "signin" ? "Sign in to open the translator." : "Sign up to get your linked session."}
                  </p>
                </div>

                {/* Tabs with sliding indicator */}
                <div className="relative flex p-1 rounded-xl bg-surface-container border border-outline-variant/20">
                  <motion.div
                    layoutId="auth-tab-indicator"
                    className="absolute top-1 bottom-1 w-[calc(50%-4px)] rounded-lg bg-surface-container-high shadow-sm border border-outline-variant/20"
                    animate={{ x: mode === "signin" ? 0 : "100%" }}
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    style={{ left: 4 }}
                  />
                  <button onClick={() => { setMode("signin"); setError(null); setInfo(null) }} className={`relative z-10 flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${mode==="signin" ? "text-on-surface" : "text-on-surface-variant hover:text-on-surface"}`}>Sign in</button>
                  <button onClick={() => { setMode("signup"); setError(null); setInfo(null) }} className={`relative z-10 flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${mode==="signup" ? "text-on-surface" : "text-on-surface-variant hover:text-on-surface"}`}>Sign up</button>
                </div>

                <motion.form
                  key={mode}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  onSubmit={handleSubmit}
                  className="flex flex-col gap-3.5"
                >
                  <motion.label initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.05 }} className="flex flex-col gap-1.5">
                    <span className="font-label-sm text-xs uppercase tracking-wider text-on-surface-variant">Email</span>
                    <input
                      value={email} onChange={e=>setEmail(e.target.value)}
                      type="email" autoComplete="email" placeholder="you@example.com"
                      className="h-11 rounded-xl bg-surface-container border border-outline-variant/40 px-3.5 text-sm text-on-surface placeholder:text-on-surface-variant/50 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
                    />
                  </motion.label>

                  <motion.label initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="flex flex-col gap-1.5">
                    <span className="font-label-sm text-xs uppercase tracking-wider text-on-surface-variant">Password</span>
                    <input
                      value={password} onChange={e=>setPassword(e.target.value)}
                      type="password" autoComplete={mode==="signin"?"current-password":"new-password"} placeholder="••••••••"
                      className="h-11 rounded-xl bg-surface-container border border-outline-variant/40 px-3.5 text-sm text-on-surface placeholder:text-on-surface-variant/50 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
                    />
                    <span className="font-technical-data text-xs text-outline">At least 6 characters</span>
                  </motion.label>

                  <AnimatePresence mode="wait">
                    {error && (
                      <motion.div
                        key="error"
                        initial={{ opacity: 0, y: -6, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.2 }}
                        className="rounded-xl bg-error-container/15 border border-error/30 px-3.5 py-2.5 flex gap-2.5 items-start"
                      >
                        <span className="material-symbols-outlined text-error text-[18px] mt-0.5">error</span>
                        <span className="font-technical-data text-xs leading-relaxed text-error">{error}</span>
                      </motion.div>
                    )}
                    {info && !error && (
                      <motion.div
                        key="info"
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="rounded-xl bg-primary/10 border border-primary/20 px-3.5 py-2.5 font-technical-data text-xs text-primary leading-relaxed"
                      >
                        {info}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <motion.button
                    type="submit"
                    disabled={submitting}
                    whileHover={{ scale: submitting ? 1 : 1.01 }}
                    whileTap={{ scale: submitting ? 1 : 0.99 }}
                    className="mt-1 h-11 rounded-xl bg-primary text-on-primary font-medium text-sm hover:bg-primary/90 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-md shadow-primary/20 transition-colors"
                  >
                    <AnimatePresence mode="wait">
                      {submitting ? (
                        <motion.span key="spin" initial={{ opacity: 0, rotate: -90 }} animate={{ opacity: 1, rotate: 0 }} exit={{ opacity: 0 }} className="h-4 w-4 rounded-full border-2 border-on-primary/30 border-t-on-primary animate-spin" />
                      ) : (
                        <motion.span key="icon" initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} className="material-symbols-outlined text-[18px]">{mode==="signin" ? "login" : "person_add"}</motion.span>
                      )}
                    </AnimatePresence>
                    {submitting ? "Please wait…" : mode==="signin" ? "Sign in → Translator" : "Create account"}
                  </motion.button>

                  <p className="font-technical-data text-xs text-center text-on-surface-variant leading-relaxed">
                    By continuing you agree translations are assistive only — not medical/legal.<br />
                    <a href="#/privacy" onClick={e=>{e.preventDefault();navigate("#/privacy")}} className="text-primary hover:underline">Privacy</a>
                    <span className="mx-1">·</span>
                    <a href="#/research" onClick={e=>{e.preventDefault();navigate("#/research")}} className="text-primary hover:underline">Research limits</a>
                  </p>
                </motion.form>
              </div>

              <div className="px-6 sm:px-8 py-3 bg-surface-container/50 border-t border-outline-variant/20 flex items-center justify-between font-technical-data text-xs">
                <span className="text-on-surface-variant flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-primary" /> Zero-Cloud · 543 landmarks</span>
                <span className="text-outline hidden sm:inline">v1.2-rc · SignSage</span>
              </div>
            </motion.div>
          </BlurFade>
        </div>
      </div>
    </div>
  )
}
