import Link from "next/link"
import { ArrowRight, BookOpen, Sparkles } from "lucide-react"
import { Button } from "components/ui/button"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <header className="px-6 md:px-10 h-16 flex items-center justify-between border-b">
        <Link href="/" className="flex items-center gap-2">
          <BookOpen className="h-6 w-6 text-primary" />
          <span className="text-xl font-bold">Summr</span>
        </Link>
        <nav className="flex gap-4 sm:gap-6">
          <Link
            href="/auth/login"
            className="text-sm font-medium hover:underline underline-offset-4"
          >
            Log In
          </Link>
          <Link
            href="/auth/signup"
            className="text-sm font-medium hover:underline underline-offset-4"
          >
            Sign Up
          </Link>
        </nav>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="w-full py-16 md:py-32 xl:py-48 px-6 md:px-10">
          <div className="max-w-6xl mx-auto grid gap-12 lg:grid-cols-2 items-center">
            <div className="flex flex-col space-y-6">
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl xl:text-6xl">
                Your notes, summarized by AI
              </h1>
              <p className="max-w-xl text-muted-foreground md:text-xl">
                Summr helps you focus on what matters by automatically summarizing your notes and thoughts with powerful AI.
              </p>
              <div className="flex flex-col gap-3 min-[400px]:flex-row">
                <Link href="/auth/signup">
                  <Button size="lg" className="gap-1">
                    Get Started <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/auth/login">
                  <Button size="lg" variant="outline">
                    Log In
                  </Button>
                </Link>
              </div>
            </div>

            <div className="flex justify-center">
              <div className="relative w-full max-w-[500px] aspect-square">
                <div className="absolute top-4 left-4 right-4 bottom-4 rounded-xl bg-gradient-to-br from-primary/20 to-primary/10 shadow-xl" />
                <div className="absolute inset-0 rounded-xl bg-white dark:bg-gray-950 shadow-lg p-6 flex flex-col">
                  <div className="h-6 flex items-center justify-between mb-4">
                    <div className="flex gap-1">
                      <div className="h-2 w-2 rounded-full bg-red-500" />
                      <div className="h-2 w-2 rounded-full bg-yellow-500" />
                      <div className="h-2 w-2 rounded-full bg-green-500" />
                    </div>
                    <div className="flex items-center gap-2 text-xs bg-secondary px-2 rounded text-secondary-foreground">
                      <Sparkles className="h-3 w-3" />
                      <span>AI Summarized</span>
                    </div>
                  </div>
                  <div className="flex-1 flex flex-col gap-3 overflow-hidden">
                    <div className="h-6 w-1/2 bg-muted rounded" />
                    <div className="h-4 w-full bg-muted/60 rounded" />
                    <div className="h-4 w-full bg-muted/60 rounded" />
                    <div className="h-4 w-2/3 bg-muted/60 rounded" />
                    <div className="mt-4 h-4 w-full bg-muted/40 rounded" />
                    <div className="h-4 w-full bg-muted/40 rounded" />
                    <div className="h-4 w-4/5 bg-muted/40 rounded" />
                    <div className="mt-4 p-3 bg-primary/10 rounded-lg">
                      <div className="h-4 w-full bg-primary/20 rounded mb-2" />
                      <div className="h-4 w-5/6 bg-primary/20 rounded" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section className="w-full py-16 md:py-32 bg-muted px-6 md:px-10">
          <div className="max-w-6xl mx-auto text-center space-y-6">
            <div className="inline-block rounded-lg bg-primary px-3 py-1 text-sm text-primary-foreground">
              Key Features
            </div>
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
              Everything you need to stay organized
            </h2>
            <p className="max-w-2xl mx-auto text-muted-foreground md:text-xl">
              Summr combines powerful note-taking features with AI summarization to help you capture, organize, and digest information effortlessly.
            </p>

            <div className="mt-12 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
              {[
                {
                  icon: <Sparkles className="h-10 w-10 text-primary mb-4" />,
                  title: "AI Summarization",
                  text: "Automatically generate concise summaries of your notes with powerful AI technology.",
                },
                {
                  icon: (
                    <svg className="h-10 w-10 text-primary mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                    </svg>
                  ),
                  title: "Seamless Sync",
                  text: "Your notes are automatically synced across all your devices, so you're always up to date.",
                },
                {
                  icon: (
                    <svg className="h-10 w-10 text-primary mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                      <polyline points="14 2 14 8 20 8" />
                    </svg>
                  ),
                  title: "Rich Text Editing",
                  text: "Create beautiful, well-formatted notes with our intuitive and powerful text editor.",
                },
                {
                  icon: (
                    <svg className="h-10 w-10 text-primary mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <line x1="4" y1="9" x2="20" y2="9" />
                      <line x1="4" y1="15" x2="20" y2="15" />
                      <line x1="10" y1="3" x2="8" y2="21" />
                      <line x1="16" y1="3" x2="14" y2="21" />
                    </svg>
                  ),
                  title: "Organize with Tags",
                  text: "Categorize your notes with tags to quickly find what you need when you need it.",
                },
                {
                  icon: (
                    <svg className="h-10 w-10 text-primary mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                    </svg>
                  ),
                  title: "Secure Storage",
                  text: "Your notes are encrypted and stored securely, protecting your private information.",
                },
                {
                  icon: (
                    <svg className="h-10 w-10 text-primary mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                      <path d="M2 12h20" />
                    </svg>
                  ),
                  title: "Access Anywhere",
                  text: "Access your notes from any device with a web browser, no installation required.",
                },
              ].map((feature, idx) => (
                <div key={idx} className="flex flex-col items-center text-center border bg-card p-6 rounded-lg shadow-sm">
                  {feature.icon}
                  <h3 className="text-xl font-bold">{feature.title}</h3>
                  <p className="mt-2 text-muted-foreground">{feature.text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="w-full py-16 md:py-32 px-6 md:px-10">
          <div className="max-w-2xl mx-auto text-center space-y-6">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
              Ready to get started?
            </h2>
            <p className="text-muted-foreground md:text-xl">
              Sign up for free and start organizing your thoughts with AI-powered summaries.
            </p>
            <div className="flex justify-center">
              <Link href="/auth/signup">
                <Button size="lg" className="gap-1">
                  Get Started <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="flex flex-col sm:flex-row gap-2 justify-center items-center px-6 md:px-10 py-8 border-t text-center">
        <p className="text-xs text-muted-foreground">© 2025 Summr. All rights reserved.</p>
        <nav className="sm:ml-auto flex gap-4 sm:gap-6">
          <Link className="text-xs hover:underline underline-offset-4" href="#">
            Terms of Service
          </Link>
          <Link className="text-xs hover:underline underline-offset-4" href="#">
            Privacy
          </Link>
        </nav>
      </footer>
    </div>
  )
}
