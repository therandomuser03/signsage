import { Link } from "react-router-dom";
import {
  ArrowDown,
  ArrowUpRight,
  Camera,
  Monitor,
  ScanLine,
  Server,
} from "lucide-react";
import DotGrid from "@/components/DotGrid";
import GhostFibers from "@/components/GhostFibers";
import { Footer } from "@/components/footer";
import { StartCta } from "@/features/auth/StartCta";
import { Navbar } from "@/components/navigation/Navbar";
import { Reveal } from "@/components/reveal";
import { useTheme } from "@/components/theme-provider";

function ProductPreview() {
  return (
    <div className="overflow-hidden rounded-3xl border border-transparent bg-card shadow-2xl shadow-black/5">
      <div className="flex items-center justify-between border-b border-transparent px-5 py-4">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-green-500 motion-safe:animate-pulse" />
          <span className="text-sm font-medium">Sign → Text</span>
        </div>

        <span className="text-xs text-foreground/50">Live</span>
      </div>

      <div className="grid min-h-105 place-items-center bg-muted">
        <div className="text-center">
          <div className="mx-auto mb-4 flex h-28 w-28 items-center justify-center rounded-full bg-muted-foreground/10">
            <div className="h-16 w-16 rounded-full bg-muted-foreground/20" />
          </div>

          <p className="text-sm text-foreground/40">
            Camera preview
          </p>
        </div>
      </div>

      <div className="space-y-5 border-t border-transparent p-6">
        <div>
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-foreground/40">
            Detected sign
          </p>

          <p className="text-2xl font-semibold tracking-tight">
            HELLO
          </p>
        </div>

        <div>
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-foreground/40">
            Translation
          </p>

          <p className="text-xl text-foreground/70">
            Hello!
          </p>
        </div>
      </div>
    </div>
  );
}

const HOW_IT_WORKS = [
  {
    number: "01",
    title: "Capture",
    text: "Landmark information is extracted from the camera feed in the browser.",
  },
  {
    number: "02",
    title: "Interpret",
    text: "The extracted representation is sent to the SignSage interpretation pipeline.",
  },
  {
    number: "03",
    title: "Communicate",
    text: "The resulting interpretation is presented as text or a visual sign representation.",
  },
];

const CAPABILITIES = [
  {
    number: "01",
    title: "Sign → Text",
    text: "Experiment with real-time sign-to-text using browser-extracted landmarks.",
    to: "/app/translate",
  },
  {
    number: "02",
    title: "Text → Sign",
    text: "Demonstrate how interpreted language can be represented visually.",
    to: "/app/text-to-sign",
  },
  {
    number: "03",
    title: "Translation History",
    text: "Review past translations with user-controlled application data.",
    to: "/app/history",
  },
];

const ISL_FLOW = [
  {
    title: "Indian Sign Language",
    text: "The everyday communication SignSage explores.",
  },
  {
    title: "Landmark representation",
    text: "Body, face, and hand landmarks extracted in the browser.",
  },
  {
    title: "Interpretation pipeline",
    text: "Representations sent to the SignSage backend for interpretation.",
  },
  {
    title: "Text / visual representation",
    text: "Results presented as text or visual sign representations.",
  },
];

const PRIVACY_FLOW = [
  {
    icon: Camera,
    title: "Camera",
    text: "Video is captured on the device.",
  },
  {
    icon: Monitor,
    title: "Browser",
    text: "Frames are processed locally.",
  },
  {
    icon: ScanLine,
    title: "Landmark representation",
    text: "Only landmarks leave the browser.",
  },
  {
    icon: Server,
    title: "Backend",
    text: "Landmarks are interpreted server-side.",
  },
];

export default function LandingPage() {
  const { theme } = useTheme();

  return (
    <main className="overflow-x-clip bg-background text-foreground">
      <Navbar />

      {/* Hero */}
      <section className="relative overflow-hidden px-6 pb-20 pt-36 md:pb-32 md:pt-44">
        <div
          className="absolute inset-0 opacity-70 mask-[linear-gradient(to_bottom,black_72%,transparent)]"
          aria-hidden="true"
        >
          <GhostFibers
            lineColor={theme === "light" ? "#15803d" : "#22c55e"}
            glowColor={theme === "light" ? "#16a34a" : "#4ade80"}
            blueBoost={0.5}
            brightness={theme === "light" ? 2.5 : 0.85}
            glowIntensity={theme === "light" ? 1.6 : 1.1}
            layers={theme === "light" ? 5 : 4}
            lightMode={theme === "light"}
          />
        </div>
        <div className="relative z-10 mx-auto max-w-6xl">
          <div className="max-w-4xl">
            <p
              className="animate-rise mb-6 text-sm font-semibold uppercase tracking-[0.2em] text-foreground/40"
              style={{ animationDelay: "40ms" }}
            >
              Indian Sign Language
            </p>

            <h1
              className="animate-rise text-5xl font-semibold tracking-[-0.04em] md:text-7xl lg:text-8xl"
              style={{ animationDelay: "120ms" }}
            >
              Communication,
              <br />
              without the barrier.
            </h1>

            <p
              className="animate-rise mt-8 max-w-2xl text-lg font-medium leading-8 text-foreground/55 md:text-xl"
              style={{ animationDelay: "200ms" }}
            >
              SignSage is an Indian Sign Language communication
              platform — real-time interpretation between signs, text,
              and visual representations.
            </p>

            <div
              className="animate-rise mt-10 flex flex-wrap gap-3"
              style={{ animationDelay: "280ms" }}
            >
              <StartCta className="rounded-xl px-6 py-3.5 text-sm font-medium" />

              <Link
                to="/how-it-works"
                className="rounded-xl border border-transparent bg-card px-6 py-3.5 text-sm font-medium transition-colors hover:bg-muted"
              >
                See how it works
              </Link>
            </div>
          </div>

          <div
            className="animate-rise mt-20 md:mt-28"
            style={{ animationDelay: "360ms" }}
          >
            <ProductPreview />
          </div>
        </div>
      </section>

      {/* How it works */}
      <section
        id="how-it-works"
        className="relative scroll-mt-24 overflow-hidden border-t border-transparent px-6 py-24 md:py-32"
      >
        <div className="absolute inset-0" aria-hidden="true">
          <DotGrid
            dotSize={5}
            gap={26}
            baseColor={theme === "light" ? "#d4d4d8" : "#3f3f46"}
            activeColor="#22c55e"
            proximity={120}
          />
        </div>
        <div
          aria-hidden="true"
          className="absolute inset-x-0 top-0 h-32 bg-linear-to-b from-background to-transparent"
        />
        <div className="relative z-10 mx-auto max-w-6xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              How it works
            </p>

            <h2 className="mt-5 max-w-3xl text-4xl font-semibold tracking-tight md:text-6xl">
              From movement to meaning.
            </h2>
          </Reveal>

          <div className="mt-16 grid gap-5 md:grid-cols-3">
            {HOW_IT_WORKS.map((item, i) => (
              <Reveal key={item.number} delay={i * 90}>
                <div className="group rounded-3xl border border-transparent bg-card/70 p-7 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-transparent hover:shadow-xl hover:shadow-black/5">
                  <span className="text-sm text-green-500 font-medium text-foreground/35">
                    {item.number}
                  </span>

                  <h3 className="mt-12 text-2xl font-semibold">
                    {item.title}
                  </h3>

                  <p className="mt-4 leading-7 text-foreground/55">
                    {item.text}
                  </p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ISL story */}
      <section
        id="isl"
        className="scroll-mt-24 border-t border-transparent px-6 py-24 md:py-32"
      >
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Focus
            </p>

            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              Built for Indian Sign Language.
            </h2>

            <p className="mt-6 max-w-2xl text-lg leading-8 text-foreground/55">
              SignSage brings Indian Sign Language communication
              together — from capture to interpretation.
            </p>
          </Reveal>

          <div className="mt-14">
            {ISL_FLOW.map((step, i) => (
              <Reveal key={step.title} delay={i * 60}>
                <div className="flex items-start gap-5 py-5">
                  <span className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-transparent bg-muted text-sm font-semibold text-foreground/60">
                    {i + 1}
                  </span>
                  <div>
                    <h3 className="text-xl font-semibold tracking-tight">
                      {step.title}
                    </h3>
                    <p className="mt-1 leading-7 text-foreground/55">
                      {step.text}
                    </p>
                  </div>
                </div>
                {i < ISL_FLOW.length - 1 && (
                  <ArrowDown
                    aria-hidden="true"
                    className="ml-6.5 h-5 w-5 text-foreground/25"
                  />
                )}
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section
        id="capabilities"
        className="scroll-mt-24 border-t border-transparent px-6 py-24 md:py-32"
      >
        <div className="mx-auto max-w-6xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Capabilities
            </p>

            <h2 className="mt-5 max-w-3xl text-4xl font-semibold tracking-tight md:text-6xl">
              Built around the communication experience.
            </h2>
          </Reveal>

          <div className="mt-12">
            {CAPABILITIES.map((item, i) => (
              <Reveal key={item.number} delay={i * 60}>
                <Link
                  to={item.to}
                  className="group flex items-center gap-6 border-t border-transparent py-7 transition-colors last:border-b last:border-transparent hover:bg-muted/50 sm:gap-10 sm:px-4"
                >
                  <span className="text-sm font-medium text-foreground/35">
                    {item.number}
                  </span>
                  <span className="flex-1">
                    <span className="block text-2xl font-semibold tracking-tight transition-transform duration-300 group-hover:translate-x-1 md:text-3xl">
                      {item.title}
                    </span>
                    <span className="mt-2 block max-w-xl leading-7 text-foreground/55">
                      {item.text}
                    </span>
                  </span>
                  <ArrowUpRight
                    aria-hidden="true"
                    className="h-6 w-6 shrink-0 text-foreground/30 transition-all duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-foreground"
                  />
                </Link>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Privacy */}
      <section
        id="privacy"
        className="scroll-mt-24 border-y border-transparent bg-card px-6 py-24 md:py-32"
      >
        <div className="mx-auto max-w-4xl text-center">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Privacy
            </p>

            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              Designed to minimize unnecessary data collection.
            </h2>

            <p className="mx-auto mt-8 max-w-2xl text-lg leading-8 text-foreground/55">
              SignSage processes camera input through landmark
              representations rather than sending raw camera frames
              to the backend.
            </p>
          </Reveal>

          <div className="mx-auto mt-14 grid max-w-2xl gap-2 text-left sm:grid-cols-2">
            {PRIVACY_FLOW.map((step, i) => (
              <Reveal key={step.title} delay={i * 70}>
                <div className="flex h-full items-start gap-4 rounded-2xl border border-transparent bg-background p-5">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-muted text-foreground/70">
                    <step.icon aria-hidden="true" className="h-5 w-5" />
                  </span>
                  <div>
                    <h3 className="font-semibold tracking-tight">
                      {step.title}
                    </h3>
                    <p className="mt-1 text-sm leading-6 text-foreground/55">
                      {step.text}
                    </p>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>

          <Reveal delay={120}>
            <p className="mx-auto mt-10 inline-block rounded-full border border-green-600/25 bg-green-500/10 px-5 py-2.5 text-sm font-medium text-green-700 dark:text-green-400">
              Raw camera frames stay in the browser.
            </p>
          </Reveal>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-28 md:py-40">
        <div className="mx-auto max-w-4xl text-center">
          <Reveal>
            <h2 className="text-5xl font-semibold tracking-[-0.04em] md:text-7xl">
              Let’s make communication more accessible.
            </h2>

            <p className="mx-auto mt-6 max-w-xl text-lg text-foreground/55">
              Open SignSage and start interpreting.
            </p>

            <StartCta className="mt-10 inline-flex rounded-xl px-7 py-4 text-sm font-medium" />
          </Reveal>
        </div>
      </section>

      <Footer />
    </main>
  );
}
