import { Link } from "react-router-dom";
import {
  ArrowUpRight,
  Bookmark,
  Hand,
  History,
  Languages,
  ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Footer } from "@/components/footer";
import { Navbar } from "@/components/navigation/Navbar";
import { Reveal } from "@/components/reveal";
import Silk from "@/components/Silk";
import { useTheme } from "@/components/theme-provider";
import { Button } from "@/components/ui/button";

const CAPABILITIES = [
  {
    icon: Languages,
    title: "Sign → Text",
    status: "Live",
    text: "Open the camera and sign. Landmarks stream to the interpretation pipeline and results appear as readable text with backend-reported confidence.",
    to: "/app/translate",
    action: "Open Translate",
  },
  {
    icon: Hand,
    title: "Text → Sign",
    status: "Live",
    text: "Type English and get a gloss sequence mapped against the sign library, performed on the signing avatar where assets are bundled.",
    to: "/app/text-to-sign",
    action: "Open Text → Sign",
  },
  {
    icon: History,
    title: "Translation History",
    status: "Account",
    text: "Translations you choose to save are kept per account. Search, filter, review details, and delete anything — individually or all at once.",
    to: "/app/history",
    action: "Open History",
  },
  {
    icon: Bookmark,
    title: "Saved Signs",
    status: "Account",
    text: "Keep useful signs close at hand. Saved signs are your stored translations, browsable as cards with detail and removal.",
    to: "/app/saved-signs",
    action: "Open Saved Signs",
  },
  {
    icon: ShieldCheck,
    title: "Privacy-conscious processing",
    status: "Always on",
    text: "Landmarks are extracted in the browser. Raw camera frames never leave the device, and history is stored only when saving is enabled.",
    to: "/privacy",
    action: "How privacy works",
  },
];

export default function CapabilitiesPage() {
  const { theme } = useTheme();

  return (
    <main className="overflow-x-clip bg-background text-foreground">
      <Navbar />

      <section className="relative overflow-hidden px-6 pb-16 pt-36 md:pt-44">
        <div
          className="absolute inset-0 opacity-60 [mask-image:linear-gradient(to_bottom,black_70%,transparent)]"
          aria-hidden="true"
        >
          <Silk
            speed={4}
            scale={1}
            color={theme === "light" ? "#22c55e" : "#14532d"}
            noiseIntensity={1}
            lightMode={theme === "light"}
          />
        </div>
        <div className="relative z-10 mx-auto max-w-6xl">
          <p className="animate-rise text-sm font-semibold uppercase tracking-[0.2em] text-foreground/40">
            Capabilities
          </p>
          <h1
            className="animate-rise mt-6 max-w-3xl text-5xl font-semibold tracking-[-0.04em] md:text-7xl"
            style={{ animationDelay: "100ms" }}
          >
            What SignSage does.
          </h1>
          <p
            className="animate-rise mt-8 max-w-2xl text-lg font-medium leading-8 text-foreground/55"
            style={{ animationDelay: "180ms" }}
          >
            Everything below exists in the application today. Nothing
            listed is aspirational.
          </p>
        </div>
      </section>

      <section className="px-6 pb-24">
        <div className="mx-auto grid max-w-6xl gap-5 md:grid-cols-2">
          {CAPABILITIES.map((cap, i) => (
            <Reveal key={cap.title} delay={(i % 2) * 80}>
              <Card className="group h-full transition-all duration-300 hover:-translate-y-1 hover:border-transparent hover:shadow-xl hover:shadow-black/5">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted text-foreground/70 transition-colors group-hover:bg-green-500/10 group-hover:text-green-700 dark:group-hover:text-green-400">
                      <cap.icon aria-hidden="true" className="h-5 w-5" />
                    </span>
                    <Badge variant="secondary">{cap.status}</Badge>
                  </div>
                  <CardTitle className="mt-4 text-2xl">{cap.title}</CardTitle>
                  <CardDescription className="text-base leading-7">
                    {cap.text}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Link
                    to={cap.to}
                    className="inline-flex items-center gap-1 text-sm font-medium text-green-700 transition-colors hover:text-green-600 dark:text-green-400 dark:hover:text-green-300"
                  >
                    {cap.action}
                    <ArrowUpRight aria-hidden="true" className="h-4 w-4 transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
                  </Link>
                </CardContent>
              </Card>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="px-6 pb-28">
        <div className="mx-auto max-w-4xl text-center">
          <Reveal>
            <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
              Try them yourself.
            </h2>
            <div className="mt-8">
              <Button className="px-6 py-3.5" render={<Link to="/start" />}>
                Start interpreting
              </Button>
            </div>
          </Reveal>
        </div>
      </section>

      <Footer />
    </main>
  );
}
