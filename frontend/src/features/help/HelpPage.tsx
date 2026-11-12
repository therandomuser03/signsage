import { ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Footer } from "@/components/footer";
import { Navbar } from "@/components/navigation/Navbar";
import { Reveal } from "@/components/reveal";
import Silk from "@/components/Silk";
import { useTheme } from "@/components/theme-provider";

/**
 * ISL dictionary help page.
 *
 * Lists common Indian Sign Language words and links each to the ISL portal
 * homepage so users can look up a sign demonstration.
 *
 * NOTE: indiansignlanguage.org blocks automated requests (403 on
 * /dictionary/<word>), so the per-word pattern below could not be verified
 * against the live site. Every word card therefore links to the portal
 * homepage instead. The portal belongs to a third party; this page only links
 * to it and is not affiliated with it.
 *
 * The per-word URL pattern is kept only as a clearly marked, unverified
 * reference: ISL_BASE_URL + "/" + <word>.
 */
const ISL_PORTAL_HOME = "https://indiansignlanguage.org/";

// Unverified per-word reference pattern (kept as a clearly marked constant,
// not used for links): https://indiansignlanguage.org/dictionary/<word>
const ISL_BASE_URL = "https://indiansignlanguage.org/dictionary";

const ISL_WORDS = [
  "hello",
  "yes",
  "no",
  "you",
  "help",
  "thank you",
  "water",
  "where",
  "me",
  "rest",
  "friend",
  "hospital",
  "market",
  "school",
  "please",
  "sorry",
  "stop",
  "go",
  "come",
  "eat",
  "drink",
  "sleep",
  "love",
  "name",
  "what",
  "who",
  "when",
  "why",
  "how",
  "good",
  "bad",
];

export default function HelpPage() {
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
            ISL Dictionary
          </p>
          <h1
            className="animate-rise mt-6 text-5xl font-semibold tracking-[-0.04em] md:text-7xl"
            style={{ animationDelay: "100ms" }}
          >
            Learn how to sign
          </h1>
          <p
            className="animate-rise mt-8 max-w-2xl text-lg font-medium leading-8 text-foreground/55 md:text-xl"
            style={{ animationDelay: "180ms" }}
          >
            Common Indian Sign Language words, with links to their
            demonstrations on{" "}
            <a
              href="https://indiansignlanguage.org/"
              className="text-foreground underline underline-offset-4 hover:text-primary transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              indiansignlanguage.org
            </a>
            .
          </p>
        </div>
      </section>

      <section className="border-t border-transparent px-6 pb-24 md:py-32">
        <div className="mx-auto max-w-6xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Browse signs
            </p>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              Pick a word to see its sign
            </h2>
            <p className="mt-4 max-w-2xl text-lg leading-8 text-foreground/55">
              Each card opens the sign&apos;s page on indiansignlanguage.org
              so you can watch the demonstration and learn the hand shape.
            </p>
            <p className="mt-4 text-xs text-foreground/40">
              Per-word dictionary links on indiansignlanguage.org follow the
              unverified pattern{" "}
              <code className="font-mono">{ISL_BASE_URL}/&lt;word&gt;</code>
              . This page links to the portal homepage instead because the
              per-word pattern could not be confirmed against the live site.
              The portal belongs to a third party; SignSage only links to it
              and is not affiliated with it.
            </p>
          </Reveal>

          <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {ISL_WORDS.map((word) => (
              <Card
                key={word}
                className="group cursor-pointer hover:bg-card/50 transition-colors"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold tracking-tight">
                    {word.charAt(0).toUpperCase() + word.slice(1)}
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between">
                    <Badge variant="secondary">ISL sign</Badge>
                    <a
                      href={ISL_PORTAL_HOME}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 text-xs font-medium text-foreground hover:text-primary transition-colors px-3 py-1.5 rounded-lg border border-border hover:bg-muted"
                    >
                      View sign on ISL portal <ExternalLink size={14} />
                    </a>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
