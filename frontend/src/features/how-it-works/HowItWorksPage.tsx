import { Link } from "react-router-dom";
import { ArrowDown, ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import { StartCta } from "@/features/auth/StartCta";

const STEPS = [
  {
    number: "01",
    title: "Capture",
    lead: "Camera input is processed in the browser to extract relevant landmark information.",
    body: "Your device camera stays on your device. MediaPipe models running locally track body pose, facial mesh, and both hands — 543 landmarks per frame — and turn each frame into numbers, not pictures.",
    points: ["Body pose (33 points)", "Face mesh (468 points)", "Left and right hands (21 points each)"],
  },
  {
    number: "02",
    title: "Interpret",
    lead: "The extracted representation is passed through the SignSage interpretation pipeline.",
    body: "Landmark frames stream over a live connection to the backend, where the interpretation pipeline reads the sequence the way a reader follows a sentence — frame by frame, in order.",
    points: ["Live landmark streaming", "Sequence-based interpretation", "Confidence reported per result"],
  },
  {
    number: "03",
    title: "Communicate",
    lead: "The resulting interpretation can be presented as text or visual sign representation.",
    body: "Signs become readable text you can verify at a glance. Going the other way, typed English maps to a gloss sequence performed on a signing avatar.",
    points: ["Sign → text in real time", "Text → gloss → avatar", "History for what you keep"],
  },
];

const FLOW = ["Camera", "Landmarks", "Pipeline", "Text / Visual"];

export default function HowItWorksPage() {
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
        <div className="relative z-10 mx-auto max-w-4xl">
          <p className="animate-rise text-sm font-semibold uppercase tracking-[0.2em] text-foreground/40">
            How it works
          </p>
          <h1
            className="animate-rise mt-6 text-5xl font-semibold tracking-[-0.04em] md:text-7xl"
            style={{ animationDelay: "100ms" }}
          >
            From movement to meaning.
          </h1>
          <p
            className="animate-rise mt-8 max-w-2xl text-lg font-medium leading-8 text-foreground/55"
            style={{ animationDelay: "180ms" }}
          >
            Three stages turn signing into text — and text back into
            signs. Here is what happens at each one.
          </p>
        </div>
      </section>

      <section className="px-6 pb-8">
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <Card>
              <CardContent className="flex flex-wrap items-center gap-3 px-6 py-6">
                {FLOW.map((stage, i) => (
                  <span key={stage} className="flex items-center gap-3">
                    <Badge variant={i === 0 || i === FLOW.length - 1 ? "default" : "secondary"} className="px-3 py-1">
                      {stage}
                    </Badge>
                    {i < FLOW.length - 1 && (
                      <ArrowRight aria-hidden="true" className="h-4 w-4 text-foreground/30" />
                    )}
                  </span>
                ))}
              </CardContent>
            </Card>
          </Reveal>
        </div>
      </section>

      <section className="px-6 py-16">
        <div className="mx-auto max-w-4xl space-y-6">
          {STEPS.map((step, i) => (
            <Reveal key={step.number} delay={i * 70}>
              <Card>
                <CardHeader>
                  <p className="text-sm font-semibold text-foreground/40">{step.number}</p>
                  <CardTitle className="text-3xl">{step.title}</CardTitle>
                  <CardDescription className="text-base leading-7">
                    {step.lead}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="leading-7 text-foreground/60">{step.body}</p>
                  <ul className="flex flex-wrap gap-2">
                    {step.points.map((point) => (
                      <Badge key={point} variant="secondary" className="font-normal">
                        {point}
                      </Badge>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="px-6 py-20">
        <div className="mx-auto max-w-4xl text-center">
          <Reveal>
            <ArrowDown aria-hidden="true" className="mx-auto h-5 w-5 text-foreground/25" />
            <h2 className="mt-6 text-3xl font-semibold tracking-tight md:text-4xl">
              See it in action.
            </h2>
            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <StartCta className="px-6 py-3.5" />
              <Button variant="outline" asChild className="px-6 py-3.5">
                <Link to="/capabilities">Explore capabilities</Link>
              </Button>
            </div>
          </Reveal>
        </div>
      </section>

      <Footer />
    </main>
  );
}
