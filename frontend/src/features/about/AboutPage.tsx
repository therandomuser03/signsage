import { Link } from "react-router-dom";
import { ArrowDown } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import Silk from "@/components/Silk";
import { Footer } from "@/components/footer";
import { Navbar } from "@/components/navigation/Navbar";
import { Reveal } from "@/components/reveal";
import { useTheme } from "@/components/theme-provider";
import { apiClient } from "@/services/api/instance";
import type { ModelInfo } from "@/types/api";
import { useEffect, useState } from "react";

const APPROACH = [
  {
    title: "Camera",
    text: "Video is captured on the device and never uploaded.",
  },
  {
    title: "Landmark extraction",
    text: "Body, face, and hand landmarks are extracted in the browser.",
  },
  {
    title: "Interpretation pipeline",
    text: "The numerical representation is sent to the SignSage backend.",
  },
  {
    title: "Text / visual representation",
    text: "Results come back as text, or play on a signing avatar.",
  },
];

const STACK = [
  "TypeScript",
  "React",
  "Vite",
  "Tailwind CSS",
  "MediaPipe Tasks Vision",
  "WebSocket",
  "Supabase",
  "Three.js",
  "FastAPI",
  "PyTorch",
];

export default function AboutPage() {
  const { theme } = useTheme();
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [modelsError, setModelsError] = useState(false);

  useEffect(() => {
    let live = true;
    apiClient
      .models()
      .then((res) => live && setModels(res.models))
      .catch(() => live && setModelsError(true));
    return () => {
      live = false;
    };
  }, []);

  return (
    <main className="overflow-x-clip bg-background text-foreground">
      <Navbar />

      {/* 1. Hero */}
      <section className="relative overflow-hidden px-6 pb-24 pt-36 md:pb-32 md:pt-44">
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
            About SignSage
          </p>
          <h1
            className="animate-rise mt-6 text-5xl font-semibold tracking-[-0.04em] md:text-7xl"
            style={{ animationDelay: "100ms" }}
          >
            Technology for more accessible communication.
          </h1>
          <p
            className="animate-rise mt-8 max-w-2xl text-lg font-medium leading-8 text-foreground/55 md:text-xl"
            style={{ animationDelay: "180ms" }}
          >
            SignSage is an Indian Sign Language communication
            platform — exploring interaction between signs, text, and
            visual representations.
          </p>
        </div>
      </section>

      {/* 2. The problem */}
      <section className="border-t border-transparent px-6 py-24 md:py-32">
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              The problem
            </p>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              A gap where language should be.
            </h2>
            <div className="mt-8 max-w-2xl space-y-5 text-lg leading-8 text-foreground/55">
              <p>
                When signing and non-signing people meet without a shared
                language, ordinary moments — appointments, classrooms,
                service counters — become harder than they should be.
              </p>
              <p>
                SignSage explores whether a camera, a browser, and an
                interpretation pipeline can narrow that gap: not by replacing
                human interpreters, but by prototyping what assisted
                ISL communication could feel like.
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      {/* 3. The approach */}
      <section className="border-t border-transparent px-6 py-24 md:py-32">
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              The approach
            </p>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              From camera to meaning.
            </h2>
          </Reveal>
          <div className="mt-12">
            {APPROACH.map((step, i) => (
              <Reveal key={step.title} delay={i * 60}>
                <div className="flex items-start gap-5 py-5">
                  <span className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-transparent bg-muted text-sm font-semibold text-foreground/60">
                    {i + 1}
                  </span>
                  <div>
                    <h3 className="text-xl font-semibold tracking-tight">{step.title}</h3>
                    <p className="mt-1 leading-7 text-foreground/55">{step.text}</p>
                  </div>
                </div>
                {i < APPROACH.length - 1 && (
                  <ArrowDown aria-hidden="true" className="ml-[26px] h-5 w-5 text-foreground/25" />
                )}
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* 4. ISL focus */}
      <section className="border-t border-transparent bg-card px-6 py-24 md:py-32">
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Focus
            </p>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              Built for Indian Sign Language.
            </h2>
            <div className="mt-8 max-w-2xl space-y-5 text-lg leading-8 text-foreground/55">
              <p>
                The project focuses specifically on ISL — its vocabulary,
                its visual grammar, its everyday use — rather than
                claiming to translate every sign language at once.
              </p>
              <p>
                Scope stays within everyday phrases it handles well,
                while broader vocabulary remains under active
                development. What works is labeled as working; what
                doesn’t is labeled too.
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      {/* 5. Focused scope */}
      <section className="border-t border-transparent px-6 py-24 md:py-32">
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Scope
            </p>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              Focused by design.
            </h2>
            <div className="mt-8 max-w-2xl space-y-5 text-lg leading-8 text-foreground/55">
              <p>
                SignSage concentrates on a bounded set of everyday ISL
                communication — a focused vocabulary it handles well —
                rather than claiming to translate everything at once.
              </p>
            </div>
          </Reveal>
          <Reveal delay={80}>
            <Card className="mt-10">
              <CardHeader>
                <CardTitle>Current models</CardTitle>
              </CardHeader>
              <CardContent>
                {models === null && !modelsError ? (
                  <div className="space-y-3" aria-label="Loading models">
                    <Skeleton className="h-10 w-full" />
                    <Skeleton className="h-10 w-full" />
                  </div>
                ) : modelsError || !models ? (
                  <Alert>
                    <AlertTitle>Model status unavailable</AlertTitle>
                    <AlertDescription>
                      The backend could not be reached. Model details
                      appear here when it is online.
                    </AlertDescription>
                  </Alert>
                ) : (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Model</TableHead>
                          <TableHead>Direction</TableHead>
                          <TableHead>Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {models.map((m) => (
                          <TableRow key={m.id}>
                            <TableCell className="font-medium">{m.id}</TableCell>
                            <TableCell className="text-foreground/60">{m.direction}</TableCell>
                            <TableCell>
                              <Badge variant={m.loaded ? "default" : "secondary"}>
                                {m.promoted ? "promoted" : m.loaded ? "loaded" : "idle"}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>
          </Reveal>
        </div>
      </section>

      {/* 6. Privacy */}
      <section className="border-t border-transparent px-6 py-24 md:py-32">
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Privacy
            </p>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              Private by architecture.
            </h2>
            <div className="mt-8 max-w-2xl space-y-5 text-lg leading-8 text-foreground/55">
              <p>
                Camera frames are processed into landmark representations
                in the browser. Only those numerical representations
                travel to the backend — raw video never leaves the
                device, because the system never needs it to.
              </p>
              <p>
                Server-side, only text results are stored, per account,
                and only when saving is enabled. There is nothing to
                leak because there is nothing extra to keep.
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      {/* 7. Technology */}
      <section className="border-t border-transparent bg-card px-6 py-24 md:py-32">
        <div className="mx-auto max-w-4xl">
          <Reveal>
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-foreground/40">
              Technology
            </p>
            <h2 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
              An honest stack.
            </h2>
            <div className="mt-10 flex flex-wrap gap-2">
              {STACK.map((t) => (
                <Badge key={t} variant="secondary" className="px-3 py-1 text-sm font-normal">
                  {t}
                </Badge>
              ))}
            </div>
            <p className="mt-8 max-w-2xl leading-7 text-foreground/55">
              Browser-side landmark tracking, a real-time socket
              protocol, and a Python backend — instrumented
              end to end, with evaluation results published alongside
              the demo instead of hidden behind it.
            </p>
          </Reveal>
        </div>
      </section>

      {/* 8. CTA */}
      <section className="px-6 py-28 md:py-40">
        <div className="mx-auto max-w-4xl text-center">
          <Reveal>
            <h2 className="text-5xl font-semibold tracking-[-0.04em] md:text-7xl">
              Explore SignSage.
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg text-foreground/55">
              Try SignSage and see where it stands.
            </p>
            <Link
              to="/app/translate"
              className="mt-10 inline-flex rounded-xl bg-primary px-7 py-4 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/80"
            >
              Start interpreting
            </Link>
          </Reveal>
        </div>
      </section>

      <Footer />
    </main>
  );
}
