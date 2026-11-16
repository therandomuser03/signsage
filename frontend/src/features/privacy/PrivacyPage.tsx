import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ShieldCheck } from "lucide-react";
import { Footer } from "@/components/footer";
import { Navbar } from "@/components/navigation/Navbar";
import { Reveal } from "@/components/reveal";
import Silk from "@/components/Silk";
import { useTheme } from "@/components/theme-provider";

const SECTIONS = [
  {
    value: "camera",
    title: "Camera processing",
    body: "Video is captured through your browser's camera API and drawn to a local preview. For the live camera interpreter path, landmark extraction runs on-device in the same page and frames are processed as they arrive and discarded — the application keeps no video and takes no snapshots. Only numerical landmark arrays travel to the backend for that path.",
  },
  {
    value: "landmarks",
    title: "Landmark representation",
    body: "Each processed frame becomes 543 points — body pose, facial mesh, and both hands — each with x, y, and z coordinates. This numerical representation is what travels to the backend for interpretation. It describes movement geometry, not appearance: no faces, no rooms, no backgrounds.",
  },
  {
    value: "groq",
    title: "Third-party AI services (Groq)",
    body: "Recognised words and composed sentences may be sent to Groq to refine sentences and generate speech. Camera video is never sent in the live path; only hand, body and face landmarks are. If you use the microphone fallback, a short recording is sent to Groq to be transcribed and is not stored by SignSage.",
  },
  {
    value: "translation",
    title: "Translation data",
    body: "When saving is enabled, the backend stores the text result of a translation — input text, output text, direction, confidence, and model version. When saving is off, results live only in the current session and disappear when you leave. You control this per session from Settings.",
  },
  {
    value: "account",
    title: "Account and history data",
    body: "Sign-in is handled by Supabase, which stores standard authentication records (email, credentials) under its own terms. Your translation history is stored per account on the SignSage backend and is only visible to you. Deleting an entry — or clearing all history — removes it permanently.",
  },
  {
    value: "control",
    title: "Your control",
    body: "Camera access can be revoked at any time in the browser. Saving can be switched off in Settings. Saved history entries can be deleted individually or wiped entirely from History and Settings. There is no hidden retention: what the interface lets you delete is actually deleted.",
  },
];

export default function PrivacyPage() {
  const { theme } = useTheme();
  // Base UI accordion holds an array of open values; slicing to the last
  // entry preserves the previous single-open collapsible behavior.
  const [openSections, setOpenSections] = useState<string[]>(["camera"]);

  return (
    <main className="overflow-x-clip bg-background text-foreground">
      <Navbar />

      <section className="relative overflow-hidden px-6 pb-14 pt-36 md:pt-44">
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
            Privacy
          </p>
          <h1
            className="animate-rise mt-6 text-5xl font-semibold tracking-[-0.04em] md:text-7xl"
            style={{ animationDelay: "100ms" }}
          >
            Private by architecture.
          </h1>
          <p
            className="animate-rise mt-8 max-w-2xl text-lg font-medium leading-8 text-foreground/55"
            style={{ animationDelay: "180ms" }}
          >
            Raw camera frames remain in the browser. SignSage works with
            landmark representations rather than sending raw video to
            the backend.
          </p>
        </div>
      </section>

      <section className="px-6 pb-16">
        <div className="mx-auto max-w-6xl">
          <Reveal>
            <Alert>
              <ShieldCheck aria-hidden="true" />
              <AlertTitle>The one rule</AlertTitle>
              <AlertDescription>
                If your camera data never leaves your device, it cannot leak
                from that path. The live camera interpreter path sends
                landmark representations, not raw video, to the backend.
              </AlertDescription>
            </Alert>
          </Reveal>
        </div>
      </section>

      <section className="px-6 pb-24">
        <div className="mx-auto max-w-6xl">
          <Reveal>
            <Accordion
              value={openSections}
              onValueChange={(v) => setOpenSections(v.slice(-1))}
              className="w-full"
            >
              {SECTIONS.map((section) => (
                <AccordionItem key={section.value} value={section.value}>
                  <AccordionTrigger className="text-left text-lg font-semibold">
                    {section.title}
                  </AccordionTrigger>
                  <AccordionContent className="text-base leading-7 text-foreground/60">
                    {section.body}
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </Reveal>
          <Reveal delay={80}>
            <p className="mt-10 text-sm leading-6 text-foreground/50">
              Questions about your data? Manage it directly in{" "}
              <Link to="/app/translation-log" className="font-medium text-foreground underline underline-offset-4">
                Translation Log
              </Link>{" "}
              and{" "}
              <Link to="/app/settings" className="font-medium text-foreground underline underline-offset-4">
                Settings
              </Link>
              .
            </p>
          </Reveal>
        </div>
      </section>

      <Footer />
    </main>
  );
}
