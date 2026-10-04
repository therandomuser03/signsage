import { useEffect, useRef, useState } from "react";
import { CircleAlert, PackageOpen, Play, Sparkles } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AvatarViewer,
  type AvatarViewerHandle,
} from "@/features/text-to-sign/components/AvatarViewer";
import { useSignAnimation } from "@/features/text-to-sign/hooks/useSignAnimation";
import {
  localGlossMapper,
  remoteGlossMapper,
} from "@/features/text-to-sign/services/glossMapper";
import type { GlossSequence, SignManifest } from "@/features/text-to-sign/types";
import { apiClient } from "@/services/api/instance";
import { loadPrefs } from "@/features/settings/preferences";

const VRM_URL = "/models/avatar/signsage.vrm";
const MAX_CHARS = 280;

type Mode = "dictionary" | "assisted";

export default function TextToSignPage() {
  const avatarRef = useRef<AvatarViewerHandle>(null);
  const [manifest, setManifest] = useState<SignManifest | null>(null);
  const [manifestError, setManifestError] = useState(false);
  const [avatarReady, setAvatarReady] = useState<boolean | null>(null);
  const [mode, setMode] = useState<Mode>("dictionary");
  const [text, setText] = useState("");
  const [sequence, setSequence] = useState<GlossSequence | null>(null);
  const [working, setWorking] = useState(false);
  const [glossError, setGlossError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    fetch("/models/avatar/manifest.json")
      .then((res) => {
        if (!res.ok) throw new Error("missing manifest");
        return res.json();
      })
      .then((data) => live && setManifest(data as SignManifest))
      .catch(() => live && setManifestError(true));
    fetch(VRM_URL, { method: "HEAD" })
      .then((res) => live && setAvatarReady(res.ok))
      .catch(() => live && setAvatarReady(false));
    return () => {
      live = false;
    };
  }, []);

  const { perform, state } = useSignAnimation(
    avatarRef,
    manifest ?? {},
    mode === "dictionary"
      ? localGlossMapper
      : (t, m) => remoteGlossMapper(t, m, apiClient, loadPrefs().saveHistory)
  );

  const generate = async () => {
    const value = text.trim();
    if (!manifest || value.length === 0) return;
    setWorking(true);
    setGlossError(null);
    try {
      const result =
        mode === "dictionary"
          ? localGlossMapper(value, manifest)
          : await remoteGlossMapper(value, manifest, apiClient, loadPrefs().saveHistory);
      setSequence(result);
    } catch {
      setGlossError(
        mode === "assisted"
          ? "The backend gloss service could not be reached. Try the dictionary instead."
          : "Gloss generation failed."
      );
      setSequence(null);
    } finally {
      setWorking(false);
    }
  };

  const play = async () => {
    await generate();
    await perform(text.trim());
  };

  const progress =
    state.queue.length > 0 && state.currentIndex >= 0
      ? ((state.currentIndex + 1) / state.queue.length) * 100
      : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Text → Sign</h1>
          <p className="mt-1 text-sm text-foreground/55">
            English text, mapped to a sign gloss sequence. Dictionary
            lookup — not natural ISL grammar.
          </p>
        </div>
        <div className="space-y-0.5 text-right text-xs">
          <p className="font-medium">
            {manifest ? `${Object.keys(manifest).length} signs` : "Loading library…"}
          </p>
          <p className="text-foreground/60">
            {avatarReady === null
              ? "Checking avatar…"
              : avatarReady
                ? "Avatar ready"
                : "Avatar unavailable"}
          </p>
        </div>
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-foreground/10 bg-card">
          <div className="border-b border-foreground/10 px-5 py-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
              01 · Your message
            </p>
            <p className="mt-1 text-xs text-foreground/60">
              Unknown words are dropped, never guessed.
            </p>
          </div>
          <div className="space-y-4 px-5 py-5">
            <Tabs
              value={mode}
              onValueChange={(v) => {
                setMode(v as Mode);
                setSequence(null);
              }}
            >
              <TabsList className="w-full">
                <TabsTrigger value="dictionary" className="flex-1">
                  Dictionary
                </TabsTrigger>
                <TabsTrigger value="assisted" className="flex-1">
                  Assisted
                </TabsTrigger>
              </TabsList>
            </Tabs>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="tts-input">English text</Label>
                <span className="text-xs text-foreground/40 tabular-nums">
                  {text.length}/{MAX_CHARS}
                </span>
              </div>
              <Textarea
                id="tts-input"
                value={text}
                onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
                placeholder="e.g. Hello, thank you for your help"
                rows={4}
                disabled={state.isPlaying}
              />
              <p className="text-xs text-foreground/50">
                {mode === "dictionary"
                  ? "Offline word lookup against the local sign library."
                  : "Backend-assisted mapping against the server vocabulary."}
              </p>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => void generate()}
                disabled={!manifest || working || state.isPlaying || text.trim().length === 0}
                className="flex-1"
              >
                <Sparkles aria-hidden="true" />
                {working ? "Generating…" : "Generate"}
              </Button>
              <Tooltip>
                <TooltipTrigger render={<span className="flex-1" />}>
                  <Button
                    variant="outline"
                    onClick={() => void play()}
                    disabled={
                      !manifest ||
                      !avatarReady ||
                      working ||
                      state.isPlaying ||
                      text.trim().length === 0
                    }
                    className="w-full"
                  >
                    <Play aria-hidden="true" />
                    {state.isPlaying ? "Playing…" : "Play"}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {avatarReady
                    ? "Generate the gloss and perform it on the avatar."
                    : "Needs the avatar assets, which are not bundled."}
                </TooltipContent>
              </Tooltip>
            </div>
            {glossError && (
              <Alert variant="destructive">
                <CircleAlert aria-hidden="true" />
                <AlertTitle>Gloss failed</AlertTitle>
                <AlertDescription>{glossError}</AlertDescription>
              </Alert>
            )}
            {state.isPlaying && (
              <div className="space-y-2">
                <Progress value={progress} aria-label="Playback progress" />
                <p className="text-sm text-foreground/55">
                  {state.currentGlossId
                    ? `Signing ${state.currentGlossId} (${state.currentIndex + 1} of ${state.queue.length})`
                    : "Starting…"}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-foreground/10 bg-card">
          <div className="border-b border-foreground/10 px-5 py-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
              02 · Sign sequence
            </p>
            <p className="mt-1 text-xs text-foreground/60">
              Pre-authored clips on one avatar. No generative model involved.
            </p>
          </div>
          <div className="px-5 py-5">
            <div className="aspect-video overflow-hidden rounded-lg bg-muted">
              {manifest && avatarReady ? (
                <AvatarViewer
                  ref={avatarRef}
                  vrmUrl={VRM_URL}
                  manifest={manifest}
                  className="h-full w-full"
                />
              ) : (
                <Empty className="h-full border-0">
                  <EmptyHeader>
                    <EmptyMedia variant="icon">
                      <PackageOpen aria-hidden="true" className="h-6 w-6" />
                    </EmptyMedia>
                    <EmptyTitle>
                      {manifestError
                        ? "Sign library unavailable"
                        : avatarReady === null
                          ? "Checking avatar assets…"
                          : "Visual sign assets are not currently available"}
                    </EmptyTitle>
                    <EmptyDescription>
                      {manifestError
                        ? "The sign library could not be loaded in this build."
                        : avatarReady === null
                          ? "Looking for the avatar package…"
                          : "The VRM avatar and clips are hand-authored assets and are not bundled here. Gloss preview still works — this is where the performance would play."}
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              )}
            </div>
            <div className="mt-5 border-t border-foreground/10 pt-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
                Gloss sequence
              </p>
              <div className="mt-3">
              {!sequence ? (
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <Sparkles aria-hidden="true" className="h-6 w-6" />
                </EmptyMedia>
                <EmptyTitle>No gloss yet</EmptyTitle>
                <EmptyDescription>
                  Enter text above and press Generate to see its sign sequence.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <div className="space-y-4">
              <p className="text-sm text-foreground/55">
                For: <span className="font-medium text-foreground">“{sequence.sourceText}”</span>
              </p>
              {sequence.glossIds.length > 0 ? (
                <div className="flex flex-wrap gap-2" aria-label="Gloss sequence">
                  {sequence.glossIds.map((id, i) => (
                    <Badge
                      key={`${id}-${i}`}
                      variant={i === state.currentIndex ? "default" : "secondary"}
                      className="px-3 py-1 text-sm"
                    >
                      {id}
                    </Badge>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-foreground/55">
                  No signs matched. Try words from the {Object.keys(manifest ?? {}).length}-sign library, e.g. “hello thank you”.
                </p>
              )}
              {sequence.unmatchedWords.length > 0 && (
                <Alert variant="destructive">
                  <CircleAlert aria-hidden="true" />
                  <AlertTitle>Words without a sign</AlertTitle>
                  <AlertDescription>
                    Dropped, not guessed: {sequence.unmatchedWords.join(", ")}
                  </AlertDescription>
                </Alert>
              )}
              </div>
          )}
          </div>
        </div>
      </div>
    </div>
  </div>
  </div>
  );
}
