import { useEffect, useRef, useState } from "react";
import { CircleAlert, Eraser, Mic, Play, RotateCcw, Sparkles, Square } from "lucide-react";
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
import { ProceduralAvatarViewer } from "@/features/text-to-sign/components/ProceduralAvatarViewer";
import { useSignAnimation } from "@/features/text-to-sign/hooks/useSignAnimation";
import { useSpeechInput } from "@/features/text-to-sign/hooks/useSpeechInput";
import {
  localGlossMapper,
  remoteGlossMapper,
} from "@/features/text-to-sign/services/glossMapper";
import { assetExists, shouldUseVrm } from "@/features/text-to-sign/services/assetExists";
import { PROCEDURAL_SIGNS } from "@/features/text-to-sign/services/proceduralSigns";
import type { GlossSequence, SignManifest } from "@/features/text-to-sign/types";
import { apiClient } from "@/services/api/instance";
import { loadPrefs } from "@/features/settings/preferences";

const VRM_URL = "/models/avatar/signsage.vrm";
const MANIFEST_URL = "/models/avatar/manifest.json";
const MAX_CHARS = 280;

/** Fallback manifest derived from the procedural motions so the dictionary
 * mapper still resolves glosses when manifest.json is missing or invalid. */
const FALLBACK_MANIFEST: SignManifest = Object.fromEntries(
  Object.keys(PROCEDURAL_SIGNS).map((id) => [id, { id, label: id, animationUrl: "" }]),
);

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
  const speech = useSpeechInput({ lang: "en-IN" });
  const [interimText, setInterimText] = useState("");

  useEffect(() => {
    let live = true;

    // Manifest: if the response is HTML (Vercel SPA fallback) or invalid JSON,
    // treat it as unavailable and fall back to the procedural motions.
    (async () => {
      try {
        if (!(await assetExists(MANIFEST_URL))) {
          throw new Error("manifest not available");
        }
        const res = await fetch(MANIFEST_URL);
        const data = (await res.json()) as SignManifest;
        if (live && data && typeof data === "object") {
          setManifest(data);
        } else {
          throw new Error("invalid manifest");
        }
      } catch {
        if (live) {
          setManifestError(true);
          setManifest(FALLBACK_MANIFEST);
        }
      }
    })();

    // VRM: use assetExists so an HTML 200 (Vercel rewrite) is rejected.
    (async () => {
      try {
        const exists = await assetExists(VRM_URL);
        if (live) setAvatarReady(exists);
      } catch {
        if (live) setAvatarReady(false);
      }
    })();

    return () => {
      live = false;
    };
  }, []);

  // Clear the live interim preview whenever a recognition session ends
  // (stop pressed, final committed, or session ended without a result).
  useEffect(() => {
    if (!speech.listening) setInterimText("");
  }, [speech.listening]);

  const { perform, stop, replay, state, unavailableGlosses } = useSignAnimation(
    avatarRef,
    manifest ?? {},
    mode === "dictionary"
      ? localGlossMapper
      : (t, m) => remoteGlossMapper(t, m, apiClient, loadPrefs().saveHistory)
  );

  const generate = async () => {
    const value = text.trim();
    if (!manifest || value.length === 0) return null;
    setWorking(true);
    setGlossError(null);
    try {
      const result =
        mode === "dictionary"
          ? localGlossMapper(value, manifest)
          : await remoteGlossMapper(value, manifest, apiClient, loadPrefs().saveHistory);
      setSequence(result);
      return result;
    } catch {
      setGlossError(
        mode === "assisted"
          ? "The backend gloss service could not be reached. Try the dictionary instead."
          : "Gloss generation failed."
      );
      setSequence(null);
      return null;
    } finally {
      setWorking(false);
    }
  };

  const play = async () => {
    if (state.isPlaying) return;
    const result = await generate();
    if (result && result.glossIds.length > 0) await perform(text.trim());
  };

  const clear = () => {
    stop();
    speech.stop();
    setText("");
    setInterimText("");
    setSequence(null);
    setGlossError(null);
  };

  // The VRM avatar needs hand-authored assets; the procedural avatar is
  // always available, so playback never depends on the HEAD check.
  const usingVrm = shouldUseVrm(avatarReady === true, manifest !== null);

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
              : usingVrm
                ? "Avatar ready"
                : "Procedural motion"}
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
              <div className="flex gap-2">
                <Textarea
                  id="tts-input"
                  value={text}
                  onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
                  placeholder="e.g. Hello, thank you for your help"
                  rows={4}
                  disabled={state.isPlaying}
                  className="flex-1"
                />
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <span>
                        <Button
                          variant={speech.listening ? "destructive" : "outline"}
                          size="icon"
                          disabled={!speech.supported || state.isPlaying}
                          onClick={() =>
                            speech.listening
                              ? speech.stop()
                              : speech.start(
                                  (final) => {
                                    setInterimText("");
                                    setText((prev) =>
                                      (prev ? `${prev} ${final}` : final).slice(0, MAX_CHARS)
                                    );
                                  },
                                  (interim) => setInterimText(interim)
                                )
                          }
                          aria-label={speech.listening ? "Stop listening" : "Dictate with microphone"}
                        >
                          <Mic aria-hidden="true" className={speech.listening ? "motion-safe:animate-pulse" : ""} />
                        </Button>
                      </span>
                    }
                  />
                  <TooltipContent>
                    {speech.supported
                      ? speech.listening
                        ? "Listening… press to stop."
                        : "Dictate with your microphone."
                      : "Speech recognition is not supported in this browser. Type instead."}
                  </TooltipContent>
                </Tooltip>
              </div>
              {speech.error && (
                <p className="text-xs text-destructive">{speech.error}</p>
              )}
              {speech.listening && interimText && (
                <p className="text-xs text-foreground/50">
                  <span className="font-medium">Listening:</span>{" "}
                  {interimText}
                </p>
              )}
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
                  Generate the gloss and perform it on the avatar.
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  speech.stop();
                  setInterimText("");
                }}
                disabled={!state.isPlaying}
                className="flex-1"
              >
                <Square aria-hidden="true" />
                Stop
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void replay()}
                disabled={state.isPlaying || state.queue.length === 0}
                className="flex-1"
              >
                <RotateCcw aria-hidden="true" />
                Replay
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={clear}
                disabled={state.isPlaying || (text.length === 0 && sequence === null)}
                className="flex-1"
              >
                <Eraser aria-hidden="true" />
                Clear
              </Button>
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
              {usingVrm
                ? "Pre-authored clips on one avatar. No generative model involved."
                : "Built-in 3D figure, one motion per sign. No generative model involved."}
            </p>
          </div>
          <div className="px-5 py-5">
            <div className="aspect-video overflow-hidden rounded-lg bg-muted">
              {manifest ? (
                <>
                  {usingVrm ? (
                    <AvatarViewer
                      ref={avatarRef}
                      vrmUrl={VRM_URL}
                      manifest={manifest}
                      className="h-full w-full"
                    />
                  ) : (
                    <ProceduralAvatarViewer
                      ref={avatarRef}
                      className="h-full w-full"
                      caption={
                        state.currentGlossId ??
                        (state.isPlaying ? "Signing…" : "Procedural motion")
                      }
                    />
                  )}
                  {!usingVrm && manifest && (
                    <p className="mt-2 text-xs text-foreground/50 text-center">
                      Illustrative animation (approximate). Not a validated ISL sign.
                    </p>
                  )}
                </>
              ) : (
                <Empty className="h-full border-0">
                  <EmptyHeader>
                    <EmptyMedia variant="icon">
                      <Sparkles aria-hidden="true" className="h-6 w-6" />
                    </EmptyMedia>
                    <EmptyTitle>
                      {manifestError
                        ? "Sign library unavailable"
                        : "Checking avatar assets…"}
                    </EmptyTitle>
                    <EmptyDescription>
                      {manifestError
                        ? "The sign library could not be loaded in this build."
                        : "Looking for the avatar package…"}
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              )}
            </div>
            {!usingVrm && manifest && (
              <Alert className="mt-3">
                <CircleAlert aria-hidden="true" />
                <AlertTitle>Motion mode</AlertTitle>
                <AlertDescription>
                  The hand-authored VRM avatar and clips are not bundled here,
                  so a built-in 3D figure performs a distinct motion per sign.
                  Word lookup and sequencing are real; the movements are not
                  authored ISL.
                </AlertDescription>
              </Alert>
            )}
            {unavailableGlosses.length > 0 && (
              <Alert className="mt-3" variant="destructive">
                <CircleAlert aria-hidden="true" />
                <AlertTitle>Sign animation unavailable</AlertTitle>
                <AlertDescription>
                  {unavailableGlosses.map((id) => (
                    <div key={id} className="mt-1">
                      Sign animation unavailable for <span className="font-mono">{id}</span>
                    </div>
                  ))}
                </AlertDescription>
              </Alert>
            )}
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
