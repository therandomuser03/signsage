import { useCallback, useEffect, useRef, useState } from "react";
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
import { HumanAvatarViewer } from "@/features/text-to-sign/components/HumanAvatarViewer";
import { useSignAnimation } from "@/features/text-to-sign/hooks/useSignAnimation";
import { useSpeechInput, isSpeechServiceBlocked } from "@/features/text-to-sign/hooks/useSpeechInput";
import { useServerDictation } from "@/features/text-to-sign/hooks/useServerDictation";
import {
  EXAMPLE_CHIPS,
  glossChipState,
  glossIdToText,
  missingSignLabel,
  sortVocabulary,
} from "@/features/text-to-sign/signUi";
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
const HUMAN_GLB_URL = "/models/signsage-human.glb";
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
  const [humanReady, setHumanReady] = useState<boolean | null>(null);
  const [mode, setMode] = useState<Mode>("dictionary");
  const [text, setText] = useState("");
  const [sequence, setSequence] = useState<GlossSequence | null>(null);
  const [working, setWorking] = useState(false);
  const [glossError, setGlossError] = useState<string | null>(null);
  const [learnMore, setLearnMore] = useState(false);
  const [vocabIds, setVocabIds] = useState<string[] | null>(null);
  const [dictatLang, setDictatLang] = useState<"en-IN" | "en-US">("en-IN");
  const speech = useSpeechInput({ lang: dictatLang });
  const dictation = useServerDictation();
  const [interimText, setInterimText] = useState("");
  const fallbackTriggered = useRef(false);
  const appendTranscript = useCallback(
    (t: string) => {
      setInterimText("");
      setText((prev) => (prev ? `${prev} ${t}` : t).slice(0, MAX_CHARS));
    },
    [],
  );

  // Switch to the server transcription fallback when the browser's speech
  // service is blocked (network / service-not-allowed).
  useEffect(() => {
    if (
      speech.errorCode &&
      isSpeechServiceBlocked(speech.errorCode) &&
      dictation.supported &&
      !fallbackTriggered.current
    ) {
      fallbackTriggered.current = true;
      void dictation.start(appendTranscript);
    }
    if (!speech.errorCode) fallbackTriggered.current = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speech.errorCode, dictation.supported]);

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

    // Human GLB (text-to-sign default): same existence check semantics.
    (async () => {
      try {
        const exists = await assetExists(HUMAN_GLB_URL);
        if (live) setHumanReady(exists);
      } catch {
        if (live) setHumanReady(false);
      }
    })();

    return () => {
      live = false;
    };
  }, []);

  // Vocabulary chips: the 14 signs from the backend vocabulary endpoint.
  // Falls back to the local manifest when the endpoint is unreachable.
  useEffect(() => {
    let live = true;
    apiClient
      .avatarVocabulary()
      .then((res) => live && setVocabIds(res.gloss_ids))
      .catch(() => live && setVocabIds(null));
    return () => {
      live = false;
    };
  }, []);

  // Clear the live interim preview whenever a capture session ends
  // (stop pressed, final committed, or session ended without a result).
  useEffect(() => {
    if (!speech.listening && !dictation.recording) setInterimText("");
  }, [speech.listening, dictation.recording]);

  const { perform, stop, replay, state, unavailableGlosses } = useSignAnimation(
    avatarRef,
    manifest ?? {},
    mode === "dictionary"
      ? localGlossMapper
      : (t, m) => remoteGlossMapper(t, m, apiClient, loadPrefs().saveHistory)
  );

  const generate = async (overrideText?: string) => {
    const value = (overrideText ?? text).trim();
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

  /** Play one vocabulary sign through the existing playback path. */
  const playSign = async (glossId: string) => {
    if (state.isPlaying || !manifest) return;
    await perform(glossIdToText(glossId));
  };

  const availableSigns = sortVocabulary(vocabIds ?? (manifest ? Object.keys(manifest) : []));

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
  // Human GLB is the default avatar in Text-to-Sign; VRM (if ever bundled) is
  // kept as the pre-authored-clips fallback, procedural figure is last resort.
  const usingHuman = humanReady === true;

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
            {avatarReady === null && humanReady === null
              ? "Checking avatar…"
              : usingVrm
                ? "Avatar ready"
                : usingHuman
                  ? "Human avatar ready"
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
            <div className="flex flex-wrap gap-1.5" aria-label="Example prompts">
              {EXAMPLE_CHIPS.map((chip) => (
                <button
                  key={chip}
                  type="button"
                  disabled={!manifest || working || state.isPlaying}
                  onClick={() => {
                    setText(chip);
                    setGlossError(null);
                    void generate(chip);
                  }}
                  className="rounded-full border border-foreground/15 bg-background px-3 py-1.5 text-xs text-foreground/75 transition-colors hover:border-foreground/30 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
                >
                  {chip}
                </button>
              ))}
            </div>
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
                          variant={speech.listening || dictation.recording ? "destructive" : "outline"}
                          size="icon"
                          disabled={
                            (!speech.supported && !dictation.supported) ||
                            state.isPlaying ||
                            dictation.uploading
                          }
                          onClick={() =>
                            speech.listening
                              ? speech.stop()
                              : dictation.recording
                                ? dictation.stop()
                                : speech.supported
                                  ? void speech.start(
                                      appendTranscript,
                                      (interim) => setInterimText(interim),
                                    )
                                  : void dictation.start(appendTranscript)
                          }
                          aria-label={
                            speech.listening
                              ? "Stop listening"
                              : dictation.recording
                                ? "Stop recording"
                                : "Dictate with microphone"
                          }
                        >
                          <Mic
                            aria-hidden="true"
                            className={speech.listening || dictation.recording ? "motion-safe:animate-pulse" : ""}
                          />
                        </Button>
                      </span>
                    }
                  />
                  <TooltipContent>
                    {speech.listening
                      ? "Listening… press to stop."
                      : dictation.recording
                        ? "Recording… press to stop."
                        : speech.supported
                          ? "Dictate with your microphone."
                          : dictation.supported
                            ? "Dictate via server transcription (sent to Groq; not stored)."
                            : "Speech input is not available in this browser or on the server. Type instead."}
                  </TooltipContent>
                </Tooltip>
              </div>
              {(speech.supported || dictation.supported) && (
                <div className="mt-1.5 flex flex-wrap items-center gap-2">
                  <select
                    value={dictatLang}
                    onChange={(e) => setDictatLang(e.target.value as "en-IN" | "en-US")}
                    disabled={speech.listening || dictation.recording}
                    aria-label="Dictation language"
                    className="rounded border border-foreground/20 bg-background px-1.5 py-1 text-[11px] text-foreground/75"
                  >
                    <option value="en-IN">English (India) · en-IN</option>
                    <option value="en-US">English (US) · en-US</option>
                  </select>
                  {(speech.listening || dictation.recording) && (
                    <div
                      className="h-1.5 min-w-24 flex-1 overflow-hidden rounded bg-foreground/10"
                      role="meter"
                      aria-label="Microphone input level"
                    >
                      <div
                        className="h-full bg-emerald-500 transition-[width] duration-100"
                        style={{
                          width: `${Math.round((speech.listening ? speech.level : dictation.level) * 100)}%`,
                        }}
                      />
                    </div>
                  )}
                  {dictation.recording && (
                    <span className="flex items-center gap-2 text-xs text-foreground/60">
                      <span className="inline-block h-2 w-2 rounded-full bg-red-500 motion-safe:animate-pulse" />
                      Recording… {dictation.secondsLeft}s
                      <Button type="button" size="sm" variant="destructive" onClick={dictation.stop}>
                        Stop
                      </Button>
                    </span>
                  )}
                  {dictation.uploading && (
                    <span className="text-xs text-foreground/60">Transcribing…</span>
                  )}
                </div>
              )}
              {speech.error && (
                <p className="text-xs text-destructive">{speech.error}</p>
              )}
              {dictation.error && (
                <p className="text-xs text-destructive">{dictation.error}</p>
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
            </div>
            {glossError && (
              <Alert variant="destructive">
                <CircleAlert aria-hidden="true" />
                <AlertTitle>Gloss failed</AlertTitle>
                <AlertDescription>{glossError}</AlertDescription>
              </Alert>
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
                : usingHuman
                  ? "Human rig performing a generic articulation demo — not authored ISL. No generative model involved."
                  : "Built-in 3D figure, one motion per sign. No generative model involved."}
            </p>
          </div>
          <div className="px-5 py-5">
            <div className="relative h-[420px] overflow-hidden rounded-lg bg-muted lg:h-[480px]">
              {manifest ? (
                <>
                  {usingHuman ? (
                    <HumanAvatarViewer
                      ref={avatarRef}
                      className="h-full w-full"
                      caption="Human rig · articulation demo — not an ISL sign"
                    />
                  ) : usingVrm ? (
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
                      caption="Procedural motion (illustrative)"
                    />
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
              <div className="mt-2 text-xs text-foreground/60">
                {usingHuman
                  ? "Articulation demonstration, not a validated ISL sign."
                  : "Illustrative animation, not a validated ISL sign."}{" "}
                <button
                  type="button"
                  onClick={() => setLearnMore((v) => !v)}
                  aria-expanded={learnMore}
                  className="font-medium text-foreground/75 underline underline-offset-2 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                >
                  Learn more
                </button>
                {learnMore && (
                  <p className="mt-1 leading-5 text-foreground/55">
                    {usingHuman
                      ? "The human model is real and fully rigged, but no sign animations have been authored or verified for it yet, so each word plays the same generic articulation demo (arms, fingers, blink, mouth). Word lookup and sequencing are real; the movements are not ISL signs."
                      : "The hand-authored VRM avatar and clips are not bundled here, so a built-in 3D figure performs a distinct motion per sign. Word lookup and sequencing are real; the movements are not authored ISL."}
                  </p>
                )}
              </div>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              <Tooltip>
                <TooltipTrigger render={<span className="min-w-0 flex-1" />}>
                  <Button
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
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  speech.stop();
                  setInterimText("");
                  stop();
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
            {state.isPlaying && (
              <div className="mt-3 space-y-1.5">
                <Progress value={progress} aria-label="Playback progress" />
                <p className="text-xs text-foreground/55">
                  {state.currentGlossId
                    ? `Signing ${state.currentGlossId} (${state.currentIndex + 1} of ${state.queue.length})`
                    : "Starting…"}
                </p>
              </div>
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
                  {sequence.glossIds.map((id, i) => {
                    const chip = glossChipState(i, state.currentIndex, state.isPlaying);
                    return (
                      <Badge
                        key={`${id}-${i}`}
                        variant={chip === "current" ? "default" : "secondary"}
                        aria-current={chip === "current" ? "step" : undefined}
                        className={
                          chip === "current"
                            ? "px-3 py-1 text-sm ring-2 ring-green-500/60"
                            : chip === "past"
                              ? "px-3 py-1 text-sm opacity-60"
                              : "px-3 py-1 text-sm"
                        }
                      >
                        {id}
                      </Badge>
                    );
                  })}
                </div>
              ) : (
                <p className="text-sm text-foreground/55">
                  No signs matched. Try words from the {Object.keys(manifest ?? {}).length}-sign library, e.g. “hello thank you”.
                </p>
              )}
              {sequence.unmatchedWords.length > 0 && (
                <div>
                  <p className="text-xs text-foreground/50">
                    Dropped, not guessed — no sign in the library:
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-2" aria-label="Words without a sign">
                    {sequence.unmatchedWords.map((word) => (
                      <Badge
                        key={word}
                        variant="outline"
                        className="px-3 py-1 font-normal text-foreground/50"
                        title="Not in sign library"
                      >
                        {missingSignLabel(word)}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              </div>
          )}
          </div>
        </div>
      </div>

      <section aria-label="Available signs" className="rounded-xl border border-foreground/10 bg-card px-5 py-4">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
          Available signs
        </p>
        <p className="mt-1 text-xs text-foreground/60">
          The {availableSigns.length} signs in the library. Click one to see it performed.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {availableSigns.map((id) => (
            <button
              key={id}
              type="button"
              disabled={state.isPlaying || !manifest}
              onClick={() => void playSign(id)}
              aria-label={`Play sign ${id}`}
              className="rounded-full border border-foreground/15 bg-muted px-3 py-1.5 font-mono text-xs text-foreground/80 transition-colors hover:border-green-600/40 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
            >
              {id}
            </button>
          ))}
        </div>
      </section>
    </div>
  </div>
  </div>
  );
}
