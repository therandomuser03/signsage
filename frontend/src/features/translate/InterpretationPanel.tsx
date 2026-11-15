import { useEffect, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Info,
  Loader2,
  MessageSquareText,
  Sparkles,
  Volume2,
  VolumeX,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { cn } from "cn";
import {
  GROQ_ORPHEUS_VOICES,
  SPEECH_RATE_OPTIONS,
  type SpeechController,
  type SpeechRateOption,
} from "./speech";

export type InterpretationStatus =
  | "waiting"
  | "recognising"
  | "composing"
  | "composed"
  | "failed";

export interface InterpretationProvenance {
  /** "llm" | "fallback" | null */
  method: string | null;
  /** How many signs fed the sentence; null when unknown. */
  signs: number | null;
  /** Optional model confidence (0..1) for the sentence; shown only when real. */
  confidence?: number | null;
  /** Model version string. */
  model?: string | null;
  /** History persistence outcome; null = not attempted / nothing to save. */
  saved?: boolean | null;
  historyError?: string | null;
}

export interface GlossChip {
  gloss: string;
  confidence: number;
}

const STATUS_META: Record<InterpretationStatus, { label: string; className: string; dot: string }> = {
  waiting: {
    label: "Waiting for signs",
    className: "border-foreground/15 text-foreground/55",
    dot: "bg-foreground/40",
  },
  recognising: {
    label: "Recognising signs",
    className: "border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400",
    dot: "bg-amber-400",
  },
  composing: {
    label: "Composing…",
    className: "border-foreground/15 text-foreground/65",
    dot: "bg-foreground/40",
  },
  composed: {
    label: "Sentence composed",
    className: "border-green-500/30 bg-green-500/10 text-green-600 dark:text-green-400",
    dot: "bg-green-500",
  },
  failed: {
    label: "Not confidently recognised",
    className: "border-destructive/30 bg-destructive/10 text-destructive",
    dot: "bg-destructive",
  },
};

function provenanceMethodLabel(method: string | null): string | null {
  if (method === "llm") return "AI refinement from the sign sequence";
  if (method === "fallback" || method === "rule") return "Rule-based from the sign sequence";
  return null;
}

function VoiceSelect({
  voices,
  value,
  onChange,
  label,
  allVoices = false,
}: {
  voices: Array<{ id: string; label: string }>;
  value: string | null;
  onChange: (voice: string | null) => void;
  label: string;
  allVoices?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-[11px] font-medium text-foreground/55">{label}</span>
      <select
        value={value ?? "__auto__"}
        onChange={(e) => onChange(e.target.value === "__auto__" ? null : e.target.value)}
        className="mt-1 w-full rounded border border-foreground/20 bg-background px-2 py-1.5 text-xs"
        aria-label={label}
      >
        {!allVoices && <option value="__auto__">Automatic</option>}
        {voices.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label}
          </option>
        ))}
        {allVoices && voices.length === 0 && <option value="__auto__">No voices found</option>}
      </select>
    </label>
  );
}

export function InterpretationPanel({
  sentence,
  status,
  provenance,
  glossSequence,
  emptyHint,
  speech,
}: {
  /** The final interpretation; null when nothing has been composed yet. */
  sentence: string | null;
  status: InterpretationStatus;
  provenance: InterpretationProvenance;
  /** Per-gloss recognition detail for verification; optional. */
  glossSequence?: GlossChip[] | null;
  /** Optional guidance for the waiting state. */
  emptyHint?: string;
  speech: SpeechController;
}) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { prefs, speechState } = speech;

  const clean = sentence?.trim() ?? "";
  const hasSentence = clean.length > 0;
  const statusMeta = STATUS_META[status];

  // Auto-Speak: once per newly completed sentence, never for intermediate results.
  useEffect(() => {
    if (!hasSentence || !prefs.autoSpeak) return;
    void speech.speak(clean, { dedupe: true });
    // `speech` is referentially stable except for its state fields; only the
    // sentence identity should retrigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clean, prefs.autoSpeak, hasSentence]);

  const speakCurrent = () => {
    if (!hasSentence) return;
    if (speechState === "speaking" || speechState === "loading") {
      speech.stop();
      return;
    }
    void speech.speak(clean);
  };

  const methodLabel = provenanceMethodLabel(provenance.method);
  const showSaveStatus = provenance.saved !== null && provenance.saved !== undefined;

  return (
    <section
      aria-label="Sentence interpretation"
      className="rounded-xl border border-foreground/10 bg-card"
    >
      {/* Header: title + live status */}
      <div className="flex items-center justify-between gap-3 border-b border-foreground/10 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <MessageSquareText aria-hidden="true" className="h-4 w-4 shrink-0 text-foreground/55" />
          <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-foreground/55">
            Interpretation
          </h2>
        </div>
        <Badge variant="outline" className={cn("gap-1.5 text-[11px] font-medium", statusMeta.className)}>
          {status === "composing" ? (
            <Loader2 aria-hidden="true" className="h-3 w-3 animate-spin" />
          ) : (
            <span aria-hidden="true" className={cn("h-1.5 w-1.5 rounded-full", statusMeta.dot)} />
          )}
          {statusMeta.label}
        </Badge>
      </div>

      {/* Body */}
      <div className="px-4 py-3">
        {hasSentence ? (
          <>
            {glossSequence && glossSequence.length > 0 && (
              <div className="mb-2 flex flex-wrap gap-1.5" aria-label="Recognised glosses">
                {glossSequence.map((g, i) => (
                  <Badge
                    key={`${g.gloss}-${i}`}
                    variant="secondary"
                    className="font-mono text-[11px] font-normal"
                    title={`${Math.round(g.confidence * 100)}% confidence`}
                  >
                    {g.gloss}
                  </Badge>
                ))}
              </div>
            )}
            <p
              aria-live="polite"
              className="text-2xl font-semibold leading-snug tracking-tight text-foreground"
            >
              {clean}
            </p>
            {/* Metadata — secondary, only when known */}
            <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-foreground/55">
              {methodLabel && <span>{methodLabel}</span>}
              {provenance.signs != null && provenance.signs > 0 && (
                <>
                  <span aria-hidden="true" className="text-foreground/25">·</span>
                  <span>{provenance.signs} sign{provenance.signs === 1 ? "" : "s"}</span>
                </>
              )}
              {provenance.confidence != null && (
                <>
                  <span aria-hidden="true" className="text-foreground/25">·</span>
                  <span>{Math.round(provenance.confidence * 100)}% confidence</span>
                </>
              )}
              {provenance.model && (
                <>
                  <span aria-hidden="true" className="text-foreground/25">·</span>
                  <span className="max-w-40 truncate" title={provenance.model}>{provenance.model}</span>
                </>
              )}
            </div>
            {showSaveStatus && (
              <p className={cn("mt-1 text-[11px]", provenance.saved ? "text-green-600 dark:text-green-400" : "text-destructive")}>
                {provenance.saved ? (
                  <span className="inline-flex items-center gap-1">
                    <Check aria-hidden="true" className="h-3 w-3" />
                    Saved to history
                  </span>
                ) : (
                  <>Not saved{provenance.historyError ? ` — ${provenance.historyError}` : " (sign in to save history)"}</>
                )}
              </p>
            )}
          </>
        ) : status === "composing" ? (
          <div className="flex items-center gap-3 py-5 text-sm text-foreground/60">
            <Loader2 aria-hidden="true" className="h-5 w-5 animate-spin" />
            Composing a sentence from the recognised signs…
          </div>
        ) : status === "failed" ? (
          <div className="rounded-lg border border-dashed border-destructive/25 bg-destructive/5 px-4 py-5 text-center">
            <p className="text-sm font-medium text-destructive">Didn't catch that</p>
            <p className="mt-1 text-xs text-foreground/55">
              {emptyHint ?? "Try signing the sequence again, a little slower and with your hands clearly in frame."}
            </p>
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-foreground/15 px-4 py-5 text-center">
            <p className="text-sm font-medium text-foreground/70">No interpretation yet</p>
            <p className="mt-1 text-xs text-foreground/50">
              {emptyHint ?? "Sign a few words, then press Make sentence — the completed sentence will appear here."}
            </p>
          </div>
        )}
      </div>

      <Separator className="bg-foreground/10" />

      {/* Audio output */}
      <div className="px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <Button
            type="button"
            size="sm"
            className="rounded-full"
            disabled={!hasSentence || speechState === "loading"}
            onClick={speakCurrent}
            title={hasSentence ? "Speak this interpretation" : "Compose a sentence first"}
          >
            {speechState === "loading" ? (
              <Loader2 aria-hidden="true" className="animate-spin" />
            ) : speechState === "speaking" ? (
              <VolumeX aria-hidden="true" />
            ) : (
              <Volume2 aria-hidden="true" />
            )}
            {speechState === "loading" ? "Speaking…" : speechState === "speaking" ? "Stop" : "Speak"}
          </Button>

          <label className="flex cursor-pointer items-center gap-2 text-sm text-foreground/75">
            <Switch
              checked={prefs.autoSpeak}
              onCheckedChange={(v) => speech.setAutoSpeak(v === true)}
              aria-label="Auto-Speak completed interpretations"
            />
            <span className="font-medium">Auto-Speak</span>
          </label>

          {speech.audit?.via === "browser" && prefs.ttsProvider !== "browser" && (
            <span className="text-[11px] text-foreground/45" title="Groq was unavailable; browser voice used.">
              Using browser voice
            </span>
          )}
          {speech.audit?.error && (
            <span className="text-[11px] text-destructive" role="status">
              {speech.audit.error}
            </span>
          )}

          <button
            type="button"
            onClick={() => setSettingsOpen((v) => !v)}
            aria-expanded={settingsOpen}
            className="ml-auto inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-foreground/55 transition-colors hover:bg-foreground/5 hover:text-foreground/80"
          >
            <Info aria-hidden="true" className="h-3.5 w-3.5" />
            Audio settings
            {settingsOpen ? (
              <ChevronDown aria-hidden="true" className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight aria-hidden="true" className="h-3.5 w-3.5" />
            )}
          </button>
        </div>

        {settingsOpen && (
          <div className="mt-3 rounded-lg border border-foreground/10 bg-foreground/[0.03] p-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="block">
                <span className="text-[11px] font-medium text-foreground/55">Voice provider</span>
                <select
                  value={prefs.ttsProvider}
                  onChange={(e) => speech.setTtsProvider(e.target.value as "auto" | "groq" | "browser")}
                  className="mt-1 w-full rounded border border-foreground/20 bg-background px-2 py-1.5 text-xs"
                  aria-label="Voice provider"
                >
                  <option value="auto">Auto (Groq if available, else browser)</option>
                  <option value="groq">Groq Orpheus — natural voice</option>
                  <option value="browser">Browser voice — free, offline-ish</option>
                </select>
              </label>
              {prefs.ttsProvider !== "browser" ? (
                <VoiceSelect
                  voices={GROQ_ORPHEUS_VOICES.map((v) => ({ id: v.id, label: v.label }))}
                  value={prefs.groqTtsVoice}
                  onChange={(v) => speech.setGroqVoice(v ?? "hannah")}
                  label="Orpheus voice"
                />
              ) : (
                <VoiceSelect
                  voices={speech.browserVoices.map((v) => ({ id: v.voiceURI, label: `${v.name} (${v.lang})` }))}
                  value={prefs.browserVoiceURI}
                  onChange={(v) => speech.setBrowserVoiceURI(v)}
                  label="Browser voice"
                  allVoices
                />
              )}
              {prefs.ttsProvider === "browser" && (
                <label className="block">
                  <span className="text-[11px] font-medium text-foreground/55">Playback speed</span>
                  <select
                    value={prefs.speechRate}
                    onChange={(e) => speech.setSpeechRate(Number(e.target.value) as SpeechRateOption)}
                    className="mt-1 w-full rounded border border-foreground/20 bg-background px-2 py-1.5 text-xs"
                    aria-label="Playback speed"
                  >
                    {SPEECH_RATE_OPTIONS.map((r) => (
                      <option key={r} value={r}>
                        {r}×
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
            <p className="mt-2 flex items-start gap-1 text-[10px] leading-snug text-foreground/45">
              <Sparkles aria-hidden="true" className="mt-0.5 h-3 w-3 shrink-0" />
              Groq Orpheus (free tier ≈100 requests/day, ~3.6K tokens/day, 200 characters per request, English only)
              requires SIGNSAGE_GROQ_TTS_API_KEY (or SIGNSAGE_GROQ_API_KEY) on the backend and a one-time terms
              acceptance for the model in the Groq console. If Groq is missing, rate-limited, or not accepted,
              SignSage falls back to browser voices automatically. Playback speed applies to browser voices only.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
