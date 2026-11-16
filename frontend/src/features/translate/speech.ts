// SignSage — speech engine for the Live Interpreter.
//
// Speaks ONLY completed interpretations (the UI decides when to call
// `speak`; intermediate predictions are never passed in). Order of
// preference per the user's prefs:
//
//   "piper"   — backend Piper voice (WAV, free, offline). Falls back to
//               browser voices when unavailable.
//   "groq"    — backend Groq Orpheus (WAV). Falls back to browser voices
//               when the backend is unavailable/misconfigured/limited.
//   "auto"    — backend Piper when available (Groq only when you pick it);
//               browser fallback otherwise.
//   "browser" — browser Web Speech API only; never calls the backend.
//
// API keys are NOT here: the browser asks the backend, which proxies Groq.
import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "../../services/api/instance";
import { loadPrefs, savePrefs, type TtsProvider, type TranslationPrefs } from "../settings/preferences";

export type SpeechState = "idle" | "loading" | "speaking" | "error";

/** Status of the last speech attempt. */
export interface SpeechAudit {
  /** Which provider actually produced (or would have produced) the audio. */
  via: "groq" | "piper" | "browser" | "none";
  /** Machine code from the backend 503 (e.g. TTS_MODEL_TERMS_REQUIRED), when a server voice was tried. */
  code?: string;
  /** Present when speaking failed. */
  error?: string;
}

/** Auto-Speak dedupe: speak a sentence only if it differs from the last one. */
export function shouldSpeak(lastSpoken: string | null, text: string): boolean {
  return text.trim().length > 0 && text.trim() !== lastSpoken;
}

export const GROQ_ORPHEUS_VOICES = [
  { id: "autumn", label: "Autumn (F)" },
  { id: "diana", label: "Diana (F)" },
  { id: "hannah", label: "Hannah (F)" },
  { id: "austin", label: "Austin (M)" },
  { id: "daniel", label: "Daniel (M)" },
  { id: "troy", label: "Troy (M)" },
] as const;

export type SpeechRateOption = 0.75 | 0.9 | 1 | 1.25;
export const SPEECH_RATE_OPTIONS: SpeechRateOption[] = [0.75, 0.9, 1, 1.25];

export function browserSupportsSpeech(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function pickBrowserVoice(voices: SpeechSynthesisVoice[], preferredURI: string | null): SpeechSynthesisVoice | null {
  if (voices.length === 0) return null;
  if (preferredURI) {
    const match = voices.find((v) => v.voiceURI === preferredURI);
    if (match) return match;
  }
  for (const pref of ["en-IN", "en-GB", "en-US", "en"]) {
    const match = voices.find((v) => v.lang.toLowerCase().startsWith(pref.toLowerCase()));
    if (match) return match;
  }
  return voices[0];
}

export function useSpeech() {
  const [prefs, setPrefs] = useState<TranslationPrefs>(() => loadPrefs());
  const [speechState, setSpeechState] = useState<SpeechState>("idle");
  const [audit, setAudit] = useState<SpeechAudit | null>(null);
  const [browserVoices, setBrowserVoices] = useState<SpeechSynthesisVoice[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const lastSpokenRef = useRef<string | null>(null);
  const generationRef = useRef(0);

  // Browser voices load asynchronously in some engines.
  useEffect(() => {
    if (!browserSupportsSpeech()) return;
    const synth = window.speechSynthesis;
    const update = () => setBrowserVoices(synth.getVoices());
    update();
    synth.addEventListener("voiceschanged", update);
    return () => synth.removeEventListener("voiceschanged", update);
  }, []);

  const stop = useCallback(() => {
    generationRef.current += 1; // invalidate in-flight work
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    if (browserSupportsSpeech()) window.speechSynthesis.cancel();
    setSpeechState("idle");
  }, []);

  // Never leak audio on unmount.
  useEffect(() => stop, [stop]);

  const speakWithBrowser = useCallback(
    (text: string): SpeechAudit => {
      if (!browserSupportsSpeech()) {
        return { via: "none", error: "Speech synthesis is not available in this browser." };
      }
      const synth = window.speechSynthesis;
      const voices = browserVoices.length > 0 ? browserVoices : synth.getVoices();
      const voice = pickBrowserVoice(voices, prefs.browserVoiceURI);
      if (!voice) {
        return {
          via: "none",
          error: `No browser voice is installed (${voices.length} detected). Linux: install speech-dispatcher and espeak-ng, then restart the browser. Or use Chrome or Edge.`,
        };
      }
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.voice = voice;
      utterance.lang = voice.lang;
      utterance.rate = prefs.speechRate;
      const generation = generationRef.current;
      utterance.onend = () => {
        if (generationRef.current === generation) setSpeechState("idle");
      };
      utterance.onerror = () => {
        if (generationRef.current !== generation) return;
        setSpeechState("error");
        setAudit({ via: "browser", error: "Playback was blocked or failed." });
      };
      synth.speak(utterance);
      return { via: "browser" };
    },
    [browserVoices, prefs.browserVoiceURI, prefs.speechRate],
  );

  /** Speak `text`. `dedupe` (auto-speak): skip if this exact text was just spoken. */
  const speak = useCallback(
    async (text: string, { dedupe = false }: { dedupe?: boolean } = {}) => {
      const clean = text.trim();
      if (!clean) return { via: "none" } as SpeechAudit;
      if (dedupe && !shouldSpeak(lastSpokenRef.current, clean)) {
        return { via: "none" } as SpeechAudit; // identical sentence already spoken
      }
      stop();
      const generation = generationRef.current;
      lastSpokenRef.current = clean;

      if (prefs.ttsProvider === "browser") {
        setSpeechState("speaking");
        const result = speakWithBrowser(clean);
        setAudit(result);
        if (result.via === "none") setSpeechState("error");
        return result;
      }

      // Server path: backend Piper (auto/piper) or Groq Orpheus (groq);
      // any failure falls back to browser voices.
      setSpeechState("loading");
      const backendProvider = prefs.ttsProvider === "auto" ? undefined : prefs.ttsProvider;
      try {
        const blob = await apiClient.speakWav(clean, prefs.groqTtsVoice, backendProvider);
        if (generationRef.current !== generation) return { via: "none" } as SpeechAudit; // stopped meanwhile
        const url = URL.createObjectURL(blob);
        audioUrlRef.current = url;
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => {
          if (generationRef.current !== generation) return;
          URL.revokeObjectURL(url);
          audioUrlRef.current = null;
          audioRef.current = null;
          setSpeechState("idle");
        };
        audio.onerror = () => {
          if (generationRef.current !== generation) return;
          setSpeechState("error");
          setAudit({ via: "groq", error: "Audio playback failed." });
        };
        await audio.play();
        if (generationRef.current !== generation) {
          audio.pause();
          return { via: "none" } as SpeechAudit;
        }
        setSpeechState("speaking");
        const result: SpeechAudit = { via: backendProvider === "groq" ? "groq" : "piper" };
        setAudit(result);
        return result;
      } catch (err) {
        if (generationRef.current !== generation) return { via: "none" } as SpeechAudit;
        const code = (err as { code?: string } | null)?.code;
        const result = speakWithBrowser(clean);
        if (result.via !== "none") setSpeechState("speaking");
        else setSpeechState("error");
        setAudit(result.via === "none"
          ? { via: "none", error: result.error ?? "Audio is unavailable.", code }
          : { via: "browser", code });
        return result;
      }
    },
    [prefs.ttsProvider, prefs.groqTtsVoice, speakWithBrowser, stop],
  );

  const updatePrefs = useCallback((next: Partial<TranslationPrefs>) => {
    setPrefs(savePrefs(next));
  }, []);

  const setAutoSpeak = useCallback((value: boolean) => {
    updatePrefs({ autoSpeak: value });
  }, [updatePrefs]);

  const setTtsProvider = useCallback((value: TtsProvider) => {
    updatePrefs({ ttsProvider: value });
  }, [updatePrefs]);

  const setGroqVoice = useCallback((voice: string) => {
    updatePrefs({ groqTtsVoice: voice });
  }, [updatePrefs]);

  const setBrowserVoiceURI = useCallback((uri: string | null) => {
    updatePrefs({ browserVoiceURI: uri });
  }, [updatePrefs]);

  const setSpeechRate = useCallback((rate: number) => {
    updatePrefs({ speechRate: rate });
  }, [updatePrefs]);

  return {
    prefs,
    speechState,
    audit,
    browserVoices,
    speaking: speechState === "speaking" || speechState === "loading",
    speak,
    stop,
    setAutoSpeak,
    setTtsProvider,
    setGroqVoice,
    setBrowserVoiceURI,
    setSpeechRate,
  };
}

export type SpeechController = ReturnType<typeof useSpeech>;
