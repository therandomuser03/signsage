// SignSage — microphone input via the browser's built-in speech recognition.
// No dependency, no backend call: uses window.SpeechRecognition when the
// browser provides it and reports `supported: false` otherwise (Firefox,
// some mobile browsers) so the UI can hide/disable honestly.
//
// Defaults to English (en-IN) but the locale is configurable. Only final
// transcripts are committed to the input field; interim results are exposed
// separately as a live preview so the same text is never inserted twice.

import { useCallback, useEffect, useRef, useState } from "react";

export type Rec = {
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((ev: { results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> }) => void) | null;
  onerror: ((ev: { error: string }) => void) | null;
  onend: (() => void) | null;
  continuous: boolean;
  interimResults: boolean;
  lang: string;
};

export function createRecognizer(lang: string): Rec | null {
  const w = window as unknown as Record<string, (new () => Rec) | undefined>;
  const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
  if (!Ctor) return null;
  const rec = new Ctor();
  rec.continuous = false;
  rec.interimResults = true;
  rec.lang = lang;
  return rec;
}

export function speechErrorMessage(error: string): string {
  switch (error) {
    case "not-allowed":
      return "Microphone blocked. Allow access and try again.";
    case "network":
      return "Network failure. Check your connection and try again.";
    case "audio-capture":
      return "No microphone found. Connect a microphone and try again.";
    case "service-not-allowed":
      return "Speech service not allowed. Type instead.";
    case "no-match":
      return "No speech recognized. Try again.";
    default:
      return "Speech recognition failed. Type instead.";
  }
}

export function useSpeechInput({ lang = "en-IN" }: { lang?: string } = {}) {
  const [supported] = useState(() => createRecognizer(lang) !== null);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<Rec | null>(null);
  // Last committed final transcript: guards against the service redelivering
  // the same final result and inserting it twice.
  const lastFinalRef = useRef("");
  // True once the current session has produced a final transcript.
  const gotFinalRef = useRef(false);
  // True when the user explicitly pressed stop (so onend doesn't report an
  // empty transcript for an intentional cancellation).
  const userStoppedRef = useRef(false);

  useEffect(() => () => recRef.current?.abort(), []);

  const start = useCallback(
    (onFinalTranscript: (text: string) => void, onInterimTranscript?: (text: string) => void) => {
      if (listening) return;
      const rec = createRecognizer(lang);
      if (!rec) {
        setError("Speech recognition is not supported in this browser. Type instead.");
        return;
      }
      setError(null);
      gotFinalRef.current = false;
      lastFinalRef.current = "";
      userStoppedRef.current = false;
      recRef.current = rec;
      rec.onresult = (ev) => {
        let interim = "";
        let final = "";
        for (const r of Array.from(ev.results)) {
          const t = r[0].transcript;
          if (r.isFinal) final += t;
          else interim += t;
        }
        if (final.trim()) {
          gotFinalRef.current = true;
          if (final !== lastFinalRef.current) {
            lastFinalRef.current = final;
            onFinalTranscript(final);
          }
        } else if (interim.trim() && onInterimTranscript) {
          onInterimTranscript(interim);
        }
      };
      rec.onerror = (ev) => {
        setError(speechErrorMessage(ev.error));
        setListening(false);
        recRef.current = null;
      };
      rec.onend = () => {
        setListening(false);
        recRef.current = null;
        if (!userStoppedRef.current && !gotFinalRef.current) {
          setError("No speech recognized. Try again.");
        }
        userStoppedRef.current = false;
      };
      try {
        rec.start();
        setListening(true);
      } catch {
        setError("Speech recognition could not start. Type instead.");
      }
    },
    [listening, lang],
  );

  const stop = useCallback(() => {
    userStoppedRef.current = true;
    recRef.current?.stop();
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return { supported, listening, error, start, stop, clearError };
}
