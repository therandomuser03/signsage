// SignSage — microphone input via the browser's built-in speech recognition.
// No dependency, no backend call: uses window.SpeechRecognition when the
// browser provides it and reports `supported: false` otherwise (Firefox,
// some mobile browsers) so the UI can hide/disable honestly.

import { useCallback, useEffect, useRef, useState } from "react";

type Rec = {
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

function createRecognizer(): Rec | null {
  const w = window as unknown as Record<string, (new () => Rec) | undefined>;
  const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
  if (!Ctor) return null;
  const rec = new Ctor();
  rec.continuous = false;
  rec.interimResults = true;
  rec.lang = "en-US";
  return rec;
}

export function useSpeechInput() {
  const [supported] = useState(() => createRecognizer() !== null);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<Rec | null>(null);

  useEffect(() => () => recRef.current?.abort(), []);

  const start = useCallback(
    (onTranscript: (text: string) => void) => {
      if (listening) return;
      const rec = createRecognizer();
      if (!rec) {
        setError("Speech recognition is not supported in this browser. Type instead.");
        return;
      }
      setError(null);
      recRef.current = rec;
      rec.onresult = (ev) => {
        let text = "";
        for (const r of Array.from(ev.results)) text += r[0].transcript;
        if (text.trim()) onTranscript(text);
      };
      rec.onerror = (ev) => {
        setError(
          ev.error === "not-allowed"
            ? "Microphone blocked. Allow access and try again."
            : "Speech recognition failed. Type instead.",
        );
        setListening(false);
      };
      rec.onend = () => {
        setListening(false);
        recRef.current = null;
      };
      try {
        rec.start();
        setListening(true);
      } catch {
        setError("Speech recognition could not start. Type instead.");
      }
    },
    [listening],
  );

  const stop = useCallback(() => recRef.current?.stop(), []);

  return { supported, listening, error, start, stop };
}
