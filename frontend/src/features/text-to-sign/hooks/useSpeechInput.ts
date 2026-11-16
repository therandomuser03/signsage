// SignSage — microphone input via the browser's built-in speech recognition.
// Uses window.SpeechRecognition when the browser provides it and reports
// `supported: false` otherwise (Firefox, some mobile browsers) so the UI can
// switch to the server transcription fallback honestly.
//
// Flow: getUserMedia FIRST (permission + a live AnalyserNode level meter),
// then recognition. Defaults to en-IN; en-US is selectable in the UI. Only
// final transcripts are committed to the input field; interim results are
// exposed separately as a live preview so the same text is never inserted twice.

import { useCallback, useEffect, useRef, useState } from "react";
import { createLevelMeter, micErrorMessage, playStartBeep, type LevelMeter } from "./audioLevel";

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
      return "Microphone blocked. Allow mic access (lock icon → Site settings → Microphone → Allow), reload, and try again.";
    case "audio-capture":
      return "No microphone found. Connect a microphone and try again.";
    case "no-speech":
      return "Nothing heard. Speak after the beep and check the level bar.";
    case "network":
      return "This browser blocks the speech service. Switching to server transcription.";
    case "service-not-allowed":
      return "This browser blocks the speech service. Switching to server transcription.";
    case "language-not-supported":
      return "This browser does not support that language. Switch to en-US or use server transcription.";
    case "no-match":
      return "Didn't catch that. Speak clearly and try again.";
    default:
      return "Speech recognition failed. Try again or type instead.";
  }
}

/** True for the two browser errors that mean "this speech service is unusable":
 *  the UI switches to the server transcription fallback when it sees these. */
export function isSpeechServiceBlocked(error: string): boolean {
  return error === "network" || error === "service-not-allowed";
}

const LEVEL_POLL_MS = 100;

export function useSpeechInput({ lang = "en-IN" }: { lang?: string } = {}) {
  const [supported] = useState(() => createRecognizer(lang) !== null);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Raw SpeechRecognition error code (network, service-not-allowed, ...)
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const recRef = useRef<Rec | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const meterRef = useRef<LevelMeter | null>(null);
  const levelTimerRef = useRef<number | null>(null);
  // Last committed final transcript: guards against the service redelivering
  // the same final result and inserting it twice.
  const lastFinalRef = useRef("");
  // True once the current session has produced a final transcript.
  const gotFinalRef = useRef(false);
  // True when the user explicitly pressed stop (so onend doesn't report an
  // empty transcript for an intentional cancellation).
  const userStoppedRef = useRef(false);

  const releaseMic = useCallback(() => {
    if (levelTimerRef.current !== null) {
      window.clearInterval(levelTimerRef.current);
      levelTimerRef.current = null;
    }
    meterRef.current?.dispose();
    meterRef.current = null;
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) track.stop();
      streamRef.current = null;
    }
    setLevel(0);
  }, []);

  const attachMeter = useCallback((stream: MediaStream) => {
    meterRef.current = createLevelMeter(stream);
    levelTimerRef.current = window.setInterval(() => {
      if (meterRef.current) setLevel(meterRef.current.level());
    }, LEVEL_POLL_MS);
  }, []);

  useEffect(
    () => () => {
      recRef.current?.abort();
      releaseMic();
    },
    [releaseMic],
  );

  const start = useCallback(
    async (onFinalTranscript: (text: string) => void, onInterimTranscript?: (text: string) => void) => {
      if (listening) return;
      setError(null);
      setErrorCode(null);

      // 1. Microphone permission + level meter first.
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        setError(micErrorMessage(err));
        return;
      }

      // 2. Then the recognizer (may not exist on this browser).
      const rec = createRecognizer(lang);
      if (!rec) {
        for (const track of stream.getTracks()) track.stop();
        setError("Speech recognition is not supported in this browser. Type instead.");
        return;
      }
      streamRef.current = stream;
      attachMeter(stream);
      playStartBeep();

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
        setErrorCode(ev.error);
        setListening(false);
        recRef.current = null;
        releaseMic();
      };
      rec.onend = () => {
        setListening(false);
        recRef.current = null;
        releaseMic();
        if (!userStoppedRef.current && !gotFinalRef.current) {
          setError("Nothing heard. Speak after the beep and check the level bar.");
        }
        userStoppedRef.current = false;
      };
      try {
        rec.start();
        setListening(true);
      } catch {
        setError("Speech recognition could not start. Type instead.");
        releaseMic();
      }
    },
    [listening, lang, attachMeter, releaseMic],
  );

  const stop = useCallback(() => {
    userStoppedRef.current = true;
    recRef.current?.stop();
  }, []);

  const clearError = useCallback(() => {
    setError(null);
    setErrorCode(null);
  }, []);

  return { supported, listening, error, errorCode, level, start, stop, clearError };
}
