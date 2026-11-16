// SignSage — server transcription fallback for the Sign Generator microphone.
// Used when the browser has no SpeechRecognition, or it fails with
// network / service-not-allowed. Records a short clip with MediaRecorder
// (max 8 s, visible Stop button + countdown) and POSTs it to
// /api/v1/speech/transcribe, which forwards it to Groq Whisper.
// The transcript is returned to the caller — it is NOT auto-submitted.

import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "../../../services/api/instance";
import { createLevelMeter, micErrorMessage, playStartBeep, type LevelMeter } from "./audioLevel";

export type ServerDictationState = "idle" | "requesting" | "recording" | "uploading" | "error";

export const MAX_DICTATION_SECONDS = 8;
const LEVEL_POLL_MS = 100;

function pickRecorderMimeType(): string | null {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return null;
  }
  for (const t of ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg", "audio/mp4"]) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return null;
}

function extensionForMime(mime: string): string {
  const base = mime.split(";")[0].trim().toLowerCase();
  if (base.endsWith("ogg")) return "ogg";
  if (base.endsWith("mp4")) return "mp4";
  return "webm";
}

export function useServerDictation({ maxSeconds = MAX_DICTATION_SECONDS }: { maxSeconds?: number } = {}) {
  const [state, setState] = useState<ServerDictationState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [secondsLeft, setSecondsLeft] = useState(maxSeconds);
  const [level, setLevel] = useState(0);
  const supported =
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined";

  const streamRef = useRef<MediaStream | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const countdownRef = useRef<number | null>(null);
  const levelTimerRef = useRef<number | null>(null);
  const meterRef = useRef<LevelMeter | null>(null);
  const onTextRef = useRef<((text: string) => void) | null>(null);
  const mimeRef = useRef<string>("audio/webm");

  const releaseMic = useCallback(() => {
    if (countdownRef.current !== null) {
      window.clearInterval(countdownRef.current);
      countdownRef.current = null;
    }
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

  const stop = useCallback(() => {
    const rec = recRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
  }, []);

  const start = useCallback(
    async (onText: (text: string) => void) => {
      if (state !== "idle" && state !== "error") return;
      if (!supported) {
        setError("This browser cannot record audio. Type instead.");
        setState("error");
        return;
      }
      onTextRef.current = onText;
      setError(null);
      setState("requesting");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        const mimeType = pickRecorderMimeType();
        mimeRef.current = mimeType ?? "audio/webm";
        const chunks: Blob[] = [];
        chunksRef.current = chunks;
        const rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        recRef.current = rec;
        rec.ondataavailable = (e: BlobEvent) => {
          if (e.data && e.data.size > 0) chunks.push(e.data);
        };
        rec.onstop = () => {
          void (async () => {
            setState("uploading");
            if (countdownRef.current !== null) {
              window.clearInterval(countdownRef.current);
              countdownRef.current = null;
            }
            try {
              const blob = new Blob(chunks, { type: mimeRef.current.split(";")[0] });
              const ext = extensionForMime(mimeRef.current);
              const res = await apiClient.transcribe(blob, `clip.${ext}`);
              const text = res.text?.trim();
              if (text) {
                onTextRef.current?.(text);
                setState("idle");
              } else {
                setError("Transcription returned no text. Try again.");
                setState("error");
              }
            } catch (err) {
              const code = (err as { code?: string } | null)?.code;
              setError(
                code
                  ? `Server transcription failed (${code}). Try again or type instead.`
                  : "Server transcription failed. Type instead.",
              );
              setState("error");
            } finally {
              releaseMic();
            }
          })();
        };
        rec.start();
        playStartBeep();
        meterRef.current = createLevelMeter(stream);
        levelTimerRef.current = window.setInterval(() => {
          if (meterRef.current) setLevel(meterRef.current.level());
        }, LEVEL_POLL_MS);
        setState("recording");
        setSecondsLeft(maxSeconds);
        let left = maxSeconds;
        countdownRef.current = window.setInterval(() => {
          left -= 1;
          setSecondsLeft(Math.max(0, left));
          if (left <= 0) stop();
        }, 1000);
      } catch (err) {
        releaseMic();
        setError(micErrorMessage(err));
        setState("error");
      }
    },
    [state, supported, maxSeconds, releaseMic, stop],
  );

  useEffect(
    () => () => {
      const rec = recRef.current;
      if (rec && rec.state !== "inactive") rec.stop();
      releaseMic();
    },
    [releaseMic],
  );

  return {
    supported,
    state,
    error,
    secondsLeft,
    level,
    recording: state === "recording",
    uploading: state === "uploading",
    start,
    stop,
  };
}
