// SignSage — shared audio-level metering for the microphone paths.
// Computes an RMS level (0..1) from a MediaStream via an AnalyserNode and
// plays a short start beep so the user knows recording began.

export interface LevelMeter {
  level: () => number;
  dispose: () => void;
}

export function createLevelMeter(stream: MediaStream): LevelMeter | null {
  try {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    const ctx = new Ctor();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.7;
    source.connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    return {
      level: () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i += 1) {
          const centered = buf[i] - 128;
          sum += centered * centered;
        }
        const rms = Math.sqrt(sum / buf.length) / 128; // 0..1
        return Math.min(1, rms * 1.6);
      },
      dispose: () => {
        try {
          source.disconnect();
          analyser.disconnect();
        } catch {
          /* already torn down */
        }
        void ctx.close().catch(() => undefined);
      },
    };
  } catch {
    return null;
  }
}

export function playStartBeep(): void {
  try {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    const ctx = new Ctor();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.05, ctx.currentTime);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.15);
    osc.onended = () => void ctx.close().catch(() => undefined);
  } catch {
    /* beep is a nicety, never block on it */
  }
}

/** Map a getUserMedia rejection to a user-facing message. */
export function micErrorMessage(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Microphone blocked. Allow mic access (lock icon → Site settings → Microphone → Allow), then try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
    case "OverconstrainedError":
      return "No microphone found. Connect a microphone and try again.";
    case "NotReadableError":
    case "TrackStartError":
      return "The microphone is in use by another app. Close it and try again.";
    default:
      return "Could not start the microphone. Check the browser's mic permission, then try again.";
  }
}
