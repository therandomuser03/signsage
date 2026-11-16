// SignSage — <HumanAvatarViewer />
// Renders the realistic human GLB (public/models/signsage-human.glb) inside the
// Text-to-Sign experience via the SHARED HumanAvatarScene manager (task-02).
// No second Three.js architecture is introduced.
//
// IMPORTANT — honest playback: this export has NO animation clips, and no ISL
// motion has been authored or verified for this rig. playSequence() therefore
// performs a GENERIC, clearly-labelled articulation demonstration per gloss
// (verified poses only: arm raise, finger open/close, one blink, a jaw pulse).
// It is NOT word-specific signing, the UI says so (caption + disclaimer), and
// no translation quality is implied. Translation output (gloss order, unmatched
// words, meaning) is produced by the existing mapper/backend, unchanged.
//
// Same AvatarViewerHandle contract as the VRM/procedural viewers, so
// useSignAnimation drives it unmodified.

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { HumanAvatarScene } from "@/features/avatar-test/humanAvatarScene";
import type { AvatarViewerHandle } from "./AvatarViewer";

const GLB_URL = "/models/signsage-human.glb";
const BEAT_SECONDS = 1.5;

/** Smoothstep easing between adjacent keyframe [time, value] pairs. */
function sampleKeyframes(kfs: ReadonlyArray<readonly [number, number]>, t: number): number {
  if (t <= kfs[0][0]) return kfs[0][1];
  for (let i = 0; i < kfs.length - 1; i++) {
    const [t0, v0] = kfs[i];
    const [t1, v1] = kfs[i + 1];
    if (t <= t1) {
      const u = (t - t0) / (t1 - t0);
      const s = u * u * (3 - 2 * u);
      return v0 + (v1 - v0) * s;
    }
  }
  return kfs[kfs.length - 1][1];
}

/** Generic articulation beat: raise arms → fingers open/close → blink → jaw
 * pulse → ease back to rest. Identical for every gloss on purpose — it is an
 * articulation demo, not a word-specific sign. */
const KF_RAISE = [[0, 0], [0.35, 0.55], [1.15, 0.55], [BEAT_SECONDS, 0]] as const;
const KF_CURL = [[0, 0], [0.4, 0], [0.6, 0.85], [0.85, 0.15], [1.05, 0], [BEAT_SECONDS, 0]] as const;
const KF_BLINK = [[0, 0], [0.52, 0], [0.6, 1], [0.7, 0], [BEAT_SECONDS, 0]] as const;
const KF_JAW = [[0, 0], [0.3, 0], [0.5, 0.32], [0.9, 0], [BEAT_SECONDS, 0]] as const;

interface RunningBeat {
  raf: number | null;
  startedAt: number;
  onDone: () => void;
}

interface Props {
  className?: string;
  caption?: string | null;
}

export const HumanAvatarViewer = forwardRef<AvatarViewerHandle, Props>(
  ({ className, caption }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sceneRef = useRef<HumanAvatarScene | null>(null);
    const readyRef = useRef<Promise<boolean>>(Promise.resolve(false));
    const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
    const runRef = useRef(0);
    const beatRef = useRef<RunningBeat | null>(null);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const scene = new HumanAvatarScene(canvas);
      sceneRef.current = scene;
      readyRef.current = scene.load(GLB_URL).then(
        () => {
          setStatus("ready");
          return true;
        },
        (err: unknown) => {
          console.error("SignSage human model failed to load:", err);
          setStatus("error");
          return false;
        },
      );
      return () => {
        runRef.current += 1;
        if (beatRef.current?.raf != null) cancelAnimationFrame(beatRef.current.raf);
        beatRef.current = null;
        scene.dispose();
        sceneRef.current = null;
        // Dropped WebGL context + lose context listener are handled by dispose().
      };
    }, []);

    /** Run one generic articulation beat. Resolves early on stop()/unmount. */
    const animateBeat = (run: number): Promise<void> =>
      new Promise<void>((resolve) => {
        const scene = sceneRef.current;
        if (!scene || run !== runRef.current) return resolve();
        const startedAt = performance.now();
        const tick = () => {
          if (run !== runRef.current || !sceneRef.current) {
            return resolve();
          }
          const t = (performance.now() - startedAt) / 1000;
          if (t >= BEAT_SECONDS) {
            sceneRef.current?.resetPose();
            sceneRef.current?.resetFace();
            sceneRef.current?.invalidate();
            beatRef.current = null;
            return resolve();
          }
          sceneRef.current.setArmPose(sampleKeyframes(KF_RAISE, t), sampleKeyframes(KF_RAISE, t));
          sceneRef.current.setAllFingerCurls("both", sampleKeyframes(KF_CURL, t));
          const blink = sampleKeyframes(KF_BLINK, t);
          sceneRef.current.setMorphTarget("eyeBlinkLeft", blink);
          sceneRef.current.setMorphTarget("eyeBlinkRight", blink);
          sceneRef.current.setMorphTarget("jawOpen", sampleKeyframes(KF_JAW, t));
          beatRef.current = { raf: requestAnimationFrame(tick), startedAt, onDone: resolve };
        };
        beatRef.current = { raf: requestAnimationFrame(tick), startedAt, onDone: resolve };
      });

    useImperativeHandle(ref, () => ({
      playSequence: async (glossIds: string[], onStep?: (id: string, index: number) => void) => {
        const run = ++runRef.current;
        if (glossIds.length === 0) return [];
        const ready = await readyRef.current;
        if (!ready || run !== runRef.current) return [];
        for (let index = 0; index < glossIds.length; index++) {
          if (run !== runRef.current) return [];
          onStep?.(glossIds[index], index);
          await animateBeat(run);
        }
        return []; // no animation assets exist; the caption labels this a demo
      },
      stop: () => {
        runRef.current += 1;
        if (beatRef.current?.raf != null) cancelAnimationFrame(beatRef.current.raf);
        beatRef.current = null;
        sceneRef.current?.resetPose();
        sceneRef.current?.resetFace();
      },
    }));

    return (
      <div className={className} style={{ position: "relative", width: "100%", height: "100%" }}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", display: "block" }}
          aria-label="Realistic human avatar demonstrating articulation (not a validated ISL sign)"
        />
        {status === "loading" && (
          <div style={overlayStyle}>Loading human model…</div>
        )}
        {status === "error" && (
          <div style={overlayStyle}>Couldn't load the human model. Check connection.</div>
        )}
        {caption && status === "ready" ? (
          <div
            style={{
              position: "absolute", left: 8, bottom: 8, fontSize: 12,
              background: "rgba(0,0,0,0.55)", color: "#fff",
              padding: "2px 10px", borderRadius: 999, maxWidth: "90%",
            }}
          >
            {caption}
          </div>
        ) : null}
      </div>
    );
  },
);
HumanAvatarViewer.displayName = "HumanAvatarViewer";

const overlayStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 14,
  color: "var(--muted-foreground, #888)",
  pointerEvents: "none",
};
