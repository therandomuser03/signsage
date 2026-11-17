// SignSage — <HumanAvatarViewer />
// Renders the realistic human GLB (public/models/signsage-human.glb) and plays
// per-gloss animation clips retargeted from recorded MediaPipe landmarks (or
// explicitly hand-authored approximations for signs with no recorded data).
//
// HONESTY: none of these motions have been validated by a fluent ISL signer.
// The caption keeps saying "not a validated ISL sign". Glosses without a clip
// are skipped with a clear "no clip yet" report — never replaced with the old
// generic articulation demo.
//
// Clip format (public/models/avatar/clips/<gloss>.json):
//   { fps, duration, frames: [ { <boneName>: [x,y,z,w], ... }, ... ], meta }
// Frames contain absolute local quaternions; ease in/out from rest is baked in.

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { HumanAvatarScene } from "@/features/avatar-test/humanAvatarScene";
import type { SignManifest } from "../types";
import type { AvatarViewerHandle } from "./AvatarViewer";

const GLB_URL = "/models/signsage-human.glb";
const MISSING_GLOSS_PAUSE_MS = 350;

interface ClipBoneFrames {
  /** bone name -> per-frame local quaternion */
  bones: Map<string, THREE.Quaternion[]>;
  fps: number;
  duration: number;
  authored: boolean;
}

interface Props {
  className?: string;
  caption?: string | null;
  /** Sign manifest (from /models/avatar/manifest.json). When null, every gloss
   *  reports "no clip yet" instead of playing anything. */
  manifest?: SignManifest | null;
}

export const HumanAvatarViewer = forwardRef<AvatarViewerHandle, Props>(
  ({ className, caption, manifest }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sceneRef = useRef<HumanAvatarScene | null>(null);
    const readyRef = useRef<Promise<boolean>>(Promise.resolve(false));
    const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
    const runRef = useRef(0);
    const rafRef = useRef<number | null>(null);
    const clipCache = useRef(new Map<string, ClipBoneFrames | null>());
   
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
        if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        scene.dispose();
        sceneRef.current = null;
      };
    }, []);

    /** Fetch + parse a clip JSON once; null when missing/invalid. */
    const loadClip = async (glossId: string): Promise<ClipBoneFrames | null> => {
      const cacheKey = glossId;
      const cached = clipCache.current.get(cacheKey);
      if (cached !== undefined) return cached;
      const entry = manifest?.[glossId];
      const url = entry?.clip;
      if (!url) {
        clipCache.current.set(cacheKey, null);
        return null;
      }
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`clip ${url}: HTTP ${res.status}`);
        const data = (await res.json()) as {
          fps: number;
          duration: number;
          frames: Array<Record<string, [number, number, number, number]>>;
          meta?: { authored?: boolean };
        };
        const bones = new Map<string, THREE.Quaternion[]>();
        for (const frame of data.frames) {
          for (const [bone, q] of Object.entries(frame)) {
            let arr = bones.get(bone);
            if (!arr) {
              arr = [];
              bones.set(bone, arr);
            }
            arr.push(new THREE.Quaternion(q[0], q[1], q[2], q[3]));
          }
        }
        const parsed: ClipBoneFrames = {
          bones,
          fps: data.fps,
          duration: data.duration,
          authored: Boolean(data.meta?.authored),
        };
        clipCache.current.set(cacheKey, parsed);
        return parsed;
      } catch (err) {
        console.warn(`SignSage: clip for ${glossId} unavailable:`, err);
        clipCache.current.set(cacheKey, null);
        return null;
      }
    };

    // Preload every clip in the manifest once the avatar is up, so playback
    // starts instantly.
    useEffect(() => {
      if (!manifest) return;
      let live = true;
      void readyRef.current.then((ok) => {
        if (!ok || !live) return;
        for (const id of Object.keys(manifest)) {
          if (manifest[id]?.clip) void loadClip(id);
        }
      });
      return () => {
        live = false;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [manifest]);

    /** Play one clip. Resolves early on stop()/unmount. */
    const playClip = (clip: ClipBoneFrames, run: number): Promise<void> =>
      new Promise<void>((resolve) => {
        const startedAt = performance.now();
        const frameCount = clip.bones.values().next().value?.length ?? 0;
        const tick = () => {
          const scene = sceneRef.current;
          if (!scene || run !== runRef.current) return resolve();
          const t = (performance.now() - startedAt) / 1000;
          if (t >= clip.duration) {
            scene.resetPose();
            scene.invalidate();
            rafRef.current = null;
            return resolve();
          }
          const pos = Math.min(t * clip.fps, Math.max(frameCount - 1, 0));
          const i0 = Math.floor(pos);
          const i1 = Math.min(i0 + 1, frameCount - 1);
          const s = pos - i0;
          const frame: Record<string, [number, number, number, number]> = {};
          const tmp = new THREE.Quaternion();
          for (const [bone, seq] of clip.bones) {
            tmp.copy(seq[i0]);
            if (i1 !== i0) tmp.slerp(seq[i1], s);
            frame[bone] = [tmp.x, tmp.y, tmp.z, tmp.w];
          }
          scene.applyClip(frame);
          rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
      });

    useImperativeHandle(ref, () => ({
      playSequence: async (glossIds: string[], onStep?: (id: string, index: number) => void) => {
        const run = ++runRef.current;
        if (glossIds.length === 0) return [];
        const ready = await readyRef.current;
        if (!ready || run !== runRef.current) return glossIds.slice();
        const unavailable: string[] = [];
        for (let index = 0; index < glossIds.length; index++) {
          if (run !== runRef.current) return unavailable;
          const glossId = glossIds[index];
          onStep?.(glossId, index);
          const clip = await loadClip(glossId);
          if (!clip) {
            unavailable.push(glossId);
            // brief rest beat so the sequence's progress UI still advances
            await new Promise<void>((resolve) => {
              const started = performance.now();
              const tick = () => {
                if (run !== runRef.current || performance.now() - started >= MISSING_GLOSS_PAUSE_MS) {
                  return resolve();
                }
                rafRef.current = requestAnimationFrame(tick);
              };
              rafRef.current = requestAnimationFrame(tick);
            });
            continue;
          }
          await playClip(clip, run);
        }
        return unavailable;
      },
      stop: () => {
        runRef.current += 1;
        if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        sceneRef.current?.resetPose();
        sceneRef.current?.resetFace();
      },
    }));

    return (
      <div className={className} style={{ position: "relative", width: "100%", height: "100%" }}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", display: "block" }}
          aria-label="Realistic human avatar performing sign motion retargeted from recorded data (not a validated ISL sign)"
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
