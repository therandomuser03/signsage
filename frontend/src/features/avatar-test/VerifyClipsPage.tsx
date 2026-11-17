// SignSage — DEV-only clip-verification harness (/dev/verify-clips?gloss=HELLO&frac=0.5)
// Loads the GLB, applies the clip frame nearest `frac` of its duration, renders a
// single frame, and sets window.__verifyReady for the headless screenshot driver
// (tools/render_verification.mjs). Never built into production routes.

import { useEffect, useRef, useState } from "react";
import { HumanAvatarScene } from "./humanAvatarScene";

export function VerifyClipsPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [info, setInfo] = useState("loading");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const gloss = (params.get("gloss") ?? "hello").toLowerCase();
    const frac = Math.min(1, Math.max(0, Number(params.get("frac") ?? "0.5")));
    const canvas = canvasRef.current;
    if (!canvas) return;

    const scene = new HumanAvatarScene(canvas);
    (async () => {
      try {
        await scene.load("/models/signsage-human.glb");
        const res = await fetch(`/models/avatar/clips/${gloss}.json`);
        if (!res.ok) throw new Error(`clip ${gloss}.json: HTTP ${res.status}`);
        const clip = (await res.json()) as {
          frames: Array<Record<string, [number, number, number, number]>>;
        };
        const idx = Math.round(frac * (clip.frames.length - 1));
        scene.applyClip(clip.frames[idx]);
        setInfo(`${gloss} frame ${idx}/${clip.frames.length - 1}`);
        // let the render-on-demand loop draw once
        requestAnimationFrame(() => requestAnimationFrame(() => {
          (window as unknown as { __verifyReady?: boolean }).__verifyReady = true;
        }));
      } catch (err) {
        console.error(err);
        setInfo(`error: ${String(err)}`);
        (window as unknown as { __verifyReady?: boolean }).__verifyReady = true;
      }
    })();

    return () => scene.dispose();
  }, []);

  return (
    <div style={{ background: "#11161d", width: "100vw", height: "100vh", margin: 0 }}>
      <canvas id="verify-canvas" ref={canvasRef} style={{ width: 512, height: 640, display: "block" }} />
      <div data-testid="verify-info" style={{ color: "#fff", fontSize: 14, padding: 4 }}>{info}</div>
    </div>
  );
}
