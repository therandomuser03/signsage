// SignSage — <AvatarViewer />
// Mounts the Three.js canvas and owns the AvatarController lifecycle.
// Parent components drive playback through the forwarded ref.

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { AvatarController } from "../services/avatarController";
import type { SignManifest } from "../types";

export interface AvatarViewerHandle {
  playSequence: (glossIds: string[], onStep?: (id: string, index: number) => void) => Promise<void>;
  stop: () => void;
}

interface AvatarViewerProps {
  vrmUrl: string;
  manifest: SignManifest;
  className?: string;
}

export const AvatarViewer = forwardRef<AvatarViewerHandle, AvatarViewerProps>(
  ({ vrmUrl, manifest, className }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const controllerRef = useRef<AvatarController | null>(null);
    const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

    useEffect(() => {
      if (!canvasRef.current) return;
      const controller = new AvatarController(canvasRef.current, manifest);
      controllerRef.current = controller;

      controller
        .loadAvatar(vrmUrl)
        .then(() => setStatus("ready"))
        .catch((err) => {
          console.error("SignSage avatar failed to load:", err);
          setStatus("error");
        });

      return () => controller.dispose();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [vrmUrl]);

    useImperativeHandle(ref, () => ({
      playSequence: (glossIds, onStep) => {
        if (!controllerRef.current) return Promise.resolve();
        return controllerRef.current.playSequence(glossIds, onStep);
      },
      stop: () => controllerRef.current?.stop(),
    }));

    return (
      <div className={className} style={{ position: "relative", width: "100%", height: "100%" }}>
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
        {status === "loading" && (
          <div style={overlayStyle}>Loading avatar…</div>
        )}
        {status === "error" && (
          <div style={overlayStyle}>Couldn't load the avatar. Check vrmUrl.</div>
        )}
      </div>
    );
  }
);
AvatarViewer.displayName = "AvatarViewer";

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
