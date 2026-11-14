// SignSage — procedural avatar (no VRM/GLB assets needed).
//
// Cartoon-like humanoid from Three.js primitives with visible hands, upper
// body and a simple face. Plays per-gloss motions from proceduralSigns.ts.
// Same AvatarViewerHandle contract as the VRM <AvatarViewer />: Play gains
// working 3D playback without changing callers.

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { motionFor, poseAt, type Pose, PROCEDURAL_SIGNS } from "../services/proceduralSigns";

export interface ProceduralAvatarHandle {
  /** Resolves with the gloss IDs whose animation assets were unavailable.
   * The procedural fallback has no external assets, so this is always []. */
  playSequence: (
    glossIds: string[],
    onStep?: (id: string, index: number) => void,
  ) => Promise<string[]>;
  stop: () => void;
}

interface Props {
  className?: string;
  caption?: string | null;
}

const SKIN = new THREE.Color("#e8b88a");
const SHIRT = new THREE.Color("#2f6df6");
const PANTS = new THREE.Color("#232a35");
const DARK = new THREE.Color("#1c1c1e");

/** Position a perspective camera so `box` fits fully with padding, then look at its center. */
function fitCameraToBox(camera: THREE.PerspectiveCamera, box: THREE.Box3, padding = 1.35) {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const fovRad = (camera.fov * Math.PI) / 180;
  const halfFov = Math.tan(fovRad / 2);
  const aspect = camera.aspect;

  // Distance required to cover the height at this FOV.
  let distance = size.y / (2 * halfFov);
  // Make sure the width fits too (visible width = distance * 2 * tan(fov/2) * aspect).
  if (distance * 2 * halfFov * aspect < size.x) {
    distance = size.x / (2 * halfFov * aspect);
  }
  distance *= padding;

  camera.position.set(center.x, center.y + size.y * 0.12, center.z + distance);
  camera.lookAt(center);
  camera.near = 0.1;
  camera.far = Math.max(20, distance * 4);
  camera.updateProjectionMatrix();
}

function limb(
  r: number,
  len: number,
  mat: THREE.Material,
): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 6, 12), mat);
  mesh.position.y = -len / 2;
  return mesh;
}

function hand(mat: THREE.Material): THREE.Group {
  const g = new THREE.Group();
  const palm = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.09, 0.035), mat);
  palm.position.y = -0.045;
  g.add(palm);
  // Four fingers + thumb as small capsules so the hand reads as a hand.
  for (let i = 0; i < 4; i++) {
    const f = new THREE.Mesh(new THREE.CapsuleGeometry(0.0085, 0.045, 4, 8), mat);
    f.position.set(-0.027 + i * 0.018, -0.115, 0);
    g.add(f);
  }
  const thumb = new THREE.Mesh(new THREE.CapsuleGeometry(0.0085, 0.035, 4, 8), mat);
  thumb.position.set(0.048, -0.045, 0.005);
  thumb.rotation.z = -0.7;
  g.add(thumb);
  return g;
}

interface Rig {
  shoulderL: THREE.Group;
  shoulderR: THREE.Group;
  elbowL: THREE.Group;
  elbowR: THREE.Group;
  head: THREE.Group;
  torso: THREE.Group;
  eyes: THREE.Mesh[];
}

function buildRig(): { root: THREE.Group; rig: Rig } {
  const skin = new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.7 });
  const shirt = new THREE.MeshStandardMaterial({ color: SHIRT, roughness: 0.8 });
  const pants = new THREE.MeshStandardMaterial({ color: PANTS, roughness: 0.85 });
  const dark = new THREE.MeshStandardMaterial({ color: DARK, roughness: 0.5 });

  const root = new THREE.Group();

  const hips = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.16, 0.2), pants);
  hips.position.y = 0.92;
  root.add(hips);

  // Static legs (upper body is the signing stage).
  for (const x of [-0.09, 0.09]) {
    const leg = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.55, 6, 12), pants);
    leg.position.set(x, 0.5, 0);
    root.add(leg);
    const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.07, 0.22), dark);
    shoe.position.set(x, 0.035, 0.04);
    root.add(shoe);
  }

  const torso = new THREE.Group();
  torso.position.y = 1.0;
  root.add(torso);
  const chest = new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.32, 8, 16), shirt);
  chest.position.y = 0.24;
  torso.add(chest);

  const head = new THREE.Group();
  head.position.y = 0.62;
  torso.add(head);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.115, 24, 18), skin);
  head.add(skull);
  const eyes: THREE.Mesh[] = [];
  for (const x of [-0.042, 0.042]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.014, 12, 10), dark);
    eye.position.set(x, 0.015, 0.102);
    head.add(eye);
    eyes.push(eye);
  }
  const smile = new THREE.Mesh(new THREE.TorusGeometry(0.032, 0.006, 8, 16, Math.PI * 0.9), dark);
  smile.position.set(0, -0.038, 0.1);
  smile.rotation.z = Math.PI + (Math.PI - Math.PI * 0.9) / 2;
  head.add(smile);
  // Simple hair cap.
  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.118, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.45),
    new THREE.MeshStandardMaterial({ color: "#3b2a20", roughness: 0.9 }),
  );
  hair.position.y = 0.012;
  head.add(hair);

  const mkArm = (side: 1 | -1) => {
    const shoulder = new THREE.Group();
    shoulder.position.set(0.21 * side, 0.42, 0);
    torso.add(shoulder);
    shoulder.add(limb(0.05, 0.2, shirt));
    const elbow = new THREE.Group();
    elbow.position.y = -0.28;
    shoulder.add(elbow);
    elbow.add(limb(0.042, 0.18, skin));
    const h = hand(skin);
    h.position.y = -0.28;
    elbow.add(h);
    return { shoulder, elbow };
  };
  const left = mkArm(-1);
  const right = mkArm(1);

  return {
    root,
    rig: {
      shoulderL: left.shoulder, shoulderR: right.shoulder,
      elbowL: left.elbow, elbowR: right.elbow,
      head, torso, eyes,
    },
  };
}

function applyPose(rig: Rig, p: Pose) {
  rig.shoulderL.rotation.set(p.shoulderL[0], p.shoulderL[1], p.shoulderL[2]);
  rig.shoulderR.rotation.set(p.shoulderR[0], p.shoulderR[1], p.shoulderR[2]);
  rig.elbowL.rotation.set(p.elbowL, 0, 0);
  rig.elbowR.rotation.set(p.elbowR, 0, 0);
  rig.head.rotation.set(p.head[0], p.head[1], p.head[2]);
  rig.torso.rotation.x = p.torso;
}

function restPose(): Pose {
  return {
    shoulderL: [0, 0, 0.12], shoulderR: [0, 0, -0.12],
    elbowL: -0.2, elbowR: -0.2, head: [0, 0, 0], torso: 0,
  };
}

export const ProceduralAvatarViewer = forwardRef<ProceduralAvatarHandle, Props>(
  ({ className, caption }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const runRef = useRef(0);
    const playRef = useRef<{
      queue: string[];
      index: number;
      startedAt: number;
      onStep?: (id: string, index: number) => void;
      resolve: () => void;
    } | null>(null);
    const rigRef = useRef<Rig | null>(null);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 20);
      camera.position.set(0, 1.32, 2.5);
      camera.lookAt(0, 1.02, 0);

      scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 0.9));
      const key = new THREE.DirectionalLight(0xffffff, 1.4);
      key.position.set(1.2, 2.2, 1.6);
      scene.add(key);
      const ground = new THREE.Mesh(
        new THREE.CircleGeometry(0.9, 40),
        new THREE.MeshStandardMaterial({ color: "#e5e7eb", roughness: 1 }),
      );
      ground.rotation.x = -Math.PI / 2;
      scene.add(ground);

      const { root, rig } = buildRig();
      rigRef.current = rig;
      scene.add(root);
      applyPose(rig, restPose());

      // Frame from bounds sampled across rest and every key pose so animated
      // arms/head stay in frame, not just the rest pose.
      //
      // Signing happens in the upper body (torso, head, arms, hands). Framing
      // the full figure (hips down to feet) made the signing-relevant parts
      // appear small/compact, so we frame the upper body only: torso + every
      // descendant (head, shoulders, arms, hands). Legs and shoes are excluded
      // from the framing box so hands and face fill the viewport instead of
      // sharing it with the legs. The rig's `torso` group carries all of the
      // upper body, so expanding by it (not `root`) keeps signing visible and
      // readably large without clipping raised arms.
      const box = new THREE.Box3();
      const samplePoses: Pose[] = [restPose()];
      for (const motion of Object.values(PROCEDURAL_SIGNS)) {
        for (const key of motion.keys) {
          samplePoses.push(poseAt(motion, key.at));
        }
      }
      for (const p of samplePoses) {
        applyPose(rig, p);
        box.expandByObject(rig.torso);
      }
      applyPose(rig, restPose());
      fitCameraToBox(camera, box, 1.12);

      const resize = () => {
        const w = canvas.clientWidth || 640;
        const h = canvas.clientHeight || 360;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        fitCameraToBox(camera, box, 1.12);
        camera.updateProjectionMatrix();
      };
      resize();
      window.addEventListener("resize", resize);

      let raf = 0;
      let lastBlink = performance.now();
      const tick = () => {
        raf = requestAnimationFrame(tick);
        const now = performance.now();
        // Blink every ~3.2s.
        if (now - lastBlink > 3200) {
          lastBlink = now;
          for (const e of rig.eyes) e.scale.y = 0.1;
          setTimeout(() => rig.eyes.forEach((e) => (e.scale.y = 1)), 140);
        }
        const play = playRef.current;
        if (play) {
          const motion = motionFor(play.queue[play.index]);
          const t = Math.min(1, (now - play.startedAt) / motion.durationMs);
          applyPose(rig, poseAt(motion, t));
          if (t >= 1) {
            play.index += 1;
            if (play.index >= play.queue.length) {
              playRef.current = null;
              applyPose(rig, restPose());
              play.resolve();
            } else {
              play.startedAt = now;
              play.onStep?.(play.queue[play.index], play.index);
            }
          }
        }
        renderer.render(scene, camera);
      };
      tick();

      return () => {
        cancelAnimationFrame(raf);
        window.removeEventListener("resize", resize);
        renderer.dispose();
        scene.traverse((o) => {
          const m = o as THREE.Mesh;
          if (m.isMesh) {
            m.geometry.dispose();
            const mat = m.material as THREE.Material | THREE.Material[];
            (Array.isArray(mat) ? mat : [mat]).forEach((x) => x.dispose());
          }
        });
      };
    }, []);

    useImperativeHandle(ref, () => ({
      playSequence: (glossIds, onStep) => {
        const run = ++runRef.current;
        playRef.current?.resolve();
        if (glossIds.length === 0) return Promise.resolve([]);
        return new Promise<string[]>((resolve) => {
          onStep?.(glossIds[0], 0);
          playRef.current = {
            queue: glossIds, index: 0, startedAt: performance.now(),
            onStep: (id, i) => {
              if (run === runRef.current) onStep?.(id, i);
            },
            resolve: () => {
              if (run === runRef.current) playRef.current = null;
              resolve([]);
            },
          };
        });
      },
      stop: () => {
        runRef.current += 1;
        playRef.current?.resolve();
        playRef.current = null;
        if (rigRef.current) applyPose(rigRef.current, restPose());
      },
    }));

    return (
      <div className={className} style={{ position: "relative", width: "100%", height: "100%" }}>
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
        {caption ? (
          <div
            style={{
              position: "absolute", left: 8, bottom: 8, fontSize: 12,
              background: "rgba(0,0,0,0.55)", color: "#fff",
              padding: "2px 10px", borderRadius: 999,
            }}
          >
            {caption}
          </div>
        ) : null}
      </div>
    );
  },
);
ProceduralAvatarViewer.displayName = "ProceduralAvatarViewer";
