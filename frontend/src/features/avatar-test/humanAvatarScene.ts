// SignSage — isolated avatar test scene (dev-only validation page).
//
// Vanilla three.js, matching the existing avatarController.ts architecture.
// Loads the real signsage-human.glb (plain GLB, NOT a VRM — no VRM plugin).
//
// Performance for Intel integrated graphics:
//   - no shadow maps, no post-processing, capped pixel ratio
//   - render-on-demand only (controls "change" or explicit invalidate);
//     the rAF loop contains no per-frame work besides checking the dirty flag
//   - full dispose of geometries / materials / textures

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FINGERS, sanitizeBoneName, type FingerId, type HandSide } from "./rigMap";

export interface SceneStats {
  boneCount: number;
  skinnedMeshJoints: number;
  meshCount: number;
  triangleCount: number;
  drawCalls: number;
  pixelRatio: number;
  renderWidth: number;
  renderHeight: number;
}

const DEG = Math.PI / 180;
/** Curl direction sign per hand. Fingers flex about local X (bone_dir is
 * local Y everywhere — verified from the GLB); the sign picks palm-ward curl. */
const CURL_SIGN: Record<HandSide, number> = { L: 1, R: 1 };
/** Arm raise sign per hand (raises the relaxed arms into a signing stance). */
const ARM_SIGN: Record<HandSide, number> = { L: 1, R: 1 };
/** Elbow bend sign per hand (flex forearm forward about local X). */
const ELBOW_SIGN: Record<HandSide, number> = { L: 1, R: 1 };

export class HumanAvatarScene {
  private canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private gltfLoader = new GLTFLoader();
  private model: THREE.Group | null = null;
  private bones = new Map<string, THREE.Bone>();
  private restQuaternions = new Map<string, THREE.Quaternion>();
  private morphTargets = new Map<string, THREE.Mesh[]>();
  private skinnedMeshJoints = 0;
  private meshCount = 0;
  private rafId: number | null = null;
  private dirty = true;
  private framesRendered = 0;
  private triangleCount = 0;
  private drawCalls = 0;

  /** Pose state (0..1 per finger, degrees for wrist twist). Applied from rest quaternions. */
  private curls: Record<HandSide, Record<FingerId, number>> = {
    L: { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 },
    R: { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 },
  };
  private wristTwist: Record<HandSide, number> = { L: 0, R: 0 };
  /** 0..1 raise of both upper arms (normalized convenience for demos). */
  private armRaise = 0;
  /** 0..1 extra elbow flexion stacked on top of the rest pose. */
  private elbowBend = 0;
  private morphValues = new Map<string, number>();

  private defaultView = { position: new THREE.Vector3(), target: new THREE.Vector3() };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "default" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = false;

    this.scene.background = new THREE.Color(0x11161d);

    // Hemisphere + key/fill directionals; no shadow maps.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x2a313c, 1.0));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1.6, 2.4, 1.8);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xcfd8ff, 0.5);
    fill.position.set(-1.8, 1.2, -1.2);
    this.scene.add(fill);

    this.camera = new THREE.PerspectiveCamera(34, 1, 0.05, 50);
    this.camera.position.set(0, 1.3, 2.4);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false; // on-demand rendering; damping needs a continuous loop
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 6;
    this.controls.maxPolarAngle = Math.PI * 0.75;
    this.controls.addEventListener("change", () => this.invalidate());

    this.resize();
    window.addEventListener("resize", this.resize);

    const tick = () => {
      if (this.dirty) {
        this.dirty = false;
        this.framesRendered += 1;
        this.renderer.render(this.scene, this.camera);
        this.drawCalls = this.renderer.info.render.calls;
        this.triangleCount = this.renderer.info.render.triangles;
      }
      this.rafId = requestAnimationFrame(tick);
    };
    tick();
  }

  private resize = () => {
    const { clientWidth, clientHeight } = this.canvas;
    if (!clientWidth || !clientHeight) return;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  };

  /** Request a re-render on the next frame. */
  invalidate = () => {
    this.dirty = true;
  };

  async load(url: string): Promise<void> {
    const gltf = await this.gltfLoader.loadAsync(url);
    this.model = gltf.scene;

    this.model.traverse((obj) => {
      const bone = obj as THREE.Bone;
      if (bone.isBone) {
        // Key by sanitized name: three.js strips reserved chars from node names,
        // so lookups use the same normalization (see rigMap.sanitizeBoneName).
        const key = sanitizeBoneName(bone.name);
        this.bones.set(key, bone);
        this.restQuaternions.set(key, bone.quaternion.clone());
      }
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        this.meshCount += 1;
        const dict = mesh.morphTargetDictionary;
        if (dict) {
          for (const name of Object.keys(dict)) {
            const list = this.morphTargets.get(name) ?? [];
            list.push(mesh);
            this.morphTargets.set(name, list);
          }
        }
        const sm = mesh as THREE.SkinnedMesh;
        if (sm.isSkinnedMesh && sm.skeleton) {
          this.skinnedMeshJoints = Math.max(this.skinnedMeshJoints, sm.skeleton.bones.length);
        }
      }
    });

    this.scene.add(this.model);
    this.frameWaistUp();
    this.invalidate();
  }

  /** Frame head→mid-thigh (hands in A-pose stay visible), target on the chest. */
  private frameWaistUp() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const fullHeight = box.max.y - box.min.y;
    // Retain from head top down to ~32% of total height (mid-thigh region).
    box.min.y = box.min.y + fullHeight * 0.32;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const fovRad = (this.camera.fov * Math.PI) / 180;
    const halfFov = Math.tan(fovRad / 2);
    let distance = size.y / (2 * halfFov);
    const horizontal = size.x / (2 * halfFov * this.camera.aspect);
    distance = Math.max(distance, horizontal) * 1.18;

    this.camera.position.set(center.x, center.y + size.y * 0.1, center.z + distance);
    this.controls.target.copy(center);
    this.camera.near = Math.max(0.05, distance / 100);
    this.camera.far = distance * 20;
    this.camera.updateProjectionMatrix();
    this.controls.update();

    this.defaultView.position.copy(this.camera.position);
    this.defaultView.target.copy(this.controls.target);
  }

  resetCamera() {
    this.camera.position.copy(this.defaultView.position);
    this.controls.target.copy(this.defaultView.target);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.invalidate();
  }

  getBones(): THREE.Bone[] {
    return [...this.bones.values()];
  }

  getBoneNames(): string[] {
    return [...this.bones.keys()].sort();
  }

  getMorphTargetNames(): string[] {
    return [...this.morphTargets.keys()];
  }

  setFingerCurl(side: HandSide, finger: FingerId, value: number) {
    this.curls[side][finger] = THREE.MathUtils.clamp(value, 0, 1);
    this.applyPose();
  }

  setAllFingerCurls(side: HandSide | "both", value: number) {
    const sides: HandSide[] = side === "both" ? ["L", "R"] : [side];
    for (const s of sides) {
      for (const { finger } of FINGERS) this.curls[s][finger] = value;
    }
    this.applyPose();
  }

  setWristTwist(side: HandSide, degrees: number) {
    this.wristTwist[side] = THREE.MathUtils.clamp(degrees, -75, 75);
    this.applyPose();
  }

  setArmPose(raise: number, elbowBend = raise) {
    this.armRaise = THREE.MathUtils.clamp(raise, 0, 1);
    this.elbowBend = THREE.MathUtils.clamp(elbowBend, 0, 1);
    this.applyPose();
  }

  /** Recompute every affected bone from its rest quaternion (idempotent). */
  private applyPose() {
    const xAxis = new THREE.Vector3(1, 0, 0);
    // 1) reset all bones that participate in the pose to rest
    for (const spec of FINGERS) {
      for (const side of ["L", "R"] as HandSide[]) {
        for (const name of spec.chain(side)) this.resetBone(name);
      }
    }
    this.resetBone("wrist.L");
    this.resetBone("wrist.R");
    for (const side of ["L", "R"] as HandSide[]) {
      this.resetBone(`upperarm01.${side}`);
      this.resetBone(`upperarm02.${side}`);
      this.resetBone(`lowerarm01.${side}`);
      this.resetBone(`lowerarm02.${side}`);
    }

    // 1b) arm raise + elbow bend (local X, same axis convention as fingers;
    // empirically raises the relaxed arms into a signing stance).
    if (this.armRaise !== 0 || this.elbowBend !== 0) {
      for (const side of ["L", "R"] as HandSide[]) {
        const sign = ARM_SIGN[side];
        const raise = this.armRaise * 75 * DEG * sign;
        const bend = this.elbowBend * 60 * DEG * ELBOW_SIGN[side];
        const apply = (name: string, angle: number) => {
          const key = sanitizeBoneName(name);
          const bone = this.bones.get(key);
          const rest = this.restQuaternions.get(key);
          if (!bone || !rest) return;
          bone.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromAxisAngle(xAxis, angle));
        };
        apply(`upperarm01.${side}`, raise * 0.65);
        apply(`upperarm02.${side}`, raise * 0.35);
        apply(`lowerarm01.${side}`, bend * 0.6);
        apply(`lowerarm02.${side}`, bend * 0.4);
      }
    }

    // 2) apply wrist twist (about the bone's local Y — its length axis)
    for (const side of ["L", "R"] as HandSide[]) {
      const wristKey = sanitizeBoneName(`wrist.${side}`);
      const wrist = this.bones.get(wristKey);
      const rest = this.restQuaternions.get(wristKey);
      if (wrist && rest) {
        wrist.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), this.wristTwist[side] * DEG));
      }
    }

    // 3) apply finger curls (local X-axis bend, proximal→distal)
    for (const spec of FINGERS) {
      for (const side of ["L", "R"] as HandSide[]) {
        const curl = this.curls[side][spec.finger];
        const names = spec.chain(side);
        const sign = CURL_SIGN[side];
        names.forEach((name, i) => {
          const key = sanitizeBoneName(name);
          const bone = this.bones.get(key);
          const rest = this.restQuaternions.get(key);
          if (!bone || !rest) return;
          const weight = spec.curlWeights[i] ?? 1;
          const angle = curl * spec.maxFlexDeg * weight * DEG * sign;
          bone.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromAxisAngle(xAxis, angle));
        });
      }
    }
    this.invalidate();
  }

  private resetBone(name: string) {
    const key = sanitizeBoneName(name);
    const bone = this.bones.get(key);
    const rest = this.restQuaternions.get(key);
    if (bone && rest) bone.quaternion.copy(rest);
  }

  setMorphTarget(name: string, value: number) {
    const meshes = this.morphTargets.get(name);
    if (!meshes) return;
    const v = THREE.MathUtils.clamp(value, 0, 1);
    for (const mesh of meshes) {
      const index = mesh.morphTargetDictionary?.[name];
      if (index === undefined || !mesh.morphTargetInfluences) continue;
      mesh.morphTargetInfluences[index] = v;
    }
    this.morphValues.set(name, v);
    this.invalidate();
  }

  setMorphTargets(values: Record<string, number>) {
    for (const [name, v] of Object.entries(values)) this.setMorphTarget(name, v);
  }

  resetPose() {
    for (const side of ["L", "R"] as HandSide[]) {
      for (const { finger } of FINGERS) this.curls[side][finger] = 0;
      this.wristTwist[side] = 0;
    }
    this.armRaise = 0;
    this.elbowBend = 0;
    this.applyPose();
  }

  resetFace() {
    for (const name of this.morphValues.keys()) this.setMorphTarget(name, 0);
    this.morphValues.clear();
    this.invalidate();
  }

  getFramesRendered(): number {
    return this.framesRendered;
  }

  getStats(): SceneStats {
    return {
      boneCount: this.bones.size,
      skinnedMeshJoints: this.skinnedMeshJoints,
      meshCount: this.meshCount,
      triangleCount: this.triangleCount,
      drawCalls: this.drawCalls,
      pixelRatio: this.renderer.getPixelRatio(),
      renderWidth: this.renderer.domElement.width,
      renderHeight: this.renderer.domElement.height,
    };
  }

  dispose() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    window.removeEventListener("resize", this.resize);
    this.controls.dispose();
    this.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.geometry?.dispose();
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const material of materials) {
          if (!material) continue;
          for (const value of Object.values(material)) {
            if (value instanceof THREE.Texture) value.dispose();
          }
          material.dispose();
        }
      }
    });
    this.renderer.dispose();
    this.model = null;
    this.bones.clear();
    this.restQuaternions.clear();
    this.morphTargets.clear();
  }
}
