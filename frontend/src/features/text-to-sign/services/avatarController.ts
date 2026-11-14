// SignSage — Avatar controller
//
// Loads ONE rigged VRM avatar and plays short GLB animation clips against it,
// keyed by gloss ID. This deliberately avoids per-sign 3D models: every clip
// must be authored on the same skeleton as the base avatar (export the VRM
// from VRoid once, animate it in Blender per-sign, export each sign as its
// own small GLB containing only the AnimationClip).
//
// npm dependencies: three, @pixiv/three-vrm

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRM, VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { assetExists } from "./assetExists";
import type { SignManifest } from "../types";

const CROSSFADE_SECONDS = 0.25;

export class AvatarController {
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private clock = new THREE.Clock();
  private gltfLoader = new GLTFLoader();

  private vrm: VRM | null = null;
  private mixer: THREE.AnimationMixer | null = null;
  private currentAction: THREE.AnimationAction | null = null;

  private clipCache = new Map<string, THREE.AnimationClip>();
  private manifest: SignManifest;
  /** Gloss IDs whose animation clip could not be loaded (missing asset). */
  private unavailableGlosses = new Set<string>();
  private rafId: number | null = null;
  private canvas: HTMLCanvasElement;
  private runId = 0;

  constructor(canvas: HTMLCanvasElement, manifest: SignManifest) {
    this.canvas = canvas;
    this.manifest = manifest;

    this.gltfLoader.register((parser) => new VRMLoaderPlugin(parser));

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
    this.camera.position.set(0, 1.3, 2.2);
    this.camera.lookAt(0, 1.1, 0);

    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(1, 2, 1);
    this.scene.add(key, new THREE.AmbientLight(0xffffff, 0.6));

    this.resize();
    window.addEventListener("resize", this.resize);
  }

  private resize = () => {
    const { clientWidth, clientHeight } = this.canvas;
    if (!clientWidth || !clientHeight) return;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
  };

  /** Load the base avatar once. Call before play(). */
  async loadAvatar(vrmUrl: string): Promise<void> {
    const gltf = await this.gltfLoader.loadAsync(vrmUrl);
    const vrm = gltf.userData.vrm as VRM;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);

    this.vrm = vrm;
    this.mixer = new THREE.AnimationMixer(vrm.scene);
    this.scene.add(vrm.scene);

    // Frame the loaded avatar from its actual bounds so the full character
    // stays in view regardless of the VRM's scale or pose.
    this.fitCameraToAvatar();

    this.startLoop();
  }

  /** Fit the camera to the avatar so the upper body (torso, head, hands)
   * fills the frame. Framing the full figure (hips down to feet) made the
   * signing-relevant parts appear small/compact, so we exclude the lower
   * portion (feet/lower legs) from the distance calculation. This keeps hands
   * and face readably large without clipping raised arms, and matches the
   * framing used by the procedural avatar fallback. */
  private fitCameraToAvatar() {
    if (!this.vrm) return;
    const box = new THREE.Box3().setFromObject(this.vrm.scene);
    // Exclude the bottom ~28% (feet/lower legs) so the camera frames the upper
    // body rather than the full figure. Hands at rest sit around hip level,
    // which stays inside the retained range.
    const trim = 0.28;
    box.min.y = box.min.y + (box.max.y - box.min.y) * trim;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const fovRad = (this.camera.fov * Math.PI) / 180;
    const halfFov = Math.tan(fovRad / 2);
    const aspect = this.camera.aspect;
    let distance = size.y / (2 * halfFov);
    if (distance * 2 * halfFov * aspect < size.x) {
      distance = size.x / (2 * halfFov * aspect);
    }
    distance *= 1.35;
    this.camera.position.set(center.x, center.y + size.y * 0.12, center.z + distance);
    this.camera.lookAt(center);
    this.camera.near = 0.1;
    this.camera.far = Math.max(20, distance * 4);
    this.camera.updateProjectionMatrix();
  }

  /** Fetch (and cache) the AnimationClip for a gloss ID's GLB file. */
  private async loadClip(glossId: string): Promise<THREE.AnimationClip> {
    const cached = this.clipCache.get(glossId);
    if (cached) return cached;

    const entry = this.manifest[glossId];
    if (!entry) throw new Error(`No sign manifest entry for gloss "${glossId}"`);

    // Reject missing assets early (e.g. Vercel SPA fallback returning HTML 200)
    // so the GLTFLoader never tries to parse an HTML page as a GLB.
    if (!(await assetExists(entry.animationUrl))) {
      throw new Error(`Animation asset not available: ${entry.animationUrl}`);
    }

    const gltf = await this.gltfLoader.loadAsync(entry.animationUrl);
    const clip = gltf.animations[0];
    if (!clip) throw new Error(`GLB at ${entry.animationUrl} has no animation clip`);

    this.clipCache.set(glossId, clip);
    return clip;
  }

  /**
   * Play a single gloss's clip, crossfading from whatever is currently
   * playing. Resolves once the clip has finished (non-looping).
   * Returns false when the gloss's animation asset is missing (the clip is
   * skipped rather than aborting the whole sequence).
   */
  async play(glossId: string): Promise<boolean> {
    if (!this.mixer || !this.vrm) throw new Error("Avatar not loaded yet");

    let clip: THREE.AnimationClip;
    try {
      clip = await this.loadClip(glossId);
    } catch {
      this.unavailableGlosses.add(glossId);
      return false;
    }
    const nextAction = this.mixer.clipAction(clip);
    nextAction.reset();
    nextAction.setLoop(THREE.LoopOnce, 1);
    nextAction.clampWhenFinished = true;

    if (this.currentAction) {
      nextAction.play();
      this.currentAction.crossFadeTo(nextAction, CROSSFADE_SECONDS, false);
    } else {
      nextAction.fadeIn(CROSSFADE_SECONDS).play();
    }
    this.currentAction = nextAction;

    const durationMs = clip.duration * 1000;
    await new Promise((resolve) => setTimeout(resolve, durationMs));
    return true;
  }

  /** Play a sequence of glosses back-to-back. Returns the gloss IDs whose
   * animation assets were unavailable (skipped, not fatal). */
  async playSequence(glossIds: string[], onStep?: (id: string, index: number) => void): Promise<string[]> {
    const thisRun = ++this.runId;
    const unavailable: string[] = [];
    for (let i = 0; i < glossIds.length; i++) {
      if (thisRun !== this.runId) return unavailable; // stop() was called
      onStep?.(glossIds[i], i);
      const ok = await this.play(glossIds[i]);
      if (!ok) unavailable.push(glossIds[i]);
    }
    return unavailable;
  }

  /** Gloss IDs whose animation clips could not be loaded. */
  unavailableGlossesList(): string[] {
    return [...this.unavailableGlosses];
  }

  /** Cancel an in-flight playSequence; the awaiting promise resolves. */
  stop(): void {
    this.runId += 1;
    this.currentAction?.fadeOut(CROSSFADE_SECONDS);
    this.currentAction = null;
  }

  private startLoop() {
    const tick = () => {
      const delta = this.clock.getDelta();
      this.mixer?.update(delta);
      this.vrm?.update(delta); // spring bones, look-at, expressions
      this.renderer.render(this.scene, this.camera);
      this.rafId = requestAnimationFrame(tick);
    };
    tick();
  }

  dispose() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    window.removeEventListener("resize", this.resize);
    if (this.vrm) VRMUtils.deepDispose(this.vrm.scene);
    this.renderer.dispose();
  }
}
