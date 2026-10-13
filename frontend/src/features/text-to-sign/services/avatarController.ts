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
    this.startLoop();
  }

  /** Fetch (and cache) the AnimationClip for a gloss ID's GLB file. */
  private async loadClip(glossId: string): Promise<THREE.AnimationClip> {
    const cached = this.clipCache.get(glossId);
    if (cached) return cached;

    const entry = this.manifest[glossId];
    if (!entry) throw new Error(`No sign manifest entry for gloss "${glossId}"`);

    const gltf = await this.gltfLoader.loadAsync(entry.animationUrl);
    const clip = gltf.animations[0];
    if (!clip) throw new Error(`GLB at ${entry.animationUrl} has no animation clip`);

    this.clipCache.set(glossId, clip);
    return clip;
  }

  /**
   * Play a single gloss's clip, crossfading from whatever is currently
   * playing. Resolves once the clip has finished (non-looping).
   */
  async play(glossId: string): Promise<void> {
    if (!this.mixer || !this.vrm) throw new Error("Avatar not loaded yet");

    const clip = await this.loadClip(glossId);
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
  }

  /** Play a sequence of glosses back-to-back. */
  async playSequence(glossIds: string[], onStep?: (id: string, index: number) => void): Promise<void> {
    const thisRun = ++this.runId;
    for (let i = 0; i < glossIds.length; i++) {
      if (thisRun !== this.runId) return; // stop() was called
      onStep?.(glossIds[i], i);
      await this.play(glossIds[i]);
    }
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
