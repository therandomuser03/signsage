// SignSage — rig map for the Ara 3D Gen "signsage-human.glb" (MakeHuman-style rig).
//
// Everything here is derived from the ACTUAL loaded skeleton at runtime.
// The expectations below are only used to (a) label the MakeHuman finger
// numbering and (b) report what is missing — no bone is assumed to exist.
//
// MakeHuman finger numbering: finger1 = thumb … finger5 = pinky.
// Metacarpals are numbered one behind their finger: metacarpal1 feeds
// finger2 (index). The thumb has NO metacarpal and hangs directly off wrist.

export type HandSide = "L" | "R";
export type FingerId = "thumb" | "index" | "middle" | "ring" | "pinky";

export interface FingerSpec {
  finger: FingerId;
  label: string;
  /** Expected chain in parent→child order, templated with the hand side. */
  chain: (side: HandSide) => string[];
  /** Per-joint curl weights (proximal→distal); metacarpal, when present, is first. */
  curlWeights: number[];
  /** Max curl angle in degrees at full slider (distributed per joint weight). */
  maxFlexDeg: number;
}

export const FINGERS: FingerSpec[] = [
  { finger: "thumb", label: "Thumb", chain: (s) => [`finger1-1.${s}`, `finger1-2.${s}`, `finger1-3.${s}`], curlWeights: [0.6, 0.8, 1.0], maxFlexDeg: 50 },
  { finger: "index", label: "Index", chain: (s) => [`metacarpal1.${s}`, `finger2-1.${s}`, `finger2-2.${s}`, `finger2-3.${s}`], curlWeights: [0.25, 1.0, 1.0, 0.8], maxFlexDeg: 95 },
  { finger: "middle", label: "Middle", chain: (s) => [`metacarpal2.${s}`, `finger3-1.${s}`, `finger3-2.${s}`, `finger3-3.${s}`], curlWeights: [0.25, 1.0, 1.0, 0.8], maxFlexDeg: 95 },
  { finger: "ring", label: "Ring", chain: (s) => [`metacarpal3.${s}`, `finger4-1.${s}`, `finger4-2.${s}`, `finger4-3.${s}`], curlWeights: [0.25, 1.0, 1.0, 0.8], maxFlexDeg: 95 },
  { finger: "pinky", label: "Pinky", chain: (s) => [`metacarpal4.${s}`, `finger5-1.${s}`, `finger5-2.${s}`, `finger5-3.${s}`], curlWeights: [0.3, 1.0, 1.0, 0.8], maxFlexDeg: 90 },
];

export interface BoneLike {
  name: string;
  parent: BoneLike | null;
}

/**
 * three.js strips reserved characters (".", ":", "[", "]") from node names at
 * import time (PropertyBinding.sanitizeNodeName), so the GLB's `wrist.L` is
 * `wristL` in the live scene. Normalize both sides before comparing; the GLB's
 * original names are shown in the GLB JSON, the scene holds sanitized ones.
 */
export function sanitizeBoneName(name: string): string {
  return name.replace(/[.[\]:]/g, "");
}

export interface FingerStatus {
  finger: FingerId;
  label: string;
  side: HandSide;
  /** Chain bones that actually exist in the loaded skeleton, parent→child. */
  present: string[];
  missing: string[];
  /** True when every consecutive present pair is really parented in that order. */
  connected: boolean;
  /** Per-joint presence matching FINGERS.curlWeights (false = bone absent). */
  jointPresent: boolean[];
}

export interface HandStatus {
  side: HandSide;
  wrist: string | null;
  fingers: FingerStatus[];
}

/** Map over the real skeleton; missing bones are reported, never assumed. */
export function inspectHands(bones: BoneLike[]): HandStatus[] {
  const byName = new Map(bones.map((b) => [sanitizeBoneName(b.name), b]));
  return (["L", "R"] as HandSide[]).map((side) => {
    const wristName = `wrist.${side}`;
    const findBone = (name: string) => byName.get(sanitizeBoneName(name));
    const fingers: FingerStatus[] = FINGERS.map((spec) => {
      const expected = spec.chain(side);
      const present: string[] = [];
      const missing: string[] = [];
      const jointPresent: boolean[] = [];
      for (const name of expected) {
        const ok = findBone(name) !== undefined;
        jointPresent.push(ok);
        (ok ? present : missing).push(findBone(name)?.name ?? name);
      }
      // Connectivity: walk the expected chain; for every consecutive pair that
      // exists, the child must actually be parented to the previous bone
      // (skipping expected members that are absent).
      let connected = present.length > 0;
      let anchor = findBone(wristName) ?? null;
      for (const name of expected) {
        const bone = findBone(name);
        if (!bone) continue;
        if (!anchor) {
          connected = false;
          break;
        }
        const parentName = bone.parent ? sanitizeBoneName(bone.parent.name) : null;
        const anchorName = anchor ? sanitizeBoneName(anchor.name) : null;
        if (parentName !== anchorName) connected = false;
        anchor = bone;
      }
      return { finger: spec.finger, label: spec.label, side, present, missing, connected, jointPresent };
    });
    const wristBone = findBone(wristName);
    return { side, wrist: wristBone?.name ?? null, fingers };
  });
}

export interface CategorySpec {
  label: string;
  match: (name: string) => boolean;
}

/** Coarse anatomical grouping of the whole 164-joint skeleton, for the panel. */
export const BONE_CATEGORIES: CategorySpec[] = [
  { label: "Spine", match: (n) => /^spine\d/.test(n) },
  { label: "Pelvis / breasts", match: (n) => /^(pelvis|breast)/.test(n) },
  { label: "Neck", match: (n) => /^neck\d/.test(n) },
  { label: "Head", match: (n) => n === "head" },
  { label: "Clavicles / shoulders", match: (n) => /^(clavicle|shoulder01)/.test(n) },
  { label: "Upper arms", match: (n) => /^upperarm\d/.test(n) },
  { label: "Lower arms", match: (n) => /^lowerarm\d/.test(n) },
  { label: "Wrists", match: (n) => /^wrist/.test(n) },
  { label: "Fingers (both hands)", match: (n) => /^(metacarpal\d|finger\d-\d)/.test(n) },
  { label: "Upper legs", match: (n) => /^upperleg\d/.test(n) },
  { label: "Lower legs", match: (n) => /^lowerleg\d/.test(n) },
  { label: "Feet", match: (n) => /^foot/.test(n) },
  { label: "Toes", match: (n) => /^toe\d-\d/.test(n) },
  { label: "Jaw / tongue / mouth", match: (n) => /^(jaw|tongue\d|special04|oris)/.test(n) },
  { label: "Eyes / eyelids", match: (n) => /^(eye|special0[56]|oculi|orbicularis)/.test(n) },
  { label: "Face muscles", match: (n) => /^(levator|temporalis|risorius|special0[13])/.test(n) },
  { label: "Other", match: () => true },
];

export function categorizeBones(names: string[]): { label: string; bones: string[] }[] {
  const clean = names.map(sanitizeBoneName);
  const assigned = new Set<string>();
  const groups: { label: string; bones: string[] }[] = [];
  for (const cat of BONE_CATEGORIES) {
    if (cat.label === "Other") continue;
    const bones = clean.filter((n) => !assigned.has(n) && cat.match(n));
    bones.forEach((b) => assigned.add(b));
    groups.push({ label: cat.label, bones });
  }
  groups.push({ label: "Other", bones: clean.filter((n) => !assigned.has(n)) });
  return groups;
}

/** Split actual morph-target names into inspection groups (ARKit-style names from the GLB). */
export function categorizeMorphs(names: string[]): { label: string; targets: string[] }[] {
  const groups = [
    { label: "Eyes — blink / wide / squint", targets: names.filter((n) => /^eye(Blink|Wide|Squint)/.test(n)) },
    { label: "Brow", targets: names.filter((n) => /^brow/.test(n)) },
    { label: "Jaw / mouth / nose", targets: names.filter((n) => /^(jaw|mouth|nose)/.test(n)) },
    { label: "Speech visemes", targets: names.filter((n) => /^viseme_/.test(n)) },
  ];
  const matched = new Set(groups.flatMap((g) => g.targets));
  const rest = names.filter((n) => !matched.has(n));
  if (rest.length) groups.push({ label: "Other", targets: rest });
  return groups;
}
