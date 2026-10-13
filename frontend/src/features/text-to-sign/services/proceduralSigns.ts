// SignSage — procedural sign motions (fallback when VRM/GLB assets are absent).
//
// Distinct, labeled motion per gloss so a text->sign sequence visibly plays.
// This is a motion PREVIEW, not authored ISL: the UI must say so. When real
// GLB clips ship, the VRM AvatarController replaces this without changing
// the GlossSequence shape or the AvatarViewerHandle contract.

export interface Pose {
  /** shoulder euler [x,y,z] rad, +x raises arm forward, +z moves arm outward */
  shoulderL: [number, number, number];
  shoulderR: [number, number, number];
  /** elbow flexion rad (negative bends forearm forward/up) */
  elbowL: number;
  elbowR: number;
  /** head euler [nod(+down), shake, tilt] */
  head: [number, number, number];
  /** torso lean forward rad */
  torso: number;
}

export interface SignMotion {
  durationMs: number;
  keys: { at: number; pose: Partial<Pose> }[];
}

const REST: Pose = {
  shoulderL: [0, 0, 0.12],
  shoulderR: [0, 0, -0.12],
  elbowL: -0.2,
  elbowR: -0.2,
  head: [0, 0, 0],
  torso: 0,
};

function full(over: Partial<Pose>): Pose {
  return { ...REST, ...over };
}

const WAVE_R: Partial<Pose> = { shoulderR: [-1.9, 0, -0.5], elbowR: -0.3 };
const WAVE_L: Partial<Pose> = { shoulderR: [-1.9, 0, -1.1], elbowR: -0.3 };

export const PROCEDURAL_SIGNS: Record<string, SignMotion> = {
  // Open-palm wave: right arm up, swaying side to side.
  HELLO: {
    durationMs: 1500,
    keys: [
      { at: 0, pose: {} },
      { at: 0.25, pose: WAVE_R },
      { at: 0.45, pose: WAVE_L },
      { at: 0.65, pose: WAVE_R },
      { at: 0.85, pose: WAVE_L },
      { at: 1, pose: {} },
    ],
  },
  // Hand to chest then forward + slight bow: gratitude.
  THANK_YOU: {
    durationMs: 1500,
    keys: [
      { at: 0, pose: {} },
      { at: 0.35, pose: { shoulderR: [-0.9, 0, -0.3], elbowR: -1.6, head: [0.15, 0, 0], torso: 0.08 } },
      { at: 0.65, pose: { shoulderR: [-1.2, 0, 0], elbowR: -0.2, head: [0.3, 0, 0], torso: 0.18 } },
      { at: 1, pose: {} },
    ],
  },
  // Head bowed, palms together in front.
  SORRY: {
    durationMs: 1400,
    keys: [
      { at: 0, pose: {} },
      {
        at: 0.5,
        pose: {
          shoulderL: [-0.8, 0.5, 0.4], shoulderR: [-0.8, -0.5, -0.4],
          elbowL: -1.2, elbowR: -1.2, head: [0.45, 0, 0], torso: 0.22,
        },
      },
      { at: 1, pose: {} },
    ],
  },
  // Right palm circles on the chest.
  PLEASE: {
    durationMs: 1500,
    keys: [
      { at: 0, pose: {} },
      { at: 0.3, pose: { shoulderR: [-0.7, 0.4, -0.3], elbowR: -1.5 } },
      { at: 0.55, pose: { shoulderR: [-0.7, -0.4, -0.3], elbowR: -1.5 } },
      { at: 0.8, pose: { shoulderR: [-0.7, 0.4, -0.3], elbowR: -1.5 } },
      { at: 1, pose: {} },
    ],
  },
  // Head nod + small right-arm pump: affirmation.
  YES: {
    durationMs: 1200,
    keys: [
      { at: 0, pose: {} },
      { at: 0.3, pose: { head: [0.35, 0, 0], shoulderR: [-1.4, 0, -0.2], elbowR: -1.1 } },
      { at: 0.55, pose: { head: [-0.1, 0, 0], shoulderR: [-1.0, 0, -0.2], elbowR: -0.8 } },
      { at: 0.8, pose: { head: [0.35, 0, 0], shoulderR: [-1.4, 0, -0.2], elbowR: -1.1 } },
      { at: 1, pose: {} },
    ],
  },
  // Head shake + forearms crossed: negation.
  NO: {
    durationMs: 1300,
    keys: [
      { at: 0, pose: {} },
      {
        at: 0.35,
        pose: {
          head: [0, 0.5, 0], shoulderL: [-0.9, 0.6, 0.3], shoulderR: [-0.9, -0.6, -0.3],
          elbowL: -1.4, elbowR: -1.4,
        },
      },
      {
        at: 0.65,
        pose: {
          head: [0, -0.5, 0], shoulderL: [-0.9, 0.6, 0.3], shoulderR: [-0.9, -0.6, -0.3],
          elbowL: -1.4, elbowR: -1.4,
        },
      },
      { at: 1, pose: {} },
    ],
  },
  // Both arms raised high: call for help.
  HELP: {
    durationMs: 1400,
    keys: [
      { at: 0, pose: {} },
      {
        at: 0.45,
        pose: {
          shoulderL: [-2.4, 0, 0.5], shoulderR: [-2.4, 0, -0.5],
          elbowL: -0.2, elbowR: -0.2, head: [-0.25, 0, 0],
        },
      },
      { at: 0.7, pose: { shoulderL: [-2.0, 0, 0.5], shoulderR: [-2.0, 0, -0.5], elbowL: -0.2, elbowR: -0.2, head: [-0.25, 0, 0] } },
      { at: 1, pose: {} },
    ],
  },
  // Right hand to the mouth: drink.
  WATER: {
    durationMs: 1300,
    keys: [
      { at: 0, pose: {} },
      { at: 0.45, pose: { shoulderR: [-0.6, 0, -0.4], elbowR: -2.2, head: [-0.15, 0, 0] } },
      { at: 0.7, pose: { shoulderR: [-0.6, 0, -0.4], elbowR: -2.4, head: [0.1, 0, 0] } },
      { at: 1, pose: {} },
    ],
  },
  // Right hand to the mouth twice: eat.
  FOOD: {
    durationMs: 1400,
    keys: [
      { at: 0, pose: {} },
      { at: 0.3, pose: { shoulderR: [-0.6, 0, -0.4], elbowR: -2.3, head: [0.05, 0, 0] } },
      { at: 0.5, pose: { shoulderR: [-0.4, 0, -0.3], elbowR: -1.2 } },
      { at: 0.75, pose: { shoulderR: [-0.6, 0, -0.4], elbowR: -2.3, head: [0.05, 0, 0] } },
      { at: 1, pose: {} },
    ],
  },
  // Index to the chest: self-reference for name.
  NAME: {
    durationMs: 1200,
    keys: [
      { at: 0, pose: {} },
      { at: 0.5, pose: { shoulderR: [-0.5, 0.5, -0.2], elbowR: -1.7, head: [0.1, 0, 0] } },
      { at: 1, pose: {} },
    ],
  },
  // Point forward at the viewer.
  YOU: {
    durationMs: 1200,
    keys: [
      { at: 0, pose: {} },
      { at: 0.5, pose: { shoulderR: [-1.5, 0, 0], elbowR: -0.05, head: [0, 0, 0] } },
      { at: 1, pose: {} },
    ],
  },
  // Point at own chest.
  ME: {
    durationMs: 1200,
    keys: [
      { at: 0, pose: {} },
      { at: 0.5, pose: { shoulderR: [-0.4, 0.6, -0.1], elbowR: -1.9, head: [0.08, 0, 0] } },
      { at: 1, pose: {} },
    ],
  },
  // Shrug: shoulders up, palms up, head tilts side to side.
  WHERE: {
    durationMs: 1400,
    keys: [
      { at: 0, pose: {} },
      {
        at: 0.4,
        pose: {
          shoulderL: [-0.7, 0, 0.9], shoulderR: [-0.7, 0, -0.9],
          elbowL: -1.3, elbowR: -1.3, head: [0, 0, 0.25], torso: -0.05,
        },
      },
      {
        at: 0.7,
        pose: {
          shoulderL: [-0.7, 0, 0.9], shoulderR: [-0.7, 0, -0.9],
          elbowL: -1.3, elbowR: -1.3, head: [0, 0, -0.25], torso: -0.05,
        },
      },
      { at: 1, pose: {} },
    ],
  },
  // Palms-up query + head tilt: question.
  HOW: {
    durationMs: 1300,
    keys: [
      { at: 0, pose: {} },
      {
        at: 0.5,
        pose: {
          shoulderL: [-0.9, 0, 0.7], shoulderR: [-0.9, 0, -0.7],
          elbowL: -1.0, elbowR: -1.0, head: [0.15, 0.3, 0.15],
        },
      },
      { at: 1, pose: {} },
    ],
  },
};

export function motionFor(glossId: string): SignMotion {
  // Unknown IDs (shouldn't happen: mapper filters to manifest) get a neutral
  // beat so playback never crashes on a missing key.
  return PROCEDURAL_SIGNS[glossId] ?? { durationMs: 800, keys: [{ at: 0, pose: {} }, { at: 1, pose: {} }] };
}

export function lerpPose(a: Pose, b: Pose, t: number): Pose {
  const lerp = (x: number, y: number) => x + (y - x) * t;
  const lerp3 = (u: [number, number, number], v: [number, number, number]): [number, number, number] => [
    lerp(u[0], v[0]),
    lerp(u[1], v[1]),
    lerp(u[2], v[2]),
  ];
  return {
    shoulderL: lerp3(a.shoulderL, b.shoulderL),
    shoulderR: lerp3(a.shoulderR, b.shoulderR),
    elbowL: lerp(a.elbowL, b.elbowL),
    elbowR: lerp(a.elbowR, b.elbowR),
    head: lerp3(a.head, b.head),
    torso: lerp(a.torso, b.torso),
  };
}

/** Pose at normalized time t for a motion (REST outside key range). */
export function poseAt(motion: SignMotion, t: number): Pose {
  const keys = motion.keys;
  let prev = { at: 0, pose: {} as Partial<Pose> };
  for (const k of keys) {
    if (t <= k.at) {
      const span = k.at - prev.at || 1;
      return lerpPose(full(prev.pose), full(k.pose), (t - prev.at) / span);
    }
    prev = k;
  }
  return full(keys[keys.length - 1].pose);
}
