// SignSage — browser landmark extraction.
//
// Mirrors backend/app/ml/contracts/landmark_contract.py:
//   543 landmarks per frame in component order
//   POSE(33) + FACE(468) + LEFT_HAND(21) + RIGHT_HAND(21), [x, y, z] each.
// Missing parts are zeros. Raw coordinates only — the backend normalizes.
// Pose *world* landmarks are excluded (backend mask covers range(543)).

import type {
  FaceLandmarker,
  HandLandmarker,
  NormalizedLandmark,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

export const TOTAL_LANDMARKS = 543;
export const POSE_COUNT = 33;
export const FACE_COUNT = 468;
export const HAND_COUNT = 21;

/** Canonical component order. Must match backend COMPONENT_ORDER exactly:
 * POSE[0:33] + FACE[33:501] + LEFT_HAND[501:522] + RIGHT_HAND[522:543]. */
export const COMPONENT_ORDER = [
  "POSE_LANDMARKS",
  "FACE_LANDMARKS",
  "LEFT_HAND_LANDMARKS",
  "RIGHT_HAND_LANDMARKS",
] as const;

export const COMPONENT_RANGES: Record<string, [number, number]> = {
  POSE_LANDMARKS: [0, 33],
  FACE_LANDMARKS: [33, 501],
  LEFT_HAND_LANDMARKS: [501, 522],
  RIGHT_HAND_LANDMARKS: [522, 543],
};

// Fail loudly at module load if the counts ever drift from the contract.
if (POSE_COUNT + FACE_COUNT + HAND_COUNT * 2 !== TOTAL_LANDMARKS) {
  throw new Error(
    `landmark contract drift: ${POSE_COUNT}+${FACE_COUNT}+${HAND_COUNT}*2 != ${TOTAL_LANDMARKS}`
  );
}

export type LandmarkPoint = [number, number, number];

export interface HolisticFrame {
  /** 543 [x, y, z] points in contract order. Ready for landmarks.frame. */
  landmarks: LandmarkPoint[];
  pose: NormalizedLandmark[];
  face: NormalizedLandmark[];
  leftHand: NormalizedLandmark[];
  rightHand: NormalizedLandmark[];
  /** True when at least one body part was detected in this frame. */
  detected: boolean;
}

const zero = (): LandmarkPoint => [0, 0, 0];

function toPoints(
  marks: NormalizedLandmark[] | undefined,
  count: number
): LandmarkPoint[] {
  const out: LandmarkPoint[] = [];
  for (let i = 0; i < count; i++) {
    const m = marks?.[i];
    out.push(m ? [m.x, m.y, m.z ?? 0] : zero());
  }
  return out;
}

export class HolisticExtractor {
  private pose: PoseLandmarker | null = null;
  private face: FaceLandmarker | null = null;
  private hands: HandLandmarker | null = null;
  private lastVideoTime = -1;

  /** Loads models. Needs network for the pinned WASM bundle on first run. */
  async init(): Promise<void> {
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
    );
    // CPU delegate everywhere: the GPU delegate can hang indefinitely on
    // software WebGL (SwiftShader/remote desktops), freezing "Initializing
    // landmark detection". CPU is reliably fast enough for 543 points at 10-30 fps.
    [this.pose, this.face, this.hands] = await Promise.all([
      vision.PoseLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
      }),
      vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      }),
      vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      }),
    ]);
  }

  detect(video: HTMLVideoElement, nowMs: number): HolisticFrame | null {
    if (!this.pose || !this.face || !this.hands) return null;
    // VIDEO mode requires a strictly increasing timestamp.
    if (video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = video.currentTime;

    const poseRes = this.pose.detectForVideo(video, nowMs);
    const faceRes = this.face.detectForVideo(video, nowMs);
    const handRes = this.hands.detectForVideo(video, nowMs);

    const pose = poseRes.landmarks?.[0] ?? [];
    const face = faceRes.faceLandmarks?.[0] ?? [];

    // Handedness labels follow the image ("Left" = appears on the left).
    // Training pose files use the person's own left/right, so a mirrored
    // selfie preview can swap hands — an accepted calibration limit, not
    // silent corruption: the backend sees the same convention every frame.
    let leftHand: NormalizedLandmark[] = [];
    let rightHand: NormalizedLandmark[] = [];
    const hands = handRes.landmarks ?? [];
    const handed = handRes.handedness ?? [];
    hands.forEach((marks, i) => {
      const label = handed[i]?.[0]?.categoryName;
      if (label === "Left" && leftHand.length === 0) leftHand = marks;
      else if (label === "Right" && rightHand.length === 0) rightHand = marks;
      else if (leftHand.length === 0) leftHand = marks;
      else if (rightHand.length === 0) rightHand = marks;
    });

    const landmarks: LandmarkPoint[] = [
      ...toPoints(pose, POSE_COUNT),
      ...toPoints(face, FACE_COUNT),
      ...toPoints(leftHand, HAND_COUNT),
      ...toPoints(rightHand, HAND_COUNT),
    ];

    return {
      landmarks,
      pose,
      face,
      leftHand,
      rightHand,
      detected:
        pose.length > 0 ||
        face.length > 0 ||
        leftHand.length > 0 ||
        rightHand.length > 0,
    };
  }

  close(): void {
    this.pose?.close();
    this.face?.close();
    this.hands?.close();
    this.pose = this.face = this.hands = null;
  }
}
