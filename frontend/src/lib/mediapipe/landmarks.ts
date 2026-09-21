export interface LandmarkPoint {
  x: number
  y: number
  z: number
  visibility?: number
}

export const POSE_LANDMARKS = 33
export const FACE_LANDMARKS = 468
export const HAND_LANDMARKS = 21
export const LANDMARK_COUNT = POSE_LANDMARKS + FACE_LANDMARKS + HAND_LANDMARKS * 2
export const LANDMARK_DIMENSIONS = LANDMARK_COUNT * 3

// ponytail: temporal detection tuned for conversational ISL; raise PAUSE_MS if you see split sentences, lower if sluggish
export const TEMPORAL_MOTION_THRESHOLD = 0.012
export const TEMPORAL_PAUSE_MS = 650
export const TEMPORAL_MIN_FRAMES = 15

export interface HolisticLandmarks {
  pose?: LandmarkPoint[]
  face?: LandmarkPoint[]
  leftHand?: LandmarkPoint[]
  rightHand?: LandmarkPoint[]
}

function flattenComponent(points: LandmarkPoint[] | undefined, count: number): number[] {
  const flattened = new Array<number>(count * 3).fill(0)
  for (let index = 0; index < Math.min(points?.length ?? 0, count); index += 1) {
    const point = points![index]
    flattened[index * 3] = Number.isFinite(point.x) ? point.x : 0
    flattened[index * 3 + 1] = Number.isFinite(point.y) ? point.y : 0
    flattened[index * 3 + 2] = Number.isFinite(point.z) ? point.z : 0
  }
  return flattened
}

/** Matches iSign component order: pose, face, left hand, then right hand. */
export function flattenHolisticLandmarks(landmarks: HolisticLandmarks): number[] {
  const flattened = [
    ...flattenComponent(landmarks.pose, POSE_LANDMARKS),
    ...flattenComponent(landmarks.face, FACE_LANDMARKS),
    ...flattenComponent(landmarks.leftHand, HAND_LANDMARKS),
    ...flattenComponent(landmarks.rightHand, HAND_LANDMARKS),
  ]

  if (flattened.length !== LANDMARK_DIMENSIONS) {
    throw new Error(`Expected ${LANDMARK_DIMENSIONS} landmark values.`)
  }
  return flattened
}

/**
 * Accept any complete Holistic component so framing is not limited to a
 * full-body view. Missing components remain zero-padded for the model.
 */
export function hasReliableSignLandmarks(landmarks: HolisticLandmarks): boolean {
  const hasComponent =
    landmarks.leftHand?.length === HAND_LANDMARKS ||
    landmarks.rightHand?.length === HAND_LANDMARKS

  if (!hasComponent) {
    return false
  }

  const visibility = (landmarks.pose ?? [])
    .map((point) => point.visibility)
    .filter((value): value is number => typeof value === "number")

  return visibility.length === 0 || visibility.reduce((sum, value) => sum + value, 0) / visibility.length >= 0.5
}

/**
 * Browser port of notebook `normalize_pose_sequence()`. Normalization happens
 * after a full sentence is buffered, so its median body scale matches training.
 */
export function normalizeLandmarkSequence(frames: number[][]): number[][] {
  if (frames.length === 0) throw new Error("Cannot normalize an empty landmark sequence.")
  if (frames.some((frame) => frame.length !== LANDMARK_DIMENSIONS)) {
    throw new Error(`Each frame must contain ${LANDMARK_DIMENSIONS} landmark values.`)
  }

  const centered = frames.map((frame) => {
    const copy = [...frame]
    const center = [0, 1, 2].map(
      (dimension) => (copy[23 * 3 + dimension] + copy[24 * 3 + dimension]) / 2,
    )
    for (let index = 0; index < copy.length; index += 3) {
      copy[index] -= center[0]
      copy[index + 1] -= center[1]
      copy[index + 2] -= center[2]
    }
    return copy
  })

  const scales = centered
    .map((frame) => standardDeviation(frame.slice(0, POSE_LANDMARKS * 3)))
    .filter((value) => Number.isFinite(value) && value > 1e-6)
    .sort((first, second) => first - second)
  const scale = scales.length === 0 ? 1 : median(scales)

  return centered.map((frame) => frame.map((value) => (Number.isFinite(value) ? value / scale : 0)))
}

export function poseMotion(previous: number[], current: number[]): number {
  if (previous.length !== LANDMARK_DIMENSIONS || current.length !== LANDMARK_DIMENSIONS) return Infinity

  let total = 0
  let dimensions = 0
  const ranges = [
    [0, POSE_LANDMARKS * 3],
    [(POSE_LANDMARKS + FACE_LANDMARKS) * 3, LANDMARK_DIMENSIONS],
  ] as const

  for (const [start, end] of ranges) {
    for (let index = start; index < end; index += 1) {
      total += Math.abs(current[index] - previous[index])
      dimensions += 1
    }
  }
  return total / dimensions
}

function standardDeviation(values: number[]): number {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

function median(values: number[]): number {
  const midpoint = Math.floor(values.length / 2)
  return values.length % 2 === 0 ? (values[midpoint - 1] + values[midpoint]) / 2 : values[midpoint]
}
