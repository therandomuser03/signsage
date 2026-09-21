/**
 * SignSage frontend landmark validation.
 *
 * The browser captures RAW MediaPipe Holistic landmarks.
 * Canonical training-time normalization is performed by the Python backend
 * in pose_preprocessing.py / normalize_pose_sequence().
 *
 * Do not add a second normalization algorithm here: doing so would create
 * training/inference distribution skew or double-normalization.
 */

export const LANDMARK_COUNT = 543
export const COORDINATES_PER_LANDMARK = 3
export const FRAME_SIZE =
  LANDMARK_COUNT * COORDINATES_PER_LANDMARK

export function isValidLandmarkFrame(
  frame: number[],
): boolean {
  return (
    Array.isArray(frame) &&
    frame.length === FRAME_SIZE &&
    frame.every(
      (value) =>
        typeof value === "number" &&
        Number.isFinite(value),
    )
  )
}

export function isValidLandmarkSequence(
  sequence: number[][],
): boolean {
  return (
    Array.isArray(sequence) &&
    sequence.length > 0 &&
    sequence.every(isValidLandmarkFrame)
  )
}

/**
 * Compatibility wrapper retained for existing imports.
 *
 * IMPORTANT:
 * This function validates and clones RAW MediaPipe landmarks.
 * It does NOT normalize them.
 *
 * The backend must call:
 *   prepare_landmarks(..., input_mode="raw")
 */
export function normalizeLandmarkSequence(
  sequence: number[][],
): number[][] {
  if (!isValidLandmarkSequence(sequence)) {
    throw new Error(
      `Invalid landmark sequence: expected non-empty frames with exactly ${FRAME_SIZE} finite values per frame.`,
    )
  }

  return sequence.map((frame) => [...frame])
}
