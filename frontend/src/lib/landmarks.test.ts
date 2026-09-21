import { describe, expect, it } from "vitest"

import {
  LANDMARK_DIMENSIONS,
  flattenHolisticLandmarks,
  hasReliableSignLandmarks,
  normalizeLandmarkSequence,
  poseMotion,
} from "./mediapipe/landmarks"

const point = (value: number, visibility = 0.9) => ({ x: value, y: value + 0.1, z: value + 0.2, visibility })

describe("browser landmark preparation", () => {
  it("uses iSign's pose, face, left-hand, right-hand order and zero pads missing components", () => {
    const flattened = flattenHolisticLandmarks({ pose: [point(1)], leftHand: [point(2)] })

    expect(flattened).toHaveLength(LANDMARK_DIMENSIONS)
    expect(flattened.slice(0, 3)).toEqual([1, 1.1, 1.2])
    expect(flattened.slice(33 * 3, 33 * 3 + 3)).toEqual([0, 0, 0])
    expect(flattened.slice((33 + 468) * 3, (33 + 468) * 3 + 3)).toEqual([2, 2.1, 2.2])
  })

  it("accepts any tracked Holistic component without requiring a full body", () => {
    const pose = Array.from({ length: 33 }, () => point(0))
    const hand = Array.from({ length: 21 }, () => point(0))

    expect(hasReliableSignLandmarks({ pose, leftHand: hand })).toBe(true)
    expect(hasReliableSignLandmarks({ leftHand: hand })).toBe(true)
    expect(hasReliableSignLandmarks({ pose })).toBe(false)
    expect(hasReliableSignLandmarks({})).toBe(false)
  })

  it("centers every frame on the hip midpoint and uses one sentence scale", () => {
    const first = new Array<number>(LANDMARK_DIMENSIONS).fill(0)
    const second = new Array<number>(LANDMARK_DIMENSIONS).fill(0)
    first[23 * 3] = 1
    first[24 * 3] = 3
    second[23 * 3] = 5
    second[24 * 3] = 7
    first[0] = 4
    second[0] = 8

    const normalized = normalizeLandmarkSequence([first, second])
    expect(normalized[0][23 * 3]).toBeCloseTo(-normalized[0][24 * 3])
    expect(normalized[1][23 * 3]).toBeCloseTo(-normalized[1][24 * 3])
    expect(poseMotion(normalized[0], normalized[1])).toBeGreaterThan(0)
  })
})
