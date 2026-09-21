import { useEffect, useRef } from "react"
import * as THREE from "three"
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js"

const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [27, 29], [29, 31],
  [24, 26], [26, 28], [28, 30], [30, 32],
]

// Standard 21-point MediaPipe Hand topology (wrist=0, thumb=1-4, index=5-8,
// middle=9-12, ring=13-16, pinky=17-20). Applied to both hands.
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]

// Feature offsets into a [1629] full-feature frame (pose(33) + face(468) +
// leftHand(21) + rightHand(21), 3 floats each -- see
// backend/app/services/pose_preprocessing.py TRUE_* constants). Only pose
// (0..98) and both hands (1503..1628) are ever rendered; face is skipped.
const LEFT_HAND_FEATURE_START = 1503
const RIGHT_HAND_FEATURE_START = 1566
const FACE_FEATURE_START = 99 * 3

interface PoseCanvasProps {
  frames: number[][]
  fps: number
  /** True for bounded demo phrase clips: the recorded prototype only has
   * real captured motion in the hand landmarks (see AUDIT_NOTES.md), so the
   * body-pose skeleton is skipped rather than drawing a fabricated/static
   * torso from zeroed-out data. */
  handsOnly?: boolean
}

export function PoseCanvas({ frames, fps, handsOnly = false }: PoseCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || frames.length === 0) return
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x080f17)
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100)
    camera.position.z = 6
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    const avatar = new THREE.Group()
    scene.add(avatar)
    scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 2.2))
    const keyLight = new THREE.DirectionalLight(0xffffff, 3)
    keyLight.position.set(2, 4, 5)
    scene.add(keyLight)
    const modelRoot = new THREE.Group()
    avatar.add(modelRoot)
    let model: THREE.Group | null = null
    let mixer: THREE.AnimationMixer | null = null
    const clock = new THREE.Clock()
    let disposed = false
    new GLTFLoader().load("/model.glb", (gltf) => {
      if (disposed) return
      model = gltf.scene
      const bounds = new THREE.Box3().setFromObject(model)
      const size = bounds.getSize(new THREE.Vector3())
      const center = bounds.getCenter(new THREE.Vector3())
      const scale = 3.6 / Math.max(size.y, 0.001)
      model.scale.setScalar(scale)
      model.position.set(-center.x * scale, -center.y * scale - 1.35, -center.z * scale)
      modelRoot.add(model)
      if (gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(model)
        mixer.clipAction(gltf.animations[0]).play()
      }
    })
    let animationFrame = 0
    let startedAt = 0
    let currentFrame: THREE.Group | null = null

    const resize = () => {
      const bounds = canvas.getBoundingClientRect()
      renderer.setSize(Math.max(1, bounds.width), Math.max(1, bounds.height), false)
      camera.aspect = Math.max(1, bounds.width) / Math.max(1, bounds.height)
      camera.updateProjectionMatrix()
    }

    const draw = (timestamp: number) => {
      if (!startedAt) startedAt = timestamp
      resize()
      currentFrame?.traverse((object) => {
        if (object instanceof THREE.LineSegments || object instanceof THREE.Points) {
          object.geometry.dispose()
          ;(object.material as THREE.Material).dispose()
        }
      })
      if (currentFrame) avatar.remove(currentFrame)
      const index = Math.floor(((timestamp - startedAt) / 1000) * fps) % frames.length
      const frame = frames[index]
      currentFrame = createAvatarFrame(frame, handsOnly)
      if (model) applyHandPose(model, frame)
      avatar.add(currentFrame)
      avatar.rotation.y = Math.sin(timestamp / 2400) * 0.18
      mixer?.update(clock.getDelta())
      renderer.render(scene, camera)
      animationFrame = window.requestAnimationFrame(draw)
    }

    animationFrame = window.requestAnimationFrame(draw)
    return () => {
      window.cancelAnimationFrame(animationFrame)
      disposed = true
      if (model) modelRoot.remove(model)
      currentFrame?.traverse((object) => {
        if (object instanceof THREE.LineSegments || object instanceof THREE.Points) {
          object.geometry.dispose()
          ;(object.material as THREE.Material).dispose()
        }
      })
      renderer.dispose()
    }
  }, [fps, frames, handsOnly])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" aria-label={handsOnly ? "Recorded demo hand-motion 3D avatar" : "Generated 3D pose avatar"} role="img" />
}

function createAvatarFrame(frame: number[], handsOnly: boolean) {
  const group = new THREE.Group()
  const pose = extract3DPoints(frame, 0, 33)
  const face = extract3DPoints(frame, FACE_FEATURE_START, 468)
  const leftHand = extract3DPoints(frame, LEFT_HAND_FEATURE_START, 21)
  const rightHand = extract3DPoints(frame, RIGHT_HAND_FEATURE_START, 21)
  const renderedLeftHand = handsOnly ? normalizeHand(leftHand, -0.55) : leftHand
  const renderedRightHand = handsOnly ? normalizeHand(rightHand, 0.55) : rightHand
  const populated = [...(handsOnly ? [] : pose), ...renderedLeftHand, ...renderedRightHand].filter((point) => point.lengthSq() > 0)
  const span = Math.max(...populated.map((point) => Math.max(Math.abs(point.x), Math.abs(point.y))), 0.5)
  const scale = 2.2 / span
  const project = (point: THREE.Vector3) => new THREE.Vector3(point.x * scale, -point.y * scale, point.z * scale * 0.35)

  if (!handsOnly) addConnections(group, pose, POSE_CONNECTIONS, 0xe9d5ff, 0.035, project)
  if (!handsOnly) {
    addConnections(group, renderedLeftHand, HAND_CONNECTIONS, 0xa5f3fc, 0.025, project)
    addConnections(group, renderedRightHand, HAND_CONNECTIONS, 0xa5f3fc, 0.025, project)
  }
  if (!handsOnly) addPoints(group, face, 0xf0abfc, 0.012, project, 0.22)
  if (!handsOnly) {
    addPoints(group, [...pose, ...renderedLeftHand, ...renderedRightHand], 0x61e7cb, 0.04, project, 0.7)
  }
  return group
}

function normalizeHand(points: THREE.Vector3[], xOffset: number) {
  const wrist = points[0]
  if (!wrist || wrist.lengthSq() === 0) return []
  const local = points.map((point) => point.clone().sub(wrist))
  const extent = Math.max(...local.map((point) => Math.max(Math.abs(point.x), Math.abs(point.y))), 0.2)
  const scale = 0.8 / extent
  return local.map((point) => new THREE.Vector3(point.x * scale + xOffset, point.y * scale, point.z * scale * 0.25))
}

function applyHandPose(model: THREE.Group, frame: number[]) {
  const bones = new Map<string, THREE.Bone>()
  model.traverse((object) => {
    if (object instanceof THREE.Bone) bones.set(object.name, object)
  })

  for (const [side, start] of [["Left", LEFT_HAND_FEATURE_START], ["Right", RIGHT_HAND_FEATURE_START]] as const) {
    const points = extract3DPoints(frame, start, 21)
    const fingers = [
      ["Index", [5, 6, 7, 8]],
      ["Middle", [9, 10, 11, 12]],
      ["Ring", [13, 14, 15, 16]],
      ["Pinky", [17, 18, 19, 20]],
      ["Thumb", [1, 2, 3, 4]],
    ] as const

    for (const [finger, indexes] of fingers) {
      for (let segment = 0; segment < 3; segment += 1) {
        const from = points[indexes[segment]]
        const to = points[indexes[segment + 1]]
        const bone = bones.get(`${side}Hand${finger}${segment + 1}`)
        if (!from || !to || !bone || (from.lengthSq() === 0 && to.lengthSq() === 0)) continue
        const angle = Math.atan2(to.y - from.y, to.x - from.x)
        const base = typeof bone.userData.signBaseZ === "number" ? bone.userData.signBaseZ : bone.rotation.z
        bone.userData.signBaseZ = base
        bone.rotation.z = base + THREE.MathUtils.clamp(angle, -1.2, 1.2)
      }
    }
  }
}

function extract3DPoints(frame: number[], featureStart: number, count: number) {
  return Array.from({ length: count }, (_, index) => new THREE.Vector3(
    Number.isFinite(frame[featureStart + index * 3]) ? frame[featureStart + index * 3] : 0,
    Number.isFinite(frame[featureStart + index * 3 + 1]) ? frame[featureStart + index * 3 + 1] : 0,
    Number.isFinite(frame[featureStart + index * 3 + 2]) ? frame[featureStart + index * 3 + 2] : 0,
  ))
}

function addConnections(group: THREE.Group, points: THREE.Vector3[], connections: ReadonlyArray<readonly [number, number]>, color: number, width: number, project: (point: THREE.Vector3) => THREE.Vector3) {
  const positions: number[] = []
  for (const [start, end] of connections) {
    const first = points[start]
    const second = points[end]
    if (!first || !second || first.lengthSq() === 0 || second.lengthSq() === 0) continue
    positions.push(...project(first).toArray(), ...project(second).toArray())
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
  group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8, linewidth: width })))
}

function addPoints(group: THREE.Group, points: THREE.Vector3[], color: number, size: number, project: (point: THREE.Vector3) => THREE.Vector3, opacity: number) {
  const positions = points.filter((point) => point.lengthSq() > 0).flatMap((point) => project(point).toArray())
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
  group.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color, size, transparent: true, opacity, sizeAttenuation: true })))
}

