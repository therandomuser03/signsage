import { useEffect, useRef } from "react"

const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [27, 29], [29, 31],
  [24, 26], [26, 28], [28, 30], [30, 32],
]

interface PoseCanvasProps {
  frames: number[][]
  fps: number
}

export function PoseCanvas({ frames, fps }: PoseCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || frames.length === 0) return
    const context = canvas.getContext("2d")
    if (!context) return

    let animationFrame = 0
    let startedAt = 0
    const draw = (timestamp: number) => {
      if (!startedAt) startedAt = timestamp
      const bounds = canvas.getBoundingClientRect()
      const ratio = window.devicePixelRatio || 1
      canvas.width = Math.max(1, Math.round(bounds.width * ratio))
      canvas.height = Math.max(1, Math.round(bounds.height * ratio))
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      context.clearRect(0, 0, bounds.width, bounds.height)

      const frame = frames[Math.floor(((timestamp - startedAt) / 1000) * fps) % frames.length]
      const points = frameToCanvas(frame, bounds.width, bounds.height)
      context.strokeStyle = "#8b5cf6"
      context.lineWidth = 3
      context.lineCap = "round"
      for (const [start, end] of POSE_CONNECTIONS) {
        const first = points[start]
        const second = points[end]
        if (!first || !second) continue
        context.beginPath()
        context.moveTo(first.x, first.y)
        context.lineTo(second.x, second.y)
        context.stroke()
      }
      context.fillStyle = "#e9d5ff"
      for (const point of points) {
        context.beginPath()
        context.arc(point.x, point.y, 3.5, 0, Math.PI * 2)
        context.fill()
      }
      animationFrame = window.requestAnimationFrame(draw)
    }

    animationFrame = window.requestAnimationFrame(draw)
    return () => window.cancelAnimationFrame(animationFrame)
  }, [fps, frames])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" aria-label="Generated pose animation" role="img" />
}

function frameToCanvas(frame: number[], width: number, height: number) {
  const points = Array.from({ length: 33 }, (_, index) => ({
    x: Number.isFinite(frame[index * 3]) ? frame[index * 3] : 0,
    y: Number.isFinite(frame[index * 3 + 1]) ? frame[index * 3 + 1] : 0,
  }))
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const span = Math.max(maxX - minX, maxY - minY, 1e-4)
  const scale = Math.min(width, height) * 0.72 / span
  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2
  return points.map((point) => ({
    x: width / 2 + (point.x - centerX) * scale,
    y: height / 2 + (point.y - centerY) * scale,
  }))
}
