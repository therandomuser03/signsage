import { useCallback, useEffect, useRef, useState } from "react"

export type ConnectionState = "connecting" | "connected" | "disconnected"

type ServerMessage =
  | { type: "translation.text"; session_id: string; text: string; latency_ms: number }
  | { type: "translation.pose"; session_id: string; gloss: string; frames: number[][]; fps: number; latency_ms: number }
  | { type: "error"; session_id: string; message: string; code: string }

interface TranslationSocketOptions {
  onMessage: (message: ServerMessage) => void
}

const endpoint = import.meta.env.VITE_SIGNSAGE_WS_URL ?? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000/api/v1/ws/translate`

function sessionId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `signsage-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function useTranslationSocket({ onMessage }: TranslationSocketOptions) {
  const socketRef = useRef<WebSocket | null>(null)
  const sessionIdRef = useRef(sessionId())
  const onMessageRef = useRef(onMessage)
  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting")

  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    const socket = new WebSocket(endpoint)
    socketRef.current = socket

    socket.addEventListener("open", () => setConnectionState("connected"))
    socket.addEventListener("close", () => setConnectionState("disconnected"))
    socket.addEventListener("error", () => setConnectionState("disconnected"))
    socket.addEventListener("message", (event) => {
      try {
        onMessageRef.current(JSON.parse(event.data) as ServerMessage)
      } catch {
        onMessageRef.current({
          type: "error",
          session_id: sessionIdRef.current,
          code: "INVALID_SERVER_MESSAGE",
          message: "Backend returned an unreadable message.",
        })
      }
    })

    return () => {
      socket.close()
      socketRef.current = null
    }
  }, [])

  const send = useCallback((message: Record<string, unknown>): boolean => {
    const socket = socketRef.current
    if (socket?.readyState !== WebSocket.OPEN) return false
    socket.send(JSON.stringify({ ...message, session_id: sessionIdRef.current }))
    return true
  }, [])

  return { connectionState, send }
}
