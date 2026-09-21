import { useCallback, useEffect, useRef, useState } from "react"
import { getSupabase } from "../contexts/AuthContext"

export type ConnectionState = "connecting" | "connected" | "reconnecting" | "disconnected"

type ServerMessage =
  | { type: "translation.text"; session_id: string; text: string; latency_ms: number; confidence?: number; is_fallback?: boolean; gated_reason?: string | null }
  | { type: "translation.pose"; session_id: string; gloss: string; frames: number[][]; fps: number; latency_ms: number; is_fallback?: boolean }
  | { type: "error"; session_id: string; message: string; code: string }

interface TranslationSocketOptions {
  onMessage: (message: ServerMessage) => void
}

function isLocalHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]"
}

export function resolveWebSocketEndpoint(
  configured: string | undefined,
  location: Pick<Location, "protocol" | "hostname">,
): string {
  const endpoint = configured?.trim()
  if (endpoint) return endpoint
  if (!isLocalHost(location.hostname)) return ""
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:8000/api/v1/ws/translate`
}

const baseEndpoint = resolveWebSocketEndpoint(
  import.meta.env.VITE_SIGNSAGE_WS_URL,
  window.location,
)

// Reconnect with capped exponential backoff rather than giving up after the
// first drop -- a backend restart or brief network blip should not require
// a full page reload to keep translating.
const RECONNECT_BASE_DELAY_MS = 1000
const RECONNECT_MAX_DELAY_MS = 15000

function sessionId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `signsage-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

async function getAuthToken(): Promise<string | null> {
  try {
    const { data } = await getSupabase().auth.getSession()
    return data.session?.access_token ?? null
  } catch { return null }
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
    let cancelled = false
    let socket: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let attempt = 0
    let hasConnectedOnce = false

    if (!baseEndpoint) {
      setConnectionState("disconnected")
      onMessageRef.current({
        type: "error",
        session_id: sessionIdRef.current,
        code: "MISSING_WEBSOCKET_URL",
        message: "Set VITE_SIGNSAGE_WS_URL to the deployed backend WebSocket URL.",
      })
      return () => { cancelled = true }
    }

    function scheduleReconnect() {
      if (cancelled) return
      setConnectionState(hasConnectedOnce ? "reconnecting" : "connecting")
      const delay = Math.min(RECONNECT_BASE_DELAY_MS * 2 ** attempt, RECONNECT_MAX_DELAY_MS)
      attempt += 1
      reconnectTimer = setTimeout(() => { void connect() }, delay)
    }

    async function connect() {
      if (cancelled) return
      const token = await getAuthToken()
      const url = token ? `${baseEndpoint}${baseEndpoint.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : baseEndpoint
      if (cancelled) return
      socket = new WebSocket(url)
      socketRef.current = socket

      socket.addEventListener("open", () => {
        attempt = 0
        hasConnectedOnce = true
        setConnectionState("connected")
      })
      socket.addEventListener("close", () => {
        if (socketRef.current === socket) socketRef.current = null
        if (!cancelled) scheduleReconnect()
      })
      socket.addEventListener("error", () => {
        // "close" always follows "error" for a WebSocket, so reconnection
        // is scheduled there -- this only updates the visible state.
        setConnectionState(hasConnectedOnce ? "reconnecting" : "connecting")
      })
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
    }
    void connect()

    return () => {
      cancelled = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      socket?.close()
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
