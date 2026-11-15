// Mirrors backend/app/schemas/websocket.py (path: /ws/translate).
import type { ResultType } from "../../types/api";

export const LANDMARK_COUNT = 543; // pose 33 + face 468 + left hand 21 + right hand 21
export const COORDINATES_PER_LANDMARK = 3;

export interface AuthMessage { type: "auth"; access_token: string; history_session_id?: string }
export interface LandmarksFrameMessage { type: "landmarks.frame"; landmarks: number[][] }
export interface SentenceBoundaryMessage { type: "sentence.boundary"; save_history?: boolean; client_key?: string }
export type ClientMessage = AuthMessage | LandmarksFrameMessage | SentenceBoundaryMessage;

export interface AuthOkMessage { type: "auth.ok"; user_id: string }
export interface TranslationTextMessage {
  type: "translation.text";
  text: string;
  accepted: boolean;
  reason: string;
  confidence: number;
  result_type: ResultType;
  history_id?: string | null;
  /** False = stabilizing interim from the rolling window (may be superseded);
   *  true = finalized by an explicit sentence.boundary. */
  final: boolean;
}
export interface ErrorMessage { type: "error"; code: string; message: string }
export type ServerMessage = AuthOkMessage | TranslationTextMessage | ErrorMessage;

export const ErrorCode = {
  INVALID_JSON: "INVALID_JSON",
  INVALID_MESSAGE: "INVALID_MESSAGE",
  MESSAGE_TOO_LARGE: "MESSAGE_TOO_LARGE",
  SESSION_ERROR: "SESSION_ERROR",
  EMPTY_SEQUENCE: "EMPTY_SEQUENCE",
  UNKNOWN_MESSAGE_TYPE: "UNKNOWN_MESSAGE_TYPE",
  MODEL_UNAVAILABLE: "MODEL_UNAVAILABLE",
  UNAUTHORIZED: "UNAUTHORIZED",
} as const;

const SERVER_TYPES = new Set(["auth.ok", "translation.text", "error"]);

export function parseServerMessage(data: string): ServerMessage | null {
  try {
    const parsed: unknown = JSON.parse(data);
    if (typeof parsed === "object" && parsed !== null && SERVER_TYPES.has((parsed as { type?: string }).type ?? "")) {
      return parsed as ServerMessage;
    }
  } catch {
    /* ignore malformed frames */
  }
  return null;
}
