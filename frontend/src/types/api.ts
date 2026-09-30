// Mirrors backend/app/schemas/*. tests/contract/test_frontend_contract.py
// fails if a backend response field is missing here.

export type Direction = "sign_to_text" | "text_to_sign";
/** Never present these alike in the UI: a model result, an avatar result and a failure are different things. */
export type ResultType = "model" | "avatar_gloss" | "failed";

export interface ErrorBody {
  code: string;
  message: string;
}

export interface HealthResponse {
  status: string;
  model_promoted: boolean;
  model_bundle_found: boolean;
  model_loaded: boolean;
}

export interface CapabilitiesResponse {
  api_version: string;
  websocket_path: string;
  auth_enabled: boolean;
  auth_required: boolean;
  history_enabled: boolean;
  sign_to_text_available: boolean;
  sign_to_text_unavailable_reason: string | null;
  text_to_sign_mode: string;
  avatar_vocabulary_size: number;
  landmark_count: number;
  max_frames_per_sequence: number;
}

export interface ModelInfo {
  id: string;
  direction: string;
  promoted: boolean;
  loaded: boolean;
  evaluation_status?: string | null;
  metrics: Record<string, number>;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface MeResponse {
  id: string;
  email?: string | null;
}

export interface AvatarGlossRequest {
  text: string;
  save_history?: boolean;
}

export interface AvatarGlossResponse {
  gloss: string[];
  unmatched_words: string[];
  source: string;
  history_id?: string | null;
}

export interface AvatarVocabularyResponse {
  gloss_ids: string[];
}

export interface LandmarkFramePayload {
  landmarks: number[][]; // 543 x [x, y, z]
}

export interface SignToTextRequest {
  frames: LandmarkFramePayload[];
  min_motion?: number | null;
  min_confidence?: number | null;
  save_history?: boolean;
}

export interface SignToTextResponse {
  text: string;
  accepted: boolean;
  reason: string;
  confidence: number;
  motion_energy: number;
  result_type: ResultType;
  history_id?: string | null;
}

export interface HistoryItem {
  id: string;
  direction: Direction;
  input_text?: string | null;
  output_text: string;
  confidence?: number | null;
  model_version: string;
  result_type: ResultType;
  created_at: string;
}

export interface HistoryListResponse {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface DeletedResponse {
  deleted: number;
}
