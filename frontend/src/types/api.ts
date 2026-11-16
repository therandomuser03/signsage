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
  word_recognition_enabled?: boolean | null;
}

export interface CapabilitiesResponse {
  api_version: string;
  websocket_path: string;
  auth_enabled: boolean;
  auth_required: boolean;
  history_enabled: boolean;
  sign_to_text_available: boolean;
  sign_to_text_unavailable_reason: string | null;
  sign_to_text_source: string;
  fallback_vocabulary: string[];
  text_to_sign_mode: string;
  avatar_vocabulary_size: number;
  landmark_count: number;
  max_frames_per_sequence: number;
  // Word-level recognition (behind SIGNSAGE_WORD_MODE_ENABLED)
  word_recognition_enabled?: boolean | null;
  word_recognition_unavailable_reason?: string | null;
  word_recognition_vocabulary?: string[] | null;
  word_recognition_threshold?: number | null;
  word_recognition_model_version?: string | null;
  word_recognition_label?: string;
  // Server-side Piper voice availability (PART: free server TTS)
  tts_piper_available?: boolean | null;
  tts_piper_reason?: string | null;
}

export interface WordRecognitionResponse {
  word: string | null;
  confidence: number;
  accepted: boolean;
  top3: Array<{ label: string; probability: number }>;
  threshold: number;
  model_version: string;
  status: string; // "ok" | "no_hands" | "disabled" | "error"
}

export interface SentenceRecognitionResponse {
  gloss_sequence: Array<{ gloss: string; confidence: number }>;
  sentence: string;
  method: "llm" | "fallback";
  accepted: boolean;
  model_version: string;
  threshold: number;
  motion_energy: number;
  status: string;
  /** True when the backend persisted this signing to the user's history (signed-in users only). Defaults false when absent. */
  history_saved?: boolean;
  /** Set when the signed-in user's history save failed: "<step>:<http_status>:<pg_code>". Null otherwise. */
  history_error?: string | null;
}

export interface TranscribeResponse {
  text: string;
}

export interface ComposeRequest {
  words: string[];
  confidences?: number[];
  save_history?: boolean;
  /** Interaction-scoped idempotency key: groups sentences of one interaction. */
  client_session_key?: string;
  /** Per-sentence idempotency key: retries of one finalized sentence collapse. */
  client_key?: string;
}

export interface ComposeResponse {
  sentence: string;
  method: string; // "llm" | "fallback"
  /** True when the backend persisted this sentence to the user's history (signed-in users only). Defaults false when absent. */
  history_saved?: boolean;
  /** Set when the signed-in user's history save failed: "<step>:<http_status>:<pg_code>". Null otherwise. */
  history_error?: string | null;
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
  client_session_key?: string;
  client_key?: string;
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
  client_session_key?: string;
  client_key?: string;
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

/** One completed sentence inside a translation session. */
export interface HistorySentence {
  sequence_number: number;
  direction: Direction;
  input_text?: string | null;
  /** Original gloss sequence, when available. */
  gloss?: string | null;
  output_text: string;
  confidence?: number | null;
  model_version: string;
  result_type: ResultType;
  created_at: string;
}

/** One translation interaction. The History page lists these. */
export interface HistorySessionSummary {
  id: string;
  direction: Direction;
  title?: string | null;
  status: string;
  created_at: string;
  ended_at?: string | null;
  /** Number of completed sentences in this session. */
  sentence_count: number;
  /** First sentence of the session (or its title when empty). */
  preview: string;
  /** Model of the first recorded sentence ('' when unknown). */
  model_version: string;
  result_type: ResultType;
}

export interface HistorySessionDetail extends HistorySessionSummary {
  /** All completed sentences, chronologically ordered. */
  sentences: HistorySentence[];
}

export interface HistoryListResponse {
  items: HistorySessionSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface BulkDeleteRequest {
  ids: string[];
}

export interface DeletedResponse {
  deleted: number;
}

export interface HistorySelfcheckStep {
  step: string;
  status?: number | string | null;
  code?: string | null;
}

export interface HistorySelfcheckResponse {
  ok: boolean;
  steps: HistorySelfcheckStep[];
}

/** Static ISL image classification (alphabet letters A-Z or supported word signs). */
export type ImageClassifyTask = "alphabet" | "word";

export interface ImageClassifyRequest {
  image: string; // base64-encoded JPEG/PNG, optionally a data URI
  task: ImageClassifyTask;
  min_confidence?: number | null;
}

export interface ImageClassifyResponse {
  label: string;
  accepted: boolean;
  reason: string; // "ok" | "low_model_confidence"
  confidence: number;
  task: ImageClassifyTask;
  index: number;
  n_classes: number;
  model_version: string;
  input_size: number[]; // [height, width]
}

export interface VideoTranslationResponse {
  text: string;
  accepted: boolean;
  reason: string;
  confidence: number;
  motion_energy: number;
  result_type: ResultType;
  frames_processed: number;
  model_version: string;
  feature_version: string;
}
