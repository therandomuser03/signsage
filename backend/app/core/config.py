"""Central application settings.

All environment-driven configuration lives here. Nothing else in the
backend should call os.environ directly -- import Settings (or the
get_settings() dependency) instead. Every field is prefixed SIGNSAGE_ as an
environment variable, e.g. SIGNSAGE_MODEL_BUNDLE_DIR.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SIGNSAGE_", extra="ignore")

    # --- Server ----------------------------------------------------------
    # Note: the process port comes from $PORT (Dockerfile/Render), not here.
    # Accepts a JSON array (Render/blueprint contract, e.g.
    # '["https://app.vercel.app"]'), a single bare origin, or a
    # comma-separated list. NoDecode keeps the raw env string so a bare
    # (non-JSON) origin cannot crash startup with SettingsError.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]
    log_level: str = "INFO"

    # --- Sign->Text model --------------------------------------------------
    # Directory containing the notebook's exported deployment bundle:
    # checkpoint/, tokenizer/, inference/, landmark_contract.json,
    # sign_to_text_deployment.json, token_vocabulary.json,
    # evaluation_results.json. See research/manifests/models/*.yaml for
    # available runs. Left unset by default -- resolve_bundle_dir() raises a
    # clear error rather than silently guessing a path.
    model_bundle_dir: Optional[str] = None

    # Whether this checkpoint has passed the deployment gate and may be
    # served to real users. As of research run 352868924 (2026-09-26) this
    # is NO-GO (WER 1.0046, 0.99% output diversity, top prediction covers
    # 44.84% of test outputs) -- keep this False until a future run's
    # evaluation_results.json genuinely passes. See backend/models/README.md.
    model_promoted: bool = False

    # Local-development escape hatch for the NO-GO gate above: when True,
    # the backend loads and serves the configured checkpoint even though it
    # has not passed the deployment gate. Default False; NEVER enable in
    # production (render.yaml deliberately omits it). Served output is still
    # labelled unpromoted via /models and /capabilities -- this flag changes
    # availability, not quality claims.
    allow_unpromoted_dev: bool = False

    device: str = "cpu"  # set to "cuda" only if a GPU is actually available at serve time

    # --- Heuristic sign fallback (no checkpoint, no promotion gate) -----------
    # Rule-based recognizer for a 5-sign static vocabulary (see
    # ml/inference/heuristic_signs.py). Always reported as
    # "heuristic-fallback" via /capabilities, never as the transformer model.
    fallback_enabled: bool = True

    # Upper bound for one synchronous model inference (REST or WebSocket
    # final). Inference runs in a worker thread that torch cannot cancel, so
    # this only frees the request with a SESSION_ERROR instead of hanging it.
    # Generous on purpose: a hung/overloaded model must fail loudly, never
    # leave a client waiting indefinitely.
    inference_timeout_s: float = 60.0

    # --- Acceptance gates (mirrors the notebook's predict_live_landmarks) --
    min_motion_energy: float = 0.002
    # The notebook's own confidence floor (LIVE_CONFIDENCE_FLOOR) was never
    # exported to the deployment manifest because it was calibrated from too
    # few validation examples to trust (see MODEL_CARD.md). Default to 0.0
    # (no confidence-based rejection) rather than inventing a threshold the
    # notebook itself did not validate.
    min_confidence: float = 0.0

    # --- Session buffering --------------------------------------------------
    max_frames_per_sequence: int = 160  # should match landmark_contract.json max_frames

    # --- Rolling (streaming) inference --------------------------------------
    # Interim translations run on the trailing window every stride frames, so
    # users see partial results without pressing Stop. Values derive from the
    # serving cadence (~10 landmark frames/s) and measured CPU inference cost
    # (~50-70 ms for an 80-frame window): a 2 s cadence costs <5% overhead.
    # Window must stay <= max_frames_per_sequence (the model's own max).
    stream_window_frames: int = 80
    stream_stride_frames: int = 20

    # --- Avatar gloss generation (English->ISL, see services/avatar_gloss_service.py) ---
    # Both optional: unset means that stage is skipped and the deterministic
    # word-lookup fallback is used, which is what actually ships first.
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_seconds: float = 6.0
    # On 429, Groq returns Retry-After; if it's under this, wait once and retry.
    # Above it (or on any other failure), fail fast to Ollama -- this is a live,
    # in-the-loop feature, not a batch job worth blocking a user for.
    groq_max_retry_wait_seconds: float = 3.0

    ollama_base_url: Optional[str] = None
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: float = 6.0

    # Optional override; defaults to the packaged app/data/sign_vocabulary.json.
    avatar_vocabulary_path: Optional[str] = None

    # --- Model labelling / WebSocket limits -----------------------------------
    model_version: str = "signsage-s2t"
    max_message_bytes: int = 262144  # one 543x3 frame is ~30 KB of JSON

    # --- Auth (Supabase Auth owns identity; see docs: Supabase = identity,
    # FastAPI = ML, Postgres = persistent data) ---------------------------------
    supabase_url: Optional[str] = None
    supabase_publishable_key: Optional[str] = None  # the public "anon"/publishable key
    # False (default): the translator works for guests; signed-in users get history.
    auth_required: bool = False
    # Tokens are verified against Supabase and cached briefly to avoid a network
    # round-trip per request. A revoked token may be accepted for up to this long.
    auth_cache_ttl_seconds: float = 30.0

    # --- History ------------------------------------------------------------------
    history_enabled: bool = True
    history_backend: str = "supabase"  # "supabase" | "memory" (dev/tests only)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(o).strip() for o in parsed if str(o).strip()]
            if isinstance(parsed, str) and parsed.strip():
                return [parsed.strip()]
            return [o.strip() for o in s.split(",") if o.strip()]
        return v

    @property
    def auth_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_publishable_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
