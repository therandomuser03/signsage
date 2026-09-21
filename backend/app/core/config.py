from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Relative paths are resolved from the backend project directory when possible.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SIGNSAGE_",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "auto"

    model_dir: Path = Path("./weights")
    text_vocab: Path = Path("./vocab/token_vocabulary.json")
    gloss_vocab: Path = Path("./vocab/gloss_vocabulary.json")
    manifest: Path = Path("./signsage_knowledge_manifest.json")

    max_frames: int = 160
    max_tokens: int = 40
    pose_output_frames: int = 60
    max_websocket_frames: int = 300
    max_text_chars: int = 2000
    max_ws_message_bytes: int = 1_000_000

    # Browser sends raw MediaPipe coordinates; backend applies training normalization.
    landmark_input_mode: str = "raw"

    # `NoDecode` is required here: pydantic-settings otherwise tries to JSON-decode
    # any list-typed env var *before* `parse_origins` below ever runs, which raises
    # a hard `SettingsError` for the documented comma-separated format (see
    # .env.example) and crashes the app on startup. NoDecode defers decoding to
    # the field_validator, which is what actually implements the CSV parsing.
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    allow_noface: bool = True

    # Groq is optional; deterministic glossing remains available.
    enable_groq_gloss: bool = True
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"

    # Ollama is optional and local.
    enable_ollama_gloss: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 8.0

    # LLM sentence refinement for Sign→Text (groq→ollama→naive)
    enable_llm_refine: bool = True
    llm_refine_timeout_seconds: float = 8.0

    # Fail-fast rather than pretending a missing checkpoint is healthy.
    require_all_models: bool = True

    # Optional Supabase auth for translator (frontend-enforced by default).
    # When enabled, WebSocket verifies Bearer token against Supabase Auth.
    enable_auth: bool = False
    supabase_url: str | None = None
    supabase_anon_key: str | None = None

    # Functional fallback for collapsed Sign→Text checkpoint (Image 1 "this is why sacrifices.")
    # When true, low-confidence/attractor outputs are replaced by demo lexicon DTW match or honest placeholder.
    enable_demo_fallback: bool = True
    demo_prototypes_dir: Path = Path("./demo_prototypes")
    translation_confidence_threshold: float = 0.45
    demo_match_threshold: float = 6.5

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return value

    @field_validator("landmark_input_mode")
    @classmethod
    def validate_landmark_mode(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"normalized", "raw"}:
            raise ValueError("SIGNSAGE_LANDMARK_INPUT_MODE must be 'normalized' or 'raw'.")
        return value

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        # Prefer cwd, which is the documented `cd backend` execution mode.
        candidate = Path.cwd() / path
        if candidate.exists():
            return candidate
        return Path(__file__).resolve().parents[2] / path

    @property
    def resolved_model_dir(self) -> Path:
        return self.resolve(self.model_dir)

    @property
    def resolved_text_vocab(self) -> Path:
        return self.resolve(self.text_vocab)

    @property
    def resolved_gloss_vocab(self) -> Path:
        return self.resolve(self.gloss_vocab)

    @property
    def resolved_manifest(self) -> Path:
        return self.resolve(self.manifest)


@lru_cache
def get_settings() -> Settings:
    return Settings()
