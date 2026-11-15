"""English -> ISL gloss contracts (deterministic lookup + constrained LLM mapping)."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AvatarGlossRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    save_history: bool = True  # only takes effect for signed-in users
    # Idempotency/session scope: client_session_key groups sentences of one
    # interaction; client_key dedupes retries of the same finalized sentence.
    client_session_key: Optional[str] = Field(None, max_length=64)
    client_key: Optional[str] = Field(None, max_length=64)


class AvatarGlossResponse(BaseModel):
    gloss: List[str]
    unmatched_words: List[str]
    source: str  # "llm:groq" | "llm:ollama" | "deterministic"
    history_id: Optional[str] = None


class AvatarVocabularyResponse(BaseModel):
    gloss_ids: List[str]
