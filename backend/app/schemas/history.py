"""History schemas (api/routes/history.py)."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HistorySentence(BaseModel):
    sequence_number: int
    direction: str
    input_text: Optional[str] = None
    output_text: str
    gloss: Optional[str] = None
    confidence: Optional[float] = None
    model_version: str
    result_type: str
    created_at: datetime


class HistorySessionSummary(BaseModel):
    id: str
    direction: str
    title: Optional[str] = None
    status: str
    created_at: datetime
    ended_at: Optional[datetime] = None
    sentence_count: int
    preview: str
    model_version: str
    result_type: str


class HistorySessionDetail(HistorySessionSummary):
    sentences: List[HistorySentence]


class HistoryListResponse(BaseModel):
    items: List[HistorySessionSummary]
    total: int
    limit: int
    offset: int


class BulkDeleteRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1, max_length=200)


class DeletedResponse(BaseModel):
    deleted: int
