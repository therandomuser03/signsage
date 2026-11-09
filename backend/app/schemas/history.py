"""History schemas (api/routes/history.py)."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class HistoryItem(BaseModel):
    id: str
    direction: str
    input_text: Optional[str] = None
    output_text: str
    confidence: Optional[float] = None
    model_version: str
    result_type: str
    created_at: datetime
    title: Optional[str] = None


class HistoryListResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    limit: int
    offset: int


class DeletedResponse(BaseModel):
    deleted: int
