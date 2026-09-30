"""Auth-related schemas. Sign-in/up happen in Supabase Auth (browser side);
this backend only exposes who the verified token belongs to."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class MeResponse(BaseModel):
    id: str
    email: Optional[str] = None
