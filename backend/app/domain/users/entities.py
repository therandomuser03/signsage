"""User domain entity -- identity only. Credentials live in Supabase Auth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class User:
    id: str
    email: Optional[str] = None
