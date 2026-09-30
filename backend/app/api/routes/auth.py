"""GET /api/v1/auth/me -- who does this token belong to?

Sign-up / sign-in / sign-out happen in Supabase Auth on the client; this
backend only verifies tokens it receives."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependencies import get_current_user
from app.core.security import AuthenticatedUser
from app.schemas.auth import MeResponse
from app.services.authentication_service import to_domain_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=MeResponse)
async def me(user: AuthenticatedUser = Depends(get_current_user)) -> MeResponse:
    domain_user = to_domain_user(user)
    return MeResponse(id=domain_user.id, email=domain_user.email)
