"""Authentication use-cases: turning a raw bearer token into a domain User."""
from __future__ import annotations

from typing import Optional

from app.core.config import Settings
from app.core.errors import AuthError
from app.core.security import AuthenticatedUser, extract_bearer_token, verify_access_token
from app.domain.users.entities import User


async def authenticate_header(
    authorization_header: Optional[str], settings: Settings
) -> Optional[AuthenticatedUser]:
    """No header -> None (guest). Present but invalid -> AuthError."""
    token = extract_bearer_token(authorization_header)
    if token is None:
        return None
    return await verify_access_token(token, settings)


def to_domain_user(user: AuthenticatedUser) -> User:
    return User(id=user.id, email=user.email)


def require_user(user: Optional[AuthenticatedUser]) -> AuthenticatedUser:
    if user is None:
        raise AuthError("Authentication required.")
    return user
