"""Aggregates all API routers."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import auth, avatar_gloss, capabilities, health, history, image_classification, translation, video_translation, islim_translation, word_translation
from app.api.websocket import translation as translation_ws

router = APIRouter()
for _r in (health, capabilities, auth, history, translation, avatar_gloss, image_classification, video_translation, islim_translation, word_translation):
    router.include_router(_r.router, prefix="/api/v1")
router.include_router(translation_ws.router)  # WebSocket routes are not REST-versioned
