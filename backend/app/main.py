from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.core.config import get_settings
from app.services.model_manager import ModelManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("signsage")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    logger.info(
        "Loading SignSage models from %s using device=%s",
        settings.resolved_model_dir,
        settings.device,
    )
    manager = ModelManager.load(settings)
    app.state.model_manager = manager
    logger.info("SignSage models loaded on %s", manager.device)
    logger.info("Sign->Text live input mode: raw MediaPipe landmarks")

    yield


app = FastAPI(
    title="SignSage Backend",
    version="1.0.0",
    description=(
        "FastAPI inference backend for SignSage. The server accepts landmark "
        "sequences and text; raw camera video remains client-side."
    ),
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/")
def root():
    return {
        "service": "signsage-backend",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket": "/api/v1/ws/translate",
    }
