"""SignSage FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.router import router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.ml.models.registry import get_sign_to_text_pipeline

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model at startup so the first user doesn't wait for the checkpoint
    # load. A failure is logged, not fatal: /health and /capabilities report it.
    current = get_settings()
    if current.model_promoted or current.allow_unpromoted_dev:
        if current.allow_unpromoted_dev and not current.model_promoted:
            logger.warning(
                "[MODEL] SIGNSAGE_ALLOW_UNPROMOTED_DEV is set: serving an "
                "UNPROMOTED checkpoint for local development only. Do not "
                "enable in production."
            )
        try:
            await run_in_threadpool(
                get_sign_to_text_pipeline, current,
                allow_unpromoted=current.allow_unpromoted_dev,
            )
            logger.info("Sign->Text model loaded at startup.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Sign->Text model failed to load at startup: %s", exc)
    yield


app = FastAPI(title="SignSage API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router)
