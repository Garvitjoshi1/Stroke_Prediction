from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.config import settings
from apps.api.routers.health import router as health_router
from apps.api.routers.predict import router as predict_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -------------------- CORS --------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Routers --------------------

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/", tags=["Home"])
def home():
    return {
        "application": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "documentation": "/docs",
    }


@app.on_event("startup")
def startup():
    logger.info("=" * 60)
    logger.info("Starting NeuroGuard API...")
    logger.info("=" * 60)


@app.on_event("shutdown")
def shutdown():
    logger.info("=" * 60)
    logger.info("Stopping NeuroGuard API...")
    logger.info("=" * 60)