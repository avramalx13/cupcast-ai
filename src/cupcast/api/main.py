from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes_analyst import router as analyst_router
from .routes_backtesting import router as backtesting_router
from .routes_data import router as data_router
from .routes_data import teams_router
from .routes_predictions import router as predictions_router
from .routes_simulation import router as simulation_router


app = FastAPI(
    title="CupCast AI",
    description="World Cup prediction engine with a custom mini-LLM analyst.",
    version="0.1.0",
)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _allowed_cors_origins() -> list[str]:
    configured_origins = os.environ.get("CUPCAST_CORS_ORIGINS")
    if not configured_origins:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
    allow_origin_regex=r"^http://192\.168\.\d{1,3}\.\d{1,3}:3000$",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions_router)
app.include_router(simulation_router)
app.include_router(analyst_router)
app.include_router(backtesting_router)
app.include_router(data_router)
app.include_router(teams_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
