from __future__ import annotations

from fastapi import APIRouter

from .schemas import TeamsResponse
from .services import get_dataset_status, get_team_directory


router = APIRouter(prefix="/data", tags=["data"])
teams_router = APIRouter(tags=["teams"])


@router.get("/status")
def data_status() -> dict[str, object]:
    return get_dataset_status()


@teams_router.get("/teams", response_model=TeamsResponse)
def teams() -> TeamsResponse:
    directory = get_team_directory()
    return TeamsResponse(**directory)
