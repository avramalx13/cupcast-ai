from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cupcast.analyst.schemas import MatchPredictionContext

from .schemas import MatchPredictionRequest, MatchPredictionResponse
from .services import get_explanation_service, get_matches, get_prediction_model, get_teams


router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/match", response_model=MatchPredictionResponse)
def predict_match(request: MatchPredictionRequest) -> MatchPredictionResponse:
    model = get_prediction_model()
    try:
        result = model.predict_match(
            team_a=request.team_a,
            team_b=request.team_b,
            teams=get_teams(),
            matches=get_matches(),
            neutral=request.neutral,
            stage=request.stage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    context = MatchPredictionContext(**result.as_dict())
    explanation = get_explanation_service().explain_match_prediction(context)
    return MatchPredictionResponse(**result.as_dict(), explanation=explanation)
