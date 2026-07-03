from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cupcast.analyst.schemas import MatchPredictionContext, ProbabilityChangeContext

from .schemas import AnalystRequest, AnalystResponse
from .services import get_explanation_service


router = APIRouter(prefix="/analyst", tags=["analyst"])


@router.post("/explain", response_model=AnalystResponse)
def explain(request: AnalystRequest) -> AnalystResponse:
    service = get_explanation_service()
    if request.kind == "match_prediction":
        if (
            request.team_a is None
            or request.team_b is None
            or request.team_a_win_probability is None
            or request.draw_probability is None
            or request.team_b_win_probability is None
        ):
            raise HTTPException(status_code=400, detail="match_prediction requires teams and probabilities")
        text = service.explain_match_prediction(
            MatchPredictionContext(
                team_a=request.team_a,
                team_b=request.team_b,
                team_a_win_probability=request.team_a_win_probability,
                draw_probability=request.draw_probability,
                team_b_win_probability=request.team_b_win_probability,
            ),
            use_mini_llm=request.use_mini_llm,
        )
        return AnalystResponse(explanation=text)

    if request.kind == "probability_change":
        if request.event is None or request.probability_changes is None:
            raise HTTPException(status_code=400, detail="probability_change requires event and probability_changes")
        text = service.explain_probability_change(
            ProbabilityChangeContext(
                event=request.event,
                probability_changes=request.probability_changes,
            ),
            use_mini_llm=request.use_mini_llm,
        )
        return AnalystResponse(explanation=text)

    raise HTTPException(status_code=400, detail=f"Unsupported analyst kind: {request.kind}")
