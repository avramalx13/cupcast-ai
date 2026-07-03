from __future__ import annotations

from cupcast.analyst.explanation_service import ExplanationService
from cupcast.analyst.schemas import MatchPredictionContext
from cupcast.mini_llm.evaluate import (
    evaluate_checkpoint,
    format_compliance_score,
    repetition_rate,
)


def test_evaluate_checkpoint_handles_missing_checkpoint_gracefully(tmp_path) -> None:
    eval_data = tmp_path / "eval.txt"
    report_path = tmp_path / "report.json"
    eval_data.write_text("### Context:\nTeam A: France\n\n### Analysis:\nTest", encoding="utf-8")

    report = evaluate_checkpoint(
        checkpoint_path=tmp_path / "missing.pt",
        eval_data_path=eval_data,
        output_path=report_path,
    )

    assert report["status"] == "missing_checkpoint"
    assert report_path.exists()


def test_repetition_rate_detects_adjacent_repetition() -> None:
    assert repetition_rate("France France has a chance") > 0.0
    assert repetition_rate("France has a chance") == 0.0


def test_format_compliance_penalizes_certainty_and_unrelated_teams() -> None:
    prompt = "### Context:\nTeam A: France\nTeam B: Brazil\nFrance win probability: 42%\n### Analysis:\n"
    result = format_compliance_score(prompt, "Germany will definitely win.")

    assert result["score"] < 1.0
    assert not result["checks"]["avoids_false_certainty"]
    assert "Germany" in result["unrelated_teams"]


def test_template_fallback_is_used_when_checkpoint_is_missing(tmp_path) -> None:
    service = ExplanationService(mini_llm_checkpoint=tmp_path / "missing.pt")
    context = MatchPredictionContext(
        team_a="France",
        team_b="Brazil",
        team_a_win_probability=0.42,
        draw_probability=0.25,
        team_b_win_probability=0.33,
    )

    explanation = service.explain_match_prediction(context, use_mini_llm=True)

    assert "structured model" in explanation
