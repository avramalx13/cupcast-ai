from __future__ import annotations

from pathlib import Path

import torch

from cupcast.mini_llm.generate import generate_text, load_model
from cupcast.mini_llm.utils import get_device

from .prompt_builder import build_match_prompt, build_probability_change_prompt
from .schemas import MatchPredictionContext, ProbabilityChangeContext, TournamentOddsContext
from .templates import (
    explain_match_template,
    explain_probability_change_template,
    explain_tournament_template,
)


class ExplanationService:
    def __init__(
        self,
        mini_llm_checkpoint: str | Path | None = None,
        min_generated_chars: int = 80,
    ) -> None:
        self.mini_llm_checkpoint = Path(mini_llm_checkpoint) if mini_llm_checkpoint else None
        self.min_generated_chars = min_generated_chars
        self._llm_loaded = False
        self._model = None
        self._tokenizer = None
        self._device: torch.device | None = None

    def explain_match_prediction(
        self,
        context: MatchPredictionContext,
        use_mini_llm: bool = False,
    ) -> str:
        fallback = explain_match_template(context)
        if not use_mini_llm:
            return fallback
        prompt = build_match_prompt(context)
        return self._generate_or_fallback(prompt, fallback)

    def explain_tournament_odds(self, context: TournamentOddsContext) -> str:
        return explain_tournament_template(context)

    def explain_full_tournament_summary(self, summary: dict[str, object]) -> str:
        top_champions = list(summary.get("top_champions", []) or [])
        if not top_champions:
            return "No full tournament simulation results are available yet."
        leader = top_champions[0]
        leader_name = str(leader.get("team", "Unknown"))
        title_probability = float(leader.get("title_probability", 0.0))
        group_predictions = dict(summary.get("group_predictions", {}) or {})
        strongest_group = _strongest_group(group_predictions)
        dark_horses = list(summary.get("dark_horses", []) or [])
        volatile_teams = list(summary.get("volatile_teams", []) or [])
        dark_horse_text = (
            f"{dark_horses[0]['team']} profiles as the main dark horse at "
            f"{float(dark_horses[0]['title_probability']):.1%} title probability"
            if dark_horses
            else "No lower-seeded team crossed the dark-horse threshold in this run"
        )
        volatile_text = (
            f"{volatile_teams[0]['team']} has one of the most uncertain group-stage paths"
            if volatile_teams
            else "Qualification volatility is limited in the available summary"
        )
        group_text = (
            f"Group {strongest_group[0]} looks strongest by combined title probability"
            if strongest_group
            else "Group strength could not be estimated from this summary"
        )
        return (
            f"{leader_name} is the most likely champion in this simulation set at {title_probability:.1%}. "
            f"{group_text}. {dark_horse_text}. {volatile_text}. These are Monte Carlo estimates from the "
            "structured match model, not guarantees. The Round of 32 placement uses a deterministic "
            "third-place approximation rather than an official FIFA matrix."
        )

    def explain_probability_change(
        self,
        context: ProbabilityChangeContext,
        use_mini_llm: bool = False,
    ) -> str:
        fallback = explain_probability_change_template(context)
        if not use_mini_llm:
            return fallback
        prompt = build_probability_change_prompt(context)
        return self._generate_or_fallback(prompt, fallback)

    def _generate_or_fallback(self, prompt: str, fallback: str) -> str:
        if not self._load_llm():
            return fallback
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        generated = generate_text(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            device=self._device,
            max_new_tokens=120,
            temperature=0.5,
            top_k=20,
        )
        analysis = generated.split("### Analysis:", 1)[-1].strip()
        if len(analysis) < self.min_generated_chars or "<unk>" in analysis:
            return fallback
        return analysis

    def _load_llm(self) -> bool:
        if self._llm_loaded:
            return self._model is not None and self._tokenizer is not None
        self._llm_loaded = True
        if self.mini_llm_checkpoint is None or not self.mini_llm_checkpoint.exists():
            return False
        self._device = get_device(None)
        self._model, self._tokenizer = load_model(str(self.mini_llm_checkpoint), self._device)
        return True


def _strongest_group(group_predictions: dict[str, object]) -> tuple[str, float] | None:
    scores = []
    for group_name, rows in group_predictions.items():
        if not isinstance(rows, list):
            continue
        score = sum(float(row.get("title_probability", 0.0)) for row in rows if isinstance(row, dict))
        scores.append((str(group_name), score))
    if not scores:
        return None
    return max(scores, key=lambda item: item[1])
