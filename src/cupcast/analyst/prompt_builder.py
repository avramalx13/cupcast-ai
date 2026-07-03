from __future__ import annotations

from .schemas import MatchPredictionContext, ProbabilityChangeContext


def build_match_prompt(context: MatchPredictionContext) -> str:
    return (
        "### Context:\n"
        f"Team A: {context.team_a}\n"
        f"Team B: {context.team_b}\n"
        f"{context.team_a} win probability: {context.team_a_win_probability:.1%}\n"
        f"Draw probability: {context.draw_probability:.1%}\n"
        f"{context.team_b} win probability: {context.team_b_win_probability:.1%}\n\n"
        "### Analysis:\n"
    )


def build_probability_change_prompt(context: ProbabilityChangeContext) -> str:
    lines = ["### Context:", f"Event: {context.event}"]
    for team, change in context.probability_changes.items():
        lines.append(f"{team} title probability before: {change['before']:.1%}")
        lines.append(f"{team} title probability after: {change['after']:.1%}")
    lines.extend(["", "### Analysis:"])
    return "\n".join(lines)
