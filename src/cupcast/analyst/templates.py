from __future__ import annotations

from .schemas import MatchPredictionContext, ProbabilityChangeContext, TournamentOddsContext


def explain_match_template(context: MatchPredictionContext) -> str:
    probabilities = [
        (context.team_a, context.team_a_win_probability),
        ("draw", context.draw_probability),
        (context.team_b, context.team_b_win_probability),
    ]
    leader, leader_probability = max(probabilities, key=lambda item: item[1])
    if leader == "draw":
        return (
            f"{context.team_a} vs {context.team_b} projects as balanced. "
            f"The draw probability is {leader_probability:.1%}, which means neither side has a clear model edge."
        )
    other = context.team_b if leader == context.team_a else context.team_a
    edge = abs(context.team_a_win_probability - context.team_b_win_probability)
    strength = "clear" if edge >= 0.12 else "narrow"
    return (
        f"{leader} is a {strength} favorite over {other}. "
        f"The structured model gives {leader} a {leader_probability:.1%} win probability, "
        f"with the draw at {context.draw_probability:.1%}. The LLM is explaining the model output, "
        "not independently choosing a winner."
    )


def explain_tournament_template(context: TournamentOddsContext) -> str:
    if not context.top_teams:
        return "No simulation results are available yet."
    leader = context.top_teams[0]
    leader_name = str(leader["team"])
    title_probability = float(leader["win_tournament_probability"])
    final_probability = float(leader.get("reach_final_probability", 0.0))
    return (
        f"{leader_name} has the strongest simulated tournament path. "
        f"They win the tournament in {title_probability:.1%} of simulations and reach the final in "
        f"{final_probability:.1%}. These odds come from repeated Monte Carlo draws using match-level probabilities."
    )


def explain_probability_change_template(context: ProbabilityChangeContext) -> str:
    if not context.probability_changes:
        return f"{context.event}. The latest simulation did not produce a material title probability change."
    largest_team, change = max(
        context.probability_changes.items(),
        key=lambda item: abs(item[1]["after"] - item[1]["before"]),
    )
    before = change["before"]
    after = change["after"]
    direction = "increased" if after > before else "decreased"
    return (
        f"{context.event}. {largest_team}'s title probability {direction} from "
        f"{before:.1%} to {after:.1%}. The update flow reruns the remaining bracket instead of asking the LLM "
        "to guess a new champion."
    )
