export type SelectedMatchup = {
  teamA: string;
  teamB: string;
};

export type PredictionResponse = {
  team_a: string;
  team_b: string;
  team_a_win_probability: number;
  draw_probability: number;
  team_b_win_probability: number;
  explanation?: string | null;
};

export type AnalystRequestPayload = {
  kind: "match_prediction";
  team_a: string;
  team_b: string;
  team_a_win_probability: number;
  draw_probability: number;
  team_b_win_probability: number;
};

export const PREDICTION_UNAVAILABLE =
  "Prediction unavailable for this matchup. Check that both teams exist in the dataset and that the prediction model has been trained.";

export const STATIC_MATCHUP_UNAVAILABLE =
  "Static demo mode can show saved report results, but this exact matchup needs API mode for live prediction.";

export function buildPredictionRequest(matchup: SelectedMatchup) {
  return {
    team_a: matchup.teamA.trim(),
    team_b: matchup.teamB.trim(),
    neutral: true,
    stage: "group"
  };
}

export function buildAnalystRequestFromPrediction(prediction: PredictionResponse): AnalystRequestPayload {
  return {
    kind: "match_prediction",
    team_a: prediction.team_a,
    team_b: prediction.team_b,
    team_a_win_probability: prediction.team_a_win_probability,
    draw_probability: prediction.draw_probability,
    team_b_win_probability: prediction.team_b_win_probability
  };
}

export function sameMatchup(a: SelectedMatchup, b: SelectedMatchup): boolean {
  return normalizeTeamName(a.teamA) === normalizeTeamName(b.teamA) && normalizeTeamName(a.teamB) === normalizeTeamName(b.teamB);
}

export function normalizeTeamName(value: string): string {
  return value.trim().toLocaleLowerCase();
}
