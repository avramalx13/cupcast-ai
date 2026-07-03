import type { PredictionResponse, SelectedMatchup } from "@/lib/matchup";
import { normalizeTeamName } from "@/lib/matchup";

type DemoMatchupsPayload = {
  source: string;
  matchups: PredictionResponse[];
  warnings?: string[];
};

export async function loadStaticDemoPrediction(matchup: SelectedMatchup): Promise<PredictionResponse | null> {
  try {
    const response = await fetch("/reports/demo_matchups.json", { cache: "no-store" });
    if (!response.ok) return null;
    const payload = (await response.json()) as DemoMatchupsPayload;
    const teamA = normalizeTeamName(matchup.teamA);
    const teamB = normalizeTeamName(matchup.teamB);
    return (
      payload.matchups?.find(
        (item) => normalizeTeamName(item.team_a) === teamA && normalizeTeamName(item.team_b) === teamB
      ) ?? null
    );
  } catch {
    return null;
  }
}
