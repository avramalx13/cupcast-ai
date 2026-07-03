"use client";

import { useEffect, useState } from "react";
import { Bot, MessageSquareText, Sparkles } from "lucide-react";
import { AnalystPanel } from "@/components/AnalystPanel";
import { AppShell, PageHero } from "@/components/AppShell";
import { ProbabilityCard } from "@/components/ProbabilityCard";
import { TeamSelector } from "@/components/TeamSelector";
import { postJson } from "@/lib/api";
import {
  buildAnalystRequestFromPrediction,
  buildPredictionRequest,
  PREDICTION_UNAVAILABLE,
  STATIC_MATCHUP_UNAVAILABLE
} from "@/lib/matchup";
import type { PredictionResponse, SelectedMatchup } from "@/lib/matchup";
import { loadStaticDemoPrediction } from "@/lib/staticDemo";
import { formatTeamSource, loadTeams } from "@/lib/teams";
import type { TeamMetadata } from "@/lib/teams";

export default function AnalystPage() {
  const [matchup, setMatchup] = useState<SelectedMatchup>({ teamA: "France", teamB: "Brazil" });
  const [teams, setTeams] = useState<TeamMetadata[]>([]);
  const [teamSource, setTeamSource] = useState("loading");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modeNote, setModeNote] = useState("");

  useEffect(() => {
    let cancelled = false;
    loadTeams().then((payload) => {
      if (cancelled) return;
      setTeams(payload.teams);
      setTeamSource(payload.source);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setPrediction(null);
    setAnswer("");
    setError("");
    setModeNote("");
  }, [matchup.teamA, matchup.teamB]);

  async function explain() {
    setLoading(true);
    setError("");
    setAnswer("");
    setPrediction(null);
    setModeNote("");
    try {
      const predicted = await postJson<PredictionResponse>("/predict/match", buildPredictionRequest(matchup));
      const response = await postJson<{ explanation: string }>(
        "/analyst/explain",
        buildAnalystRequestFromPrediction(predicted)
      );
      setPrediction(predicted);
      setAnswer(response.explanation);
    } catch (err: unknown) {
      const staticPrediction = await loadStaticDemoPrediction(matchup);
      if (staticPrediction) {
        setPrediction(staticPrediction);
        setAnswer(staticExplanation(staticPrediction));
        setModeNote("Static Portfolio Demo: this explanation is generated from a saved matchup report.");
      } else {
        const message = err instanceof Error ? err.message : PREDICTION_UNAVAILABLE;
        setError(message.includes("Unable to reach CupCast API") ? STATIC_MATCHUP_UNAVAILABLE : `${PREDICTION_UNAVAILABLE} ${message}`);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell>
      <PageHero
        title="LLM Analyst"
        eyebrow="Explanation layer"
        description="Select a matchup, run the structured predictor, and generate text from the exact probabilities returned for those teams."
        icon={Bot}
        aside={
          <div>
            <Sparkles aria-hidden className="h-5 w-5 text-emerald-200" />
            <div className="mt-3 text-2xl font-black text-white">Model-aware text</div>
            <p className="mt-2 text-sm leading-6 text-white/70">
              The mini-LLM explains probabilities; it does not independently pick winners.
            </p>
          </div>
        }
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-6 pb-12">
        <section className="form-panel -mt-10 grid gap-4 p-5 md:grid-cols-[1fr_1fr_auto]">
          <TeamSelector
            label="Team A"
            value={matchup.teamA}
            teams={teams}
            onChange={(teamA) => setMatchup((current) => ({ ...current, teamA }))}
          />
          <TeamSelector
            label="Team B"
            value={matchup.teamB}
            teams={teams}
            onChange={(teamB) => setMatchup((current) => ({ ...current, teamB }))}
          />
          <button
            className="inline-flex min-h-12 w-fit items-center gap-2 self-end rounded-md bg-[#102f1d] px-5 py-3 font-black text-white shadow-lg shadow-emerald-950/15 transition hover:-translate-y-0.5 disabled:translate-y-0 disabled:opacity-60"
            onClick={explain}
            disabled={loading || !matchup.teamA.trim() || !matchup.teamB.trim()}
          >
            {loading ? "Explaining" : "Explain"}
            <MessageSquareText aria-hidden className="h-4 w-4" />
          </button>
          <div className="text-xs font-bold uppercase text-slate-500 md:col-span-3">
            Teams source: {formatTeamSource(teamSource)}. Previous explanation clears when the matchup changes.
          </div>
        </section>

        {error && (
          <section className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-800">
            {error}
          </section>
        )}
        {modeNote && (
          <section className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm font-semibold text-amber-900">
            {modeNote}
          </section>
        )}
        {prediction && (
          <section className="grid gap-4 md:grid-cols-3">
            <ProbabilityCard label={`${prediction.team_a} win`} value={prediction.team_a_win_probability} />
            <ProbabilityCard label="Draw" value={prediction.draw_probability} />
            <ProbabilityCard label={`${prediction.team_b} win`} value={prediction.team_b_win_probability} />
          </section>
        )}
        {answer && <AnalystPanel explanation={answer} />}
      </div>
    </AppShell>
  );
}

function staticExplanation(prediction: PredictionResponse): string {
  return `${prediction.team_a} vs ${prediction.team_b} is loaded from a saved static demo matchup. ` +
    `${prediction.team_a} win probability is ${(prediction.team_a_win_probability * 100).toFixed(1)}%, ` +
    `draw is ${(prediction.draw_probability * 100).toFixed(1)}%, and ${prediction.team_b} win probability is ` +
    `${(prediction.team_b_win_probability * 100).toFixed(1)}%. Use API mode for live matchup predictions.`;
}
