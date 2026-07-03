"use client";

import { useEffect, useState } from "react";
import { Activity, ChevronRight, Sparkles, Zap } from "lucide-react";
import { AnalystPanel } from "@/components/AnalystPanel";
import { AppShell, PageHero } from "@/components/AppShell";
import { ProbabilityCard } from "@/components/ProbabilityCard";
import { TeamSelector } from "@/components/TeamSelector";
import { postJson } from "@/lib/api";
import { buildPredictionRequest, PREDICTION_UNAVAILABLE, STATIC_MATCHUP_UNAVAILABLE } from "@/lib/matchup";
import type { PredictionResponse, SelectedMatchup } from "@/lib/matchup";
import { loadStaticDemoPrediction } from "@/lib/staticDemo";
import { formatTeamSource, loadTeams } from "@/lib/teams";
import type { TeamMetadata } from "@/lib/teams";

export default function PredictionsPage() {
  const [matchup, setMatchup] = useState<SelectedMatchup>({ teamA: "France", teamB: "Brazil" });
  const [teams, setTeams] = useState<TeamMetadata[]>([]);
  const [teamSource, setTeamSource] = useState("loading");
  const [result, setResult] = useState<PredictionResponse | null>(null);
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
    setResult(null);
    setError("");
    setModeNote("");
  }, [matchup.teamA, matchup.teamB]);

  async function runPrediction() {
    setLoading(true);
    setError("");
    setModeNote("");
    setResult(null);
    try {
      const response = await postJson<PredictionResponse>("/predict/match", buildPredictionRequest(matchup));
      setResult(response);
    } catch (err: unknown) {
      const staticPrediction = await loadStaticDemoPrediction(matchup);
      if (staticPrediction) {
        setResult(staticPrediction);
        setModeNote("Static Portfolio Demo: showing a saved matchup generated from local reports.");
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
        title="Match Predictor"
        eyebrow="Neutral venue forecast"
        description="Run a national-team matchup, inspect the probabilities, and send the output into the analyst layer."
        icon={Activity}
        aside={
          <div>
            <Sparkles aria-hidden className="h-5 w-5 text-emerald-200" />
            <p className="mt-3 text-sm leading-6 text-white/70">
              The best demo inputs include team strength, venue, and model probabilities before the explanation step.
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
            className="inline-flex min-h-12 items-center justify-center gap-2 self-end rounded-md bg-[#102f1d] px-5 py-3 font-black text-white shadow-lg shadow-emerald-950/15 transition hover:-translate-y-0.5 disabled:translate-y-0 disabled:opacity-60"
            onClick={runPrediction}
            disabled={loading || !matchup.teamA.trim() || !matchup.teamB.trim()}
          >
            {loading ? "Running" : "Predict"}
            <ChevronRight aria-hidden className="h-4 w-4" />
          </button>
          <div className="text-xs font-bold uppercase text-slate-500 md:col-span-3">
            Teams source: {formatTeamSource(teamSource)}. {teams.length ? `${teams.length.toLocaleString()} teams loaded.` : "No teams loaded."}
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

        {result ? (
          <>
            <section className="grid gap-4 md:grid-cols-3">
              <ProbabilityCard label={`${result.team_a} win`} value={result.team_a_win_probability} />
              <ProbabilityCard label="Draw" value={result.draw_probability} />
              <ProbabilityCard label={`${result.team_b} win`} value={result.team_b_win_probability} />
            </section>
            <AnalystPanel explanation={result.explanation ?? "No analyst explanation was returned for this prediction."} />
          </>
        ) : (
          <section className="grid gap-4 md:grid-cols-3">
            {["France", "Brazil", "Argentina"].map((team) => (
              <div key={team} className="feature-card p-5">
                <Zap aria-hidden className="h-5 w-5 text-emerald-600" />
                <div className="mt-4 text-lg font-black text-ink">{team}</div>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Select a matchup above to generate probabilities and analyst text.
                </p>
              </div>
            ))}
          </section>
        )}
      </div>
    </AppShell>
  );
}
