"use client";

import { useState } from "react";
import { Play, RefreshCcw, Trophy, Zap } from "lucide-react";
import { AppShell, PageHero } from "@/components/AppShell";
import { ProbabilityTimeline } from "@/components/ProbabilityTimeline";
import { SimulationTable } from "@/components/SimulationTable";
import { postJson } from "@/lib/api";

type SimulationRow = {
  team: string;
  reach_final_probability: number;
  win_tournament_probability: number;
};

type SimulationResponse = {
  results: SimulationRow[];
};

type UpdateResponse = {
  event: string;
  probability_changes: Record<string, { before: number; after: number }>;
};

export default function SimulatorPage() {
  const [rows, setRows] = useState<SimulationRow[]>([]);
  const [changes, setChanges] = useState<{ team: string; before: number; after: number }[]>([]);
  const [event, setEvent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runSimulation() {
    setLoading(true);
    setError("");
    try {
      const response = await postJson<SimulationResponse>("/simulation/run", { simulations: 2000 });
      setRows(response.results.slice(0, 10));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unable to run simulation");
    } finally {
      setLoading(false);
    }
  }

  async function runUpdateExample() {
    setLoading(true);
    setError("");
    try {
      const response = await postJson<UpdateResponse>("/matches/update-result", {
        team_a: "Germany",
        team_b: "Paraguay",
        score_a: 1,
        score_b: 1,
        penalty_winner: "Paraguay",
        simulations: 1000
      });
      setEvent(response.event);
      setChanges(
        Object.entries(response.probability_changes).map(([team, change]) => ({
          team,
          before: change.before,
          after: change.after
        }))
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unable to update result");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell>
      <PageHero
        title="Tournament Simulator"
        eyebrow="Monte Carlo tournament paths"
        description="Run repeat simulations, then apply a result update to see title odds move across the bracket."
        icon={Trophy}
        aside={
          <div>
            <Zap aria-hidden className="h-5 w-5 text-emerald-200" />
            <div className="mt-3 text-2xl font-black text-white">2,000 runs</div>
            <p className="mt-2 text-sm leading-6 text-white/70">
              The simulator uses structured match probabilities, not LLM guesses.
            </p>
          </div>
        }
      >
        <button
          className="primary-action disabled:translate-y-0 disabled:opacity-60"
          onClick={runSimulation}
          disabled={loading}
        >
          {loading ? "Running" : "Run simulation"}
          <Play aria-hidden className="h-4 w-4" />
        </button>
        <button
          className="secondary-action disabled:translate-y-0 disabled:opacity-60"
          onClick={runUpdateExample}
          disabled={loading}
        >
          Update result
          <RefreshCcw aria-hidden className="h-4 w-4" />
        </button>
      </PageHero>

      <div className="mx-auto grid max-w-7xl gap-6 px-6 pb-12">
      {error && (
        <section className="-mt-8 rounded-lg border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-800 shadow-xl">
          {error}
        </section>
      )}
      <SimulationTable rows={rows} />
      {event && <div className="rounded-lg border border-white bg-white/95 p-5 font-black text-ink shadow-sm">{event}</div>}
      <ProbabilityTimeline changes={changes} />
      </div>
    </AppShell>
  );
}
