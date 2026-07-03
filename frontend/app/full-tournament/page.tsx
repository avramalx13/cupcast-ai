"use client";

import { useEffect, useMemo, useState } from "react";
import { Brackets, Play, ShieldAlert, Trophy } from "lucide-react";
import { AnalystPanel } from "@/components/AnalystPanel";
import { AppShell, PageHero } from "@/components/AppShell";
import { getJson, postJson } from "@/lib/api";

type TeamProbability = {
  team: string;
  group: string;
  win_group_probability: number;
  finish_second_probability: number;
  finish_third_probability: number;
  qualify_probability: number;
  best_third_place_probability: number;
  round_of_32_probability: number;
  round_of_16_probability: number;
  quarterfinal_probability: number;
  semifinal_probability: number;
  final_probability: number;
  title_probability: number;
};

type ChampionRow = {
  team: string;
  title_probability: number;
};

type FullTournamentSummary = {
  available?: boolean;
  message?: string;
  simulations?: number;
  seed?: number;
  top_champions?: ChampionRow[];
  group_predictions?: Record<string, TeamProbability[]>;
  team_probabilities?: TeamProbability[];
  analyst_explanation?: string;
  placement_note?: string;
};

const STATIC_MISSING =
  "Full tournament report not available. Run `python scripts/simulate_full_tournament.py --groups data/tournaments/world_cup_2026_groups.yaml --simulations 1000 --seed 42` and `python scripts/export_frontend_reports.py`.";

export default function FullTournamentPage() {
  const [summary, setSummary] = useState<FullTournamentSummary | null>(null);
  const [source, setSource] = useState<"loading" | "api" | "static" | "missing">("loading");
  const [simulations, setSimulations] = useState(1000);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    loadLatest();
  }, []);

  const progressionRows = useMemo(
    () => [...(summary?.team_probabilities ?? [])].sort((a, b) => b.title_probability - a.title_probability),
    [summary]
  );

  async function loadLatest() {
    setError("");
    try {
      const response = await getJson<FullTournamentSummary>("/simulation/full-tournament/latest");
      setSource("api");
      if (response.available === false) {
        const staticSummary = await loadStaticReport();
        if (staticSummary) {
          syncControlsFromSummary(staticSummary);
          setSummary(staticSummary);
        } else {
          syncControlsFromSummary(response);
          setSummary(response);
        }
        return;
      }
      syncControlsFromSummary(response);
      setSummary(response);
    } catch {
      const staticSummary = await loadStaticReport();
      if (staticSummary) {
        setSource("static");
        syncControlsFromSummary(staticSummary);
        setSummary(staticSummary);
      } else {
        setSource("missing");
        setSummary({ available: false, message: STATIC_MISSING });
      }
    }
  }

  async function runSimulation() {
    setLoading(true);
    setError("");
    try {
      await postJson("/simulation/full-tournament/run", {
        groups_path: "data/tournaments/world_cup_2026_groups.yaml",
        simulations,
        seed
      });
      const latest = await getJson<FullTournamentSummary>("/simulation/full-tournament/latest");
      setSource("api");
      syncControlsFromSummary(latest);
      setSummary(latest);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unable to run full tournament simulation");
    } finally {
      setLoading(false);
    }
  }

  function syncControlsFromSummary(nextSummary: FullTournamentSummary) {
    if (typeof nextSummary.simulations === "number") setSimulations(nextSummary.simulations);
    if (typeof nextSummary.seed === "number") setSeed(nextSummary.seed);
  }

  const apiMode = source === "api";

  return (
    <AppShell>
      <PageHero
        title="Full Tournament Simulator"
        eyebrow="Pre-tournament Monte Carlo"
        description="Simulate the 48-team World Cup from the group stage, including third-place qualification and approximate Round of 32 placement."
        icon={Brackets}
        aside={
          <div>
            <Trophy aria-hidden className="h-5 w-5 text-emerald-200" />
            <div className="mt-3 text-2xl font-black text-white">
              {summary?.simulations ? summary.simulations.toLocaleString() : "Cached"} runs
            </div>
            <p className="mt-2 text-sm leading-6 text-white/70">
              Group stage is simulated from scratch with the trained match predictor.
            </p>
          </div>
        }
      />

      <div className="mx-auto grid max-w-7xl gap-6 px-6 pb-12">
        <section className="form-panel -mt-10 grid gap-4 p-5 md:grid-cols-[1fr_1fr_auto]">
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Monte Carlo runs
            <input
              className="h-12 rounded-md border border-line bg-white px-3 text-ink outline-none ring-emerald-500/20 transition focus:border-emerald-500 focus:ring-4"
              type="number"
              min={1}
              max={10000}
              value={simulations}
              onChange={(event) => setSimulations(Number(event.target.value))}
            />
            <span className="text-xs font-semibold leading-5 text-slate-500">
              More runs reduce random noise in close leaderboard spots.
            </span>
          </label>
          <label className="grid gap-2 text-sm font-bold text-slate-700">
            Random seed
            <input
              className="h-12 rounded-md border border-line bg-white px-3 text-ink outline-none ring-emerald-500/20 transition focus:border-emerald-500 focus:ring-4"
              type="number"
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
            />
            <span className="text-xs font-semibold leading-5 text-slate-500">
              Reproducibility value only. It is not the number of teams.
            </span>
          </label>
          <button
            className="inline-flex min-h-12 items-center justify-center gap-2 self-end rounded-md bg-[#102f1d] px-5 py-3 font-black text-white shadow-lg shadow-emerald-950/15 transition hover:-translate-y-0.5 disabled:translate-y-0 disabled:opacity-60"
            onClick={runSimulation}
            disabled={!apiMode || loading}
          >
            {loading ? "Running" : "Run"}
            <Play aria-hidden className="h-4 w-4" />
          </button>
          <div className="text-xs font-bold uppercase text-slate-500 md:col-span-3">
            Tournament field: 48 teams from the Group A-L preset.
            Latest report: {summary?.simulations ? summary.simulations.toLocaleString() : "no"} runs
            {typeof summary?.seed === "number" ? `, seed ${summary.seed}` : ""}.
            Mode: {source}. {apiMode ? "API mode can run a fresh simulation." : "Static mode can only display exported reports."}
          </div>
        </section>

        {error && (
          <section className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-800">
            {error}
          </section>
        )}

        {summary?.message && (
          <section className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm font-semibold text-amber-900">
            {summary.message}
          </section>
        )}

        {summary?.analyst_explanation && <AnalystPanel explanation={summary.analyst_explanation} />}

        <section className="grid gap-4 lg:grid-cols-[0.8fr_1.2fr]">
          <WinnerLeaderboard rows={summary?.top_champions ?? []} />
          <NotesCard note={summary?.placement_note} />
        </section>

        <GroupPredictions groups={summary?.group_predictions ?? {}} />
        <ProgressionTable rows={progressionRows} />
      </div>
    </AppShell>
  );
}

async function loadStaticReport(): Promise<FullTournamentSummary | null> {
  try {
    const response = await fetch("/reports/full_tournament_simulation.json", { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as FullTournamentSummary;
  } catch {
    return null;
  }
}

function WinnerLeaderboard({ rows }: { rows: ChampionRow[] }) {
  return (
    <section className="overflow-hidden rounded-lg border border-white bg-white/95 shadow-sm">
      <div className="border-b border-line bg-[#102f1d] px-5 py-4 text-white">
        <h2 className="text-lg font-black">Tournament Winner Leaderboard</h2>
      </div>
      <div className="divide-y divide-line/70">
        {rows.length ? rows.slice(0, 10).map((row, idx) => (
          <div key={row.team} className="grid grid-cols-[2rem_1fr_7rem] items-center gap-3 px-5 py-3">
            <div className="text-sm font-black text-slate-400">{idx + 1}</div>
            <div className="font-black text-ink">{row.team}</div>
            <div className="text-right font-black text-emerald-700">{pct(row.title_probability)}</div>
          </div>
        )) : (
          <div className="p-5 text-sm font-semibold text-slate-600">No full tournament report loaded.</div>
        )}
      </div>
    </section>
  );
}

function NotesCard({ note }: { note?: string }) {
  return (
    <section className="rounded-lg border border-white bg-white/95 p-5 shadow-sm">
      <div className="flex items-center gap-2 text-sm font-black uppercase text-emerald-700">
        <ShieldAlert aria-hidden className="h-4 w-4" />
        Notes
      </div>
      <ul className="mt-4 grid gap-3 text-sm font-semibold leading-6 text-slate-600">
        <li>Group stage is simulated from scratch.</li>
        <li>The seed only reproduces randomness; changing it can move close teams by a few ranking spots.</li>
        <li>A 1-2 percentage point gap in title probability is not a strong separation at a few thousand runs.</li>
        <li>{note ?? "Knockout bracket placement for third-place teams uses a deterministic approximation unless official matrix support is added."}</li>
        <li>Predictions are probabilistic, not guarantees.</li>
      </ul>
    </section>
  );
}

function GroupPredictions({ groups }: { groups: Record<string, TeamProbability[]> }) {
  const names = Object.keys(groups).sort();
  if (!names.length) {
    return (
      <section className="rounded-lg border border-dashed border-emerald-300 bg-white/80 p-8 text-center shadow-sm">
        <div className="text-lg font-black text-ink">No group predictions yet</div>
        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">
          Run or export the full tournament simulation to populate Group A through Group L.
        </p>
      </section>
    );
  }

  return (
    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {names.map((groupName) => (
        <div key={groupName} className="overflow-hidden rounded-lg border border-white bg-white/95 shadow-sm">
          <div className="bg-[#102f1d] px-4 py-3 text-sm font-black uppercase text-white">
            Group {groupName}
          </div>
          <div className="divide-y divide-line/70">
            {groups[groupName].map((row) => (
              <div key={row.team} className="grid gap-2 px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-black text-ink">{row.team}</div>
                  <div className="text-sm font-black text-emerald-700">{pct(row.qualify_probability)} qualify</div>
                </div>
                <div className="grid grid-cols-4 gap-2 text-xs font-bold uppercase text-slate-500">
                  <span>Win {pct(row.win_group_probability)}</span>
                  <span>2nd {pct(row.finish_second_probability)}</span>
                  <span>3rd {pct(row.finish_third_probability)}</span>
                  <span>Best 3rd {pct(row.best_third_place_probability)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

function ProgressionTable({ rows }: { rows: TeamProbability[] }) {
  return (
    <section className="overflow-hidden rounded-lg border border-white bg-white/95 shadow-sm">
      <div className="border-b border-line bg-[#102f1d] px-5 py-4 text-white">
        <h2 className="text-lg font-black">Round Progression</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[860px] text-left text-sm">
          <thead className="bg-emerald-50 text-emerald-950">
            <tr>
              {["Team", "R32", "R16", "QF", "SF", "Final", "Champion"].map((header) => (
                <th key={header} className="px-4 py-3 font-black">{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 32).map((row) => (
              <tr key={row.team} className="border-t border-line/70">
                <td className="px-4 py-3 font-black text-ink">{row.team}</td>
                <td className="px-4 py-3">{pct(row.round_of_32_probability)}</td>
                <td className="px-4 py-3">{pct(row.round_of_16_probability)}</td>
                <td className="px-4 py-3">{pct(row.quarterfinal_probability)}</td>
                <td className="px-4 py-3">{pct(row.semifinal_probability)}</td>
                <td className="px-4 py-3">{pct(row.final_probability)}</td>
                <td className="px-4 py-3 font-black text-emerald-700">{pct(row.title_probability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function pct(value: number | undefined): string {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}
