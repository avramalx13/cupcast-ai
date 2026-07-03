"use client";

import { useEffect, useState } from "react";
import { BarChart3, Play, RefreshCcw, ShieldCheck } from "lucide-react";
import { AppShell, PageHero } from "@/components/AppShell";
import { BacktestMetrics } from "@/components/BacktestMetrics";
import { getJson, postJson } from "@/lib/api";

type ModelComparison = {
  model_name: string;
  accuracy: number;
  log_loss: number;
  brier_score: number;
  top_class_accuracy: number;
  expected_calibration_error: number;
  number_of_matches: number;
};

type BacktestResult = {
  models: ModelComparison[];
  test_tournament: string;
  dataset_source?: string;
  train_matches?: number;
  test_matches?: number;
};

type BacktestSummary = {
  status: string;
  result: BacktestResult | null;
};

type BacktestJob = {
  job_id: string;
  status: string;
  error_message?: string | null;
};

export default function BacktestingPage() {
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [job, setJob] = useState<BacktestJob | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadSummary();
  }, []);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const next = await getJson<BacktestJob>(`/backtesting/jobs/${job.job_id}`);
        setJob(next);
        if (next.status === "completed") {
          await loadSummary();
        }
        if (next.status === "failed") {
          setError(next.error_message || "Backtest job failed");
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Unable to poll backtest job");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  async function loadSummary() {
    try {
      setSummary(await getJson<BacktestSummary>("/backtesting/summary"));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unable to load backtest summary");
    }
  }

  async function runBacktest() {
    setError("");
    setLoading(true);
    try {
      const nextJob = await postJson<BacktestJob>("/backtesting/run", {
        train_before: 2022,
        test_tournament: "World Cup 2022"
      });
      setJob(nextJob);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unable to start backtest");
    } finally {
      setLoading(false);
    }
  }

  const models = summary?.result?.models ?? [];
  const best = models[0];

  return (
    <AppShell>
      <PageHero
        title="Backtesting"
        eyebrow="Model evidence"
        description="Compare forecasting models against held-out World Cup history, with calibration and baseline checks visible in the dashboard."
        icon={BarChart3}
        aside={
          <div>
            <ShieldCheck aria-hidden className="h-5 w-5 text-emerald-200" />
            <div className="mt-3 text-2xl font-black text-white">{summary?.result?.dataset_source ?? "real-data"}</div>
            <p className="mt-2 text-sm leading-6 text-white/70">
              The dashboard run uses a fast 2022 held-out split. Full comparison artifacts stay file-backed.
            </p>
          </div>
        }
      >
        <button
          className="primary-action disabled:translate-y-0 disabled:opacity-60"
          onClick={runBacktest}
          disabled={loading || job?.status === "pending" || job?.status === "running"}
        >
          {loading || job?.status === "pending" || job?.status === "running" ? "Running" : "Run backtest"}
          {loading || job?.status === "running" ? <RefreshCcw aria-hidden className="h-4 w-4" /> : <Play aria-hidden className="h-4 w-4" />}
        </button>
      </PageHero>

      <div className="mx-auto grid max-w-7xl gap-6 px-6 pb-12">
      {job && (
        <section className="glass-card -mt-8 p-4 text-sm text-slate-700">
          <span className="font-black text-ink">Job</span> <span className="font-mono">{job.job_id}</span>: {job.status}
          {job.error_message ? ` (${job.error_message})` : ""}
        </section>
      )}
      {error && (
        <section className={job ? "rounded-lg border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-800" : "-mt-8 rounded-lg border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-800 shadow-xl"}>
          {error}
        </section>
      )}
      {best ? (
        <>
          <BacktestMetrics
            accuracy={best.accuracy}
            logLoss={best.log_loss}
            brierScore={best.brier_score}
            matches={best.number_of_matches}
          />
          <section className="grid gap-4 md:grid-cols-3">
            <SummaryPill label="Tournament" value={summary?.result?.test_tournament ?? "World Cup 2022"} />
            <SummaryPill label="Train matches" value={String(summary?.result?.train_matches ?? "-")} />
            <SummaryPill label="Test matches" value={String(summary?.result?.test_matches ?? best.number_of_matches)} />
          </section>
          <ComparisonTable models={models} />
        </>
      ) : (
        <section className="rounded-lg border border-white bg-white/95 p-8 text-center text-slate-700 shadow-sm">
          {error || "No completed backtest result yet"}
        </section>
      )}
      </div>
    </AppShell>
  );
}

function SummaryPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass-card p-5">
      <div className="text-sm font-bold text-slate-500">{label}</div>
      <div className="mt-2 text-2xl font-black text-ink">{value}</div>
    </div>
  );
}

function ComparisonTable({ models }: { models: ModelComparison[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-white bg-white/95 shadow-sm">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="bg-[#102f1d] text-white">
          <tr>
            <th className="px-4 py-3 font-bold">Model</th>
            <th className="px-4 py-3 font-bold">Accuracy</th>
            <th className="px-4 py-3 font-bold">Log loss</th>
            <th className="px-4 py-3 font-bold">Brier</th>
            <th className="px-4 py-3 font-bold">ECE</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.model_name} className="border-t border-line/70">
              <td className="px-4 py-3 font-black text-ink">{formatModelName(model.model_name)}</td>
              <td className="px-4 py-3 font-semibold">{model.accuracy.toFixed(3)}</td>
              <td className="px-4 py-3 font-semibold">{model.log_loss.toFixed(3)}</td>
              <td className="px-4 py-3 font-semibold">{model.brier_score.toFixed(3)}</td>
              <td className="px-4 py-3 font-semibold">{model.expected_calibration_error.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatModelName(value: string) {
  return value.replaceAll("_", " ");
}
