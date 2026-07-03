import Link from "next/link";
import fs from "node:fs";
import path from "node:path";
import { Activity, BarChart3, Bot, ChevronRight, Database, LineChart, ShieldCheck, Sparkles, Trophy, Zap } from "lucide-react";
import { BrandDisclaimer } from "@/components/BrandDisclaimer";
import { BrandShowcase } from "@/components/BrandShowcase";
import { WorldCupMark } from "@/components/WorldCupMark";

const cards = [
  {
    href: "/predictions",
    label: "Match Predictor",
    text: "Compare two national teams with calibrated pre-match probabilities.",
    icon: Zap
  },
  {
    href: "/simulator",
    label: "Tournament Simulator",
    text: "Run tournament paths and watch title odds move after results.",
    icon: Trophy
  },
  {
    href: "/full-tournament",
    label: "Full Tournament",
    text: "Simulate a 48-team tournament from groups through the final.",
    icon: LineChart
  },
  {
    href: "/analyst",
    label: "LLM Analyst",
    text: "Turn model outputs into readable football reasoning.",
    icon: Bot
  },
  {
    href: "/backtesting",
    label: "Backtesting",
    text: "Show how the engine performed on historical World Cups.",
    icon: BarChart3
  },
  {
    href: "/reports/model_leaderboard.json",
    label: "Model Reports",
    text: "Open generated leaderboard, backtest, feature, and ablation artifacts.",
    icon: Database
  }
];

const signalTiles = [
  {
    title: "Match Strength",
    text: "Calibrated three-way probabilities for national-team matchups.",
    icon: Zap,
    accent: "bg-[#0f8a5f]"
  },
  {
    title: "Group Simulation",
    text: "Monte Carlo paths from group stage through the final.",
    icon: Trophy,
    accent: "bg-[#d9272e]"
  },
  {
    title: "Analyst Layer",
    text: "Text explains model outputs; it does not pick winners.",
    icon: Sparkles,
    accent: "bg-[#2467d5]"
  }
];

export default function Home() {
  const summary = readDatasetSummary();
  const stats = [
    { label: "Historical matches", value: summary.matches },
    { label: "Teams in latest report", value: summary.teams },
    { label: "Tournament backtests", value: "2014-2022" }
  ];

  return (
    <main className="min-h-screen overflow-hidden bg-[#07120d] text-white">
      <section className="relative min-h-[76vh]">
        <div className="home-hero-bg" />
        <div className="home-hero-lines" />
        <div className="absolute inset-x-0 bottom-0 h-28 bg-gradient-to-t from-[#07120d] to-transparent" />
        <div className="relative mx-auto flex min-h-[76vh] max-w-7xl flex-col gap-8 px-4 pb-10 pt-6 sm:px-6 sm:pt-8">
          <nav className="flex flex-wrap items-center justify-center gap-4 sm:justify-between">
            <Link href="/" className="flex items-center gap-3">
              <WorldCupMark compact />
              <span className="hidden text-sm font-bold uppercase text-white/75 sm:inline">
                CupCast AI
              </span>
            </Link>
            <div className="hidden items-center gap-2 rounded-full border border-white/20 bg-black/20 px-3 py-2 text-xs font-semibold text-white/70 backdrop-blur md:flex">
              <ShieldCheck aria-hidden className="h-4 w-4 text-emerald-300" />
              Real-data-ready portfolio build
            </div>
          </nav>

          <div className="mx-auto grid w-full max-w-6xl flex-1 content-center gap-8 text-center">
            <div className="mx-auto max-w-4xl pb-1">
              <div className="mx-auto inline-flex items-center gap-2 rounded-full border border-emerald-300/25 bg-emerald-300/10 px-3 py-2 text-xs font-bold uppercase text-emerald-100">
                <Activity aria-hidden className="h-4 w-4" />
                Football Forecasting Dashboard
              </div>
              <h1 className="mx-auto mt-6 max-w-4xl text-5xl font-black leading-[1.02] text-white text-balance md:text-7xl">
                CupCast AI
              </h1>
              <p className="mx-auto mt-6 max-w-3xl text-lg leading-8 text-white/75">
                CupCast AI combines historical international results, calibrated probability models, Monte Carlo
                tournament runs, and a custom mini-LLM analyst for portfolio-grade football forecasting.
              </p>
              <div className="mx-auto mt-5 max-w-3xl rounded-md border border-amber-200/20 bg-amber-200/10 p-3 text-sm font-semibold leading-6 text-amber-50">
                Static demo mode displays generated reports from the latest pipeline run. Use API mode for live matchup predictions.
              </div>
              <div className="mx-auto mt-4 max-w-2xl">
                <BrandDisclaimer />
              </div>
              <div className="mt-8 flex flex-wrap justify-center gap-3">
                <Link
                  href="/predictions"
                  className="primary-action"
                >
                  Open predictor
                  <ChevronRight aria-hidden className="h-4 w-4" />
                </Link>
                <Link
                  href="/backtesting"
                  className="secondary-action"
                >
                  View backtests
                </Link>
              </div>
            </div>

            <div className="mx-auto grid w-full max-w-5xl gap-4 pb-4 lg:pb-8">
              <BrandShowcase />
              <div className="grid gap-4 sm:grid-cols-3">
                {stats.map((stat) => (
                  <div key={stat.label} className="metric-card p-4 text-center">
                    <div className="text-2xl font-black text-white">{stat.value}</div>
                    <div className="mt-2 text-xs font-semibold uppercase leading-5 text-white/60">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-[#07120d] px-6 pb-12">
        <div className="mx-auto grid max-w-7xl gap-4 text-center md:grid-cols-3">
          {signalTiles.map(({ title, text, icon: Icon, accent }) => (
            <div key={title} className="signal-tile p-5">
              <div className={`mx-auto h-1.5 w-16 rounded-full ${accent}`} />
              <Icon aria-hidden className="mx-auto mt-6 h-7 w-7 text-white" />
              <div className="mt-5 text-lg font-black text-white">{title}</div>
              <p className="mt-3 text-sm leading-6 text-white/62">{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="bg-[#f4f7fb] px-6 py-10 text-ink">
        <div className="mx-auto grid max-w-7xl gap-4 text-center sm:grid-cols-2 lg:grid-cols-6">
          {cards.map(({ href, label, text, icon: Icon }) => (
            <Link
              key={label}
              href={href}
              className="feature-card group p-5"
            >
              <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-emerald-50 text-emerald-700 transition group-hover:bg-emerald-600 group-hover:text-white">
                <Icon aria-hidden className="h-5 w-5" />
              </div>
              <div className="mt-5 font-black text-ink">{label}</div>
              <p className="mt-3 text-sm leading-6 text-slate-600">{text}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="border-t border-line bg-white px-6 py-8 text-ink">
        <div className="mx-auto flex max-w-7xl flex-col items-center gap-4 text-center md:flex-row md:justify-between">
          <div className="max-w-3xl">
            <div className="flex items-center justify-center gap-2 text-sm font-black uppercase text-emerald-700 md:justify-start">
              <Database aria-hidden className="h-4 w-4" />
              Real data path
            </div>
            <p className="mt-2 max-w-3xl text-slate-600">
              The project now supports real international results, validation reports, model comparison, and generated
              portfolio documentation from the same pipeline.
            </p>
          </div>
          <Link
            href="/analyst"
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-full bg-[#102f1d] px-5 py-3 text-sm font-black text-white transition hover:-translate-y-0.5"
          >
            Ask the analyst
            <Bot aria-hidden className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </main>
  );
}

function readDatasetSummary() {
  try {
    const reportPath = path.join(process.cwd(), "..", "models", "dataset_validation_report.json");
    const report = JSON.parse(fs.readFileSync(reportPath, "utf-8")) as {
      number_of_matches?: number;
      number_of_unique_teams?: number;
      source_type?: string;
    };
    return {
      matches: report.number_of_matches ? report.number_of_matches.toLocaleString() : "n/a",
      teams: report.number_of_unique_teams ? report.number_of_unique_teams.toLocaleString() : "n/a",
      source: report.source_type ?? "unknown"
    };
  } catch {
    return { matches: "n/a", teams: "n/a", source: "unknown" };
  }
}
