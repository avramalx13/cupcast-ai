type Props = {
  accuracy: number;
  logLoss: number;
  brierScore: number;
  matches: number;
};

export function BacktestMetrics({ accuracy, logLoss, brierScore, matches }: Props) {
  const metrics = [
    ["Accuracy", accuracy.toFixed(3), "Higher is better"],
    ["Log loss", logLoss.toFixed(3), "Lower is better"],
    ["Brier score", brierScore.toFixed(3), "Probability quality"],
    ["Matches", String(matches), "Evaluation sample"]
  ];
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {metrics.map(([label, value, detail]) => (
        <div key={label} className="rounded-lg border border-white bg-white/95 p-5 shadow-sm">
          <div className="text-sm font-bold text-slate-500">{label}</div>
          <div className="mt-2 text-3xl font-black text-ink">{value}</div>
          <div className="mt-3 text-xs font-semibold uppercase text-emerald-700">{detail}</div>
        </div>
      ))}
    </div>
  );
}
