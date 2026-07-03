type Change = {
  team: string;
  before: number;
  after: number;
};

export function ProbabilityTimeline({ changes }: { changes: Change[] }) {
  if (changes.length === 0) {
    return (
      <div className="rounded-lg border border-white bg-white/90 p-6 shadow-sm">
        <div className="text-lg font-black text-ink">Probability timeline</div>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Apply a result update to see which teams gained or lost tournament equity.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {changes.map((change) => (
        <div key={change.team} className="rounded-lg border border-white bg-white/95 p-4 shadow-sm">
          <div className="flex items-center justify-between gap-4">
            <div className="font-black text-ink">{change.team}</div>
            <div className={change.after >= change.before ? "text-sm font-black text-emerald-700" : "text-sm font-black text-[#d9272e]"}>
              {change.after >= change.before ? "+" : ""}
              {((change.after - change.before) * 100).toFixed(1)} pts
            </div>
          </div>
          <div className="mt-3 grid gap-2">
            <TimelineBar label="Before" value={change.before} />
            <TimelineBar label="After" value={change.after} strong />
          </div>
        </div>
      ))}
    </div>
  );
}

function TimelineBar({ label, value, strong = false }: { label: string; value: number; strong?: boolean }) {
  const percent = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="grid gap-1">
      <div className="flex justify-between text-xs font-bold uppercase text-slate-500">
        <span>{label}</span>
        <span>{percent.toFixed(1)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className={strong ? "h-full rounded-full bg-emerald-600" : "h-full rounded-full bg-slate-300"} style={{ width: `${Math.max(2, percent)}%` }} />
      </div>
    </div>
  );
}
