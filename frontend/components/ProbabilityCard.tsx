type Props = {
  label: string;
  value: number;
};

export function ProbabilityCard({ label, value }: Props) {
  const percent = Math.round(value * 1000) / 10;

  return (
    <div className="rounded-lg border border-white bg-white/95 p-5 shadow-sm">
      <div className="text-sm font-bold text-slate-500">{label}</div>
      <div className="mt-2 text-4xl font-black text-ink">{percent}%</div>
      <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-emerald-600" style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
    </div>
  );
}
