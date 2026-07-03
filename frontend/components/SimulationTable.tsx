type Row = {
  team: string;
  reach_final_probability: number;
  win_tournament_probability: number;
};

export function SimulationTable({ rows }: { rows: Row[] }) {
  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-emerald-300 bg-white/80 p-8 text-center shadow-sm">
        <div className="text-lg font-black text-ink">No simulation results yet</div>
        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">
          Run a tournament simulation to populate final and title probabilities.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-white bg-white/95 shadow-sm">
      <table className="w-full text-left text-sm">
        <thead className="bg-[#102f1d] text-white">
          <tr>
            <th className="px-4 py-3 font-bold">Team</th>
            <th className="px-4 py-3 font-bold">Reach Final</th>
            <th className="px-4 py-3 font-bold">Win</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.team} className="border-t border-line/70">
              <td className="px-4 py-3 font-medium text-ink">{row.team}</td>
              <td className="px-4 py-3">
                <ProbabilityBar value={row.reach_final_probability} />
              </td>
              <td className="px-4 py-3">
                <ProbabilityBar value={row.win_tournament_probability} strong />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ProbabilityBar({ value, strong = false }: { value: number; strong?: boolean }) {
  const percent = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="grid min-w-36 gap-2">
      <div className="flex items-center justify-between gap-3">
        <span className="font-black text-ink">{percent.toFixed(1)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className={strong ? "h-full rounded-full bg-emerald-600" : "h-full rounded-full bg-[#d9272e]"} style={{ width: `${Math.max(2, percent)}%` }} />
      </div>
    </div>
  );
}
