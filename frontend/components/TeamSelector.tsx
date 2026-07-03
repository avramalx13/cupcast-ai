"use client";

import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import type { TeamMetadata } from "@/lib/teams";

type Props = {
  label: string;
  value: string;
  onChange: (team: string) => void;
  teams: TeamMetadata[];
  disabled?: boolean;
};

export function TeamSelector({ label, value, onChange, teams, disabled = false }: Props) {
  const [query, setQuery] = useState(value);
  const filteredTeams = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const source = needle
      ? teams.filter((team) => team.name.toLocaleLowerCase().includes(needle))
      : teams;
    return source.slice(0, 8);
  }, [query, teams]);

  function selectTeam(name: string) {
    setQuery(name);
    onChange(name);
  }

  return (
    <div className="grid gap-2 text-sm font-bold text-slate-700">
      <label htmlFor={`${label}-team`}>{label}</label>
      <div className="relative">
        <Search aria-hidden className="pointer-events-none absolute left-3 top-3.5 h-4 w-4 text-slate-400" />
        <input
          id={`${label}-team`}
          className="h-12 w-full rounded-md border border-line bg-white pl-9 pr-3 text-ink shadow-sm outline-none ring-emerald-500/20 transition focus:border-emerald-500 focus:ring-4 disabled:bg-slate-100"
          value={query}
          disabled={disabled}
          placeholder="Search national team"
          onChange={(event) => {
            setQuery(event.target.value);
            onChange(event.target.value);
          }}
          onBlur={() => {
            if (value) setQuery(value);
          }}
        />
      </div>
      <div className="min-h-24 rounded-md border border-line bg-white p-2 shadow-sm">
        {filteredTeams.length ? (
          <div className="grid gap-1">
            {filteredTeams.map((team) => (
              <button
                key={team.name}
                type="button"
                className={`flex items-center justify-between gap-3 rounded-md px-3 py-2 text-left transition hover:bg-emerald-50 ${
                  team.name === value ? "bg-emerald-100 text-emerald-950" : "text-slate-700"
                }`}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => selectTeam(team.name)}
              >
                <span className="font-black">{team.name}</span>
                <span className="shrink-0 text-xs font-semibold text-slate-500">
                  {team.confederation || "UNKNOWN"}
                  {team.match_count != null ? ` / ${team.match_count}` : ""}
                </span>
              </button>
            ))}
          </div>
        ) : (
          <div className="px-3 py-2 text-sm font-semibold text-slate-500">
            No teams found. Use API mode or export `teams_real.json`.
          </div>
        )}
      </div>
    </div>
  );
}
