import { getJson } from "@/lib/api";

export type TeamMetadata = {
  name: string;
  confederation?: string | null;
  match_count?: number | null;
  first_match_date?: string | null;
  last_match_date?: string | null;
};

export type TeamsPayload = {
  teams: TeamMetadata[];
  source: "real" | "synthetic" | "static" | "missing" | "unavailable" | string;
};

export async function loadTeams(): Promise<TeamsPayload> {
  try {
    return normalizeTeamsPayload(await getJson<TeamsPayload>("/teams"));
  } catch {
    return loadStaticTeams();
  }
}

export function formatTeamSource(source: string): string {
  switch (source) {
    case "loading":
      return "loading";
    case "real":
      return "API real dataset";
    case "synthetic":
      return "API synthetic demo";
    case "static":
      return "static exported report";
    case "unavailable":
      return "unavailable";
    case "missing":
      return "missing";
    default:
      return source;
  }
}

async function loadStaticTeams(): Promise<TeamsPayload> {
  try {
    const response = await fetch("/reports/teams_real.json", { cache: "no-store" });
    if (!response.ok) return { source: "unavailable", teams: [] };
    const payload = (await response.json()) as TeamsPayload;
    return { ...normalizeTeamsPayload(payload), source: payload.source === "real" ? "static" : payload.source };
  } catch {
    return { source: "unavailable", teams: [] };
  }
}

export function normalizeTeamsPayload(payload: TeamsPayload): TeamsPayload {
  const byName = new Map<string, TeamMetadata>();
  for (const team of payload.teams ?? []) {
    const name = String(team.name ?? "").trim();
    if (!name) continue;
    const existing = byName.get(name);
    const matchCount = Number(team.match_count ?? 0);
    if (!existing || matchCount > Number(existing.match_count ?? 0)) {
      byName.set(name, {
        name,
        confederation: team.confederation || "UNKNOWN",
        match_count: Number.isFinite(matchCount) ? matchCount : null,
        first_match_date: team.first_match_date || null,
        last_match_date: team.last_match_date || null
      });
    }
  }
  return {
    source: payload.source,
    teams: Array.from(byName.values()).sort((a, b) => a.name.localeCompare(b.name))
  };
}
