import type { Team } from "./types";

const byAbbr = new Map<string, Team>();
const byFull = new Map<string, Team>();
const byNick = new Map<string, Team>();

export function loadTeams(teams: Team[]): void {
  byAbbr.clear();
  byFull.clear();
  byNick.clear();
  for (const t of teams) {
    byAbbr.set(t.abbreviation.toUpperCase(), t);
    byFull.set(t.fullName.toLowerCase(), t);
    byNick.set(t.nickname.toLowerCase(), t);
  }
}

export function displayTeam(abbr: string): string {
  const t = byAbbr.get(abbr.toUpperCase());
  return t ? `${t.fullName} (${abbr.toUpperCase()})` : abbr.toUpperCase();
}

export function resolveUserTeam(text: string): string {
  const s = text.trim();
  if (!s) throw new Error("empty team");

  const upper = s.toUpperCase();
  if (byAbbr.has(upper)) return byAbbr.get(upper)!.abbreviation;

  const lower = s.toLowerCase();
  if (byFull.has(lower)) return byFull.get(lower)!.abbreviation;
  if (byNick.has(lower)) return byNick.get(lower)!.abbreviation;

  const partial: Team[] = [];
  for (const t of byAbbr.values()) {
    const fn = t.fullName.toLowerCase();
    const nick = t.nickname.toLowerCase();
    if ((fn.includes(lower) || nick.includes(lower)) && !partial.some((p) => p.abbreviation === t.abbreviation)) {
      partial.push(t);
    }
  }
  if (partial.length === 1) return partial[0].abbreviation;
  if (partial.length > 1) {
    const opts = partial.map((x) => `${x.nickname} (${x.abbreviation})`).join(", ");
    throw new Error(`Ambiguous "${s}"; matches: ${opts}`);
  }

  throw new Error(`Unknown team "${s}"`);
}

export function allTeamsSorted(): Team[] {
  return [...byAbbr.values()].sort((a, b) => a.fullName.localeCompare(b.fullName));
}
