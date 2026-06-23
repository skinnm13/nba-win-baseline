import type { CompletedGame } from "./types";

const FEATURE_COLS = [
  "HOME_WIN_RATE_L10",
  "HOME_MARGIN_L10",
  "HOME_REST_DAYS",
  "AWAY_WIN_RATE_L10",
  "AWAY_MARGIN_L10",
  "AWAY_REST_DAYS",
] as const;

interface TeamLongRow {
  GAME_ID: string;
  GAME_DATE: string;
  TEAM_ID: number;
  IS_HOME: number;
  WIN: number;
  PF: number;
  PA: number;
}

function parseDate(s: string): number {
  return new Date(`${s}T12:00:00`).getTime();
}

function teamLongTable(games: CompletedGame[]): TeamLongRow[] {
  const rows: TeamLongRow[] = [];
  for (const g of games) {
    rows.push({
      GAME_ID: g.GAME_ID,
      GAME_DATE: g.GAME_DATE,
      TEAM_ID: g.HOME_TEAM_ID,
      IS_HOME: 1,
      WIN: g.HOME_WIN,
      PF: g.HOME_PTS,
      PA: g.AWAY_PTS,
    });
    rows.push({
      GAME_ID: g.GAME_ID,
      GAME_DATE: g.GAME_DATE,
      TEAM_ID: g.AWAY_TEAM_ID,
      IS_HOME: 0,
      WIN: 1 - g.HOME_WIN,
      PF: g.AWAY_PTS,
      PA: g.HOME_PTS,
    });
  }
  rows.sort((a, b) => {
    if (a.TEAM_ID !== b.TEAM_ID) return a.TEAM_ID - b.TEAM_ID;
    const da = parseDate(a.GAME_DATE);
    const db = parseDate(b.GAME_DATE);
    if (da !== db) return da - db;
    return a.GAME_ID.localeCompare(b.GAME_ID);
  });
  return rows;
}

function rollingMean(values: (number | null)[], window: number, minPeriods: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const slice = values.slice(Math.max(0, i - window + 1), i + 1).filter((v): v is number => v !== null);
    if (slice.length < minPeriods) {
      out.push(NaN);
    } else {
      out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
  }
  return out;
}

function groupRoll(long: TeamLongRow[], window = 10, minPeriods = 3): Map<string, Record<string, number>> {
  const byGame = new Map<string, Record<string, number>>();
  const groups = new Map<number, TeamLongRow[]>();
  for (const row of long) {
    const g = groups.get(row.TEAM_ID) ?? [];
    g.push(row);
    groups.set(row.TEAM_ID, g);
  }

  for (const teamRows of groups.values()) {
    const wins = teamRows.map((r) => r.WIN);
    const shiftedWins: (number | null)[] = [null, ...wins.slice(0, -1)];
    const margins: (number | null)[] = [null, ...teamRows.slice(0, -1).map((r) => r.PF - r.PA)];

    const winRate = rollingMean(shiftedWins, window, minPeriods);
    const margin = rollingMean(margins, window, minPeriods);

    for (let i = 0; i < teamRows.length; i++) {
      const row = teamRows[i];
      const restDays = i === 0 ? NaN : (parseDate(row.GAME_DATE) - parseDate(teamRows[i - 1].GAME_DATE)) / 86400000;

      const prefix = row.IS_HOME === 1 ? "HOME" : "AWAY";
      const existing = byGame.get(row.GAME_ID) ?? {};
      existing[`${prefix}_WIN_RATE_L10`] = Number.isNaN(winRate[i]) ? 0 : winRate[i];
      existing[`${prefix}_MARGIN_L10`] = Number.isNaN(margin[i]) ? 0 : margin[i];
      existing[`${prefix}_REST_DAYS`] = Number.isNaN(restDays) ? 0 : restDays;
      byGame.set(row.GAME_ID, existing);
    }
  }
  return byGame;
}

export function featureVectorForGame(games: CompletedGame[], gameId: string): number[] {
  const long = teamLongTable(games);
  const feats = groupRoll(long);
  const row = feats.get(gameId);
  if (!row) throw new Error("Features missing for game");
  return FEATURE_COLS.map((c) => row[c] ?? 0);
}

export function featureColumns(): readonly string[] {
  return FEATURE_COLS;
}
