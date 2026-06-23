import type { ScheduledGame } from "./types";

export function filterSchedule(
  rows: ScheduledGame[],
  days: number,
  includeLive: boolean,
  start = new Date(),
): ScheduledGame[] {
  const d0 = new Date(start.getFullYear(), start.getMonth(), start.getDate());
  const d1 = new Date(d0);
  d1.setDate(d1.getDate() + Math.max(1, days) - 1);

  const seen = new Set<string>();
  const out: ScheduledGame[] = [];

  for (const row of rows) {
    const gid = String(row.GAME_ID ?? "").trim();
    if (!gid || seen.has(gid)) continue;

    const gd = new Date(`${row.GAME_DATE}T12:00:00`);
    if (Number.isNaN(gd.getTime())) continue;
    if (gd < d0 || gd > d1) continue;

    if (row.IS_PREDICTION_ROW !== undefined && row.IS_PREDICTION_ROW === 0) continue;

    const status = row.GAME_STATUS ?? 1;
    if (!includeLive && status === 2) continue;

    const home = String(row.HOME_ABBR ?? "").trim().toUpperCase();
    const away = String(row.AWAY_ABBR ?? "").trim().toUpperCase();
    if (!home || !away) continue;

    seen.add(gid);
    out.push({ ...row, HOME_ABBR: home, AWAY_ABBR: away });
  }

  out.sort((a, b) => {
    const d = a.GAME_DATE.localeCompare(b.GAME_DATE);
    return d !== 0 ? d : a.GAME_ID.localeCompare(b.GAME_ID);
  });
  return out;
}
