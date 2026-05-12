"""Upcoming games from stats.nba.com via ScoreboardV2 (requires network)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from src.fetch_games import _canonical_nba_season_label


@dataclass(frozen=True)
class ScheduledGame:
    game_id: str
    game_date: date
    away_abbr: str
    home_abbr: str
    game_status: int
    status_text: str

    @property
    def season_id(self) -> str:
        return _canonical_nba_season_label(pd.Timestamp(self.game_date))


def _team_id_to_abbr() -> dict[int, str]:
    from nba_api.stats.static import teams as nba_teams

    return {int(t["id"]): str(t["abbreviation"]) for t in nba_teams.get_teams()}


def _scheduled_from_day(game_date: date, *, include_live: bool, id_to_abbr: dict[int, str]) -> list[ScheduledGame]:
    from nba_api.stats.endpoints import scoreboardv2

    warnings.filterwarnings(
        "ignore",
        message=".*ScoreboardV2.*",
        category=DeprecationWarning,
    )
    mmddyyyy = game_date.strftime("%m/%d/%Y")
    sb = scoreboardv2.ScoreboardV2(game_date=mmddyyyy)
    gh = sb.get_data_frames()[0]
    if gh.empty:
        return []

    allowed = {1}
    if include_live:
        allowed.add(2)

    out: list[ScheduledGame] = []
    for _, row in gh.iterrows():
        st = int(row["GAME_STATUS_ID"])
        if st not in allowed:
            continue
        hid = int(row["HOME_TEAM_ID"])
        vid = int(row["VISITOR_TEAM_ID"])
        home_abbr = id_to_abbr.get(hid)
        away_abbr = id_to_abbr.get(vid)
        if not home_abbr or not away_abbr:
            continue
        out.append(
            ScheduledGame(
                game_id=str(row["GAME_ID"]),
                game_date=game_date,
                away_abbr=str(away_abbr).strip().upper(),
                home_abbr=str(home_abbr).strip().upper(),
                game_status=st,
                status_text=str(row.get("GAME_STATUS_TEXT") or "").strip(),
            )
        )
    return out


def upcoming_games(
    *,
    start: date | None = None,
    days: int = 7,
    include_live: bool = False,
) -> list[ScheduledGame]:
    """Return scheduled (not final) games for each calendar day in the window.

    ``days`` is the number of consecutive calendar days beginning at ``start``
    (default: today). ``include_live`` adds in-progress games (status 2).

    Uses ``ScoreboardV2`` so home and visitor team IDs match ``nba_api`` static
    rosters (same numeric IDs as in ``data/games.csv`` when fetched with
    ``--source nba``).
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    d0 = start or date.today()
    id_to_abbr = _team_id_to_abbr()
    seen: set[str] = set()
    merged: list[ScheduledGame] = []
    for i in range(days):
        d = d0 + timedelta(days=i)
        for sg in _scheduled_from_day(d, include_live=include_live, id_to_abbr=id_to_abbr):
            if sg.game_id in seen:
                continue
            seen.add(sg.game_id)
            merged.append(sg)
    merged.sort(key=lambda x: (x.game_date, x.game_id))
    return merged


def format_game_lines(games: Iterable[ScheduledGame]) -> list[str]:
    lines: list[str] = []
    for g in games:
        when = g.game_date.isoformat()
        lines.append(f"{when}  {g.away_abbr} @ {g.home_abbr}  ({g.status_text or 'Scheduled'})")
    return lines
