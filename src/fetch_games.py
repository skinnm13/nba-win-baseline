"""Download NBA game results for the baseline pipeline (stats.nba.com or FTE archive)."""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

FTE_NBA_ALLELO_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_fte_games(min_year: int = 2005) -> pd.DataFrame:
    """Load historical games from FiveThirtyEight's public `nbaallelo` dataset.

    Note: the public CSV is not updated past roughly the 2014-15 season (`year_id` 2015).
    For recent seasons, use `--source nba` on a network where stats.nba.com is reachable.
    """
    df = pd.read_csv(FTE_NBA_ALLELO_URL, low_memory=False)
    if "_iscopy" not in df.columns and "iscopy" in df.columns:
        df = df.rename(columns={"iscopy": "_iscopy"})
    g = df[
        (df["lg_id"] == "NBA")
        & (df["is_playoffs"] == 0)
        & (df["_iscopy"] == 0)
        & (df["year_id"] >= min_year)
    ].copy()
    rows: list[dict] = []
    for _, r in g.iterrows():
        loc = str(r["game_location"]).strip().upper()
        pts = int(r["pts"])
        opp_pts = int(r["opp_pts"])
        tid = str(r["team_id"])
        oid = str(r["opp_id"])
        if loc == "H":
            home_id, away_id, home_pts, away_pts = tid, oid, pts, opp_pts
        elif loc == "A":
            home_id, away_id, home_pts, away_pts = oid, tid, opp_pts, pts
        else:
            continue
        home_win = int(home_pts > away_pts)
        rows.append(
            {
                "GAME_ID": str(r["game_id"]),
                "GAME_DATE": pd.to_datetime(r["date_game"], format="%m/%d/%Y", errors="coerce"),
                "SEASON_ID": str(int(r["year_id"])),
                "HOME_TEAM_ID": home_id,
                "AWAY_TEAM_ID": away_id,
                "HOME_ABBR": home_id,
                "AWAY_ABBR": away_id,
                "HOME_PTS": home_pts,
                "AWAY_PTS": away_pts,
                "HOME_WIN": home_win,
                "POINT_DIFF": home_pts - away_pts,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.dropna(subset=["GAME_DATE"])
    out = out.sort_values("GAME_DATE").reset_index(drop=True)
    return out


def _nba_season_start_year_from_label(season: str) -> int:
    return int(str(season).strip().split("-")[0])


def nba_season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


def default_nba_season_through(as_of: date | None = None) -> str:
    """NBA season string covering `as_of` (regular season ~Oct–Apr, playoffs finish by June)."""
    d = as_of or date.today()
    start_year = d.year if d.month >= 10 else d.year - 1
    return nba_season_label(start_year)


def nba_season_labels_inclusive(first: str, last: str) -> list[str]:
    y0 = _nba_season_start_year_from_label(first)
    y1 = _nba_season_start_year_from_label(last)
    if y1 < y0:
        raise ValueError(f"nba season range: last ({last}) is before first ({first})")
    return [nba_season_label(y) for y in range(y0, y1 + 1)]


def fetch_seasons(
    seasons: list[str],
    season_types: list[str],
    sleep_s: float = 0.7,
) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder

    frames: list[pd.DataFrame] = []
    for season in seasons:
        for season_type in season_types:
            resp = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable=season_type,
            )
            frames.append(resp.get_data_frames()[0])
            time.sleep(sleep_s)
    out = pd.concat(frames, ignore_index=True)
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    return out


def _canonical_nba_season_label(game_date: pd.Timestamp) -> str:
    """One label per league year so regular season and playoffs group together (not 22025 vs 42025)."""
    ts = pd.Timestamp(game_date)
    y, m = int(ts.year), int(ts.month)
    start = y if m >= 10 else y - 1
    return f"{start}-{str(start + 1)[2:]}"


def games_to_match_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team) -> one row per GAME_ID with home/away outcome."""
    rows: list[dict] = []
    for gid, g in raw.groupby("GAME_ID"):
        if len(g) != 2:
            continue
        home_mask = g["MATCHUP"].astype(str).str.contains(" vs. ")
        away_mask = g["MATCHUP"].astype(str).str.contains(" @ ")
        if home_mask.sum() != 1 or away_mask.sum() != 1:
            continue
        home_row = g[home_mask].iloc[0]
        away_row = g[away_mask].iloc[0]

        home_pts = int(home_row["PTS"])
        away_pts = int(away_row["PTS"])
        home_win = int(home_pts > away_pts)
        game_date = home_row["GAME_DATE"]
        season_label = _canonical_nba_season_label(game_date)

        rows.append(
            {
                "GAME_ID": gid,
                "GAME_DATE": game_date,
                "SEASON_ID": season_label,
                "HOME_TEAM_ID": int(home_row["TEAM_ID"]),
                "AWAY_TEAM_ID": int(away_row["TEAM_ID"]),
                "HOME_ABBR": str(home_row["TEAM_ABBREVIATION"]),
                "AWAY_ABBR": str(away_row["TEAM_ABBREVIATION"]),
                "HOME_PTS": home_pts,
                "AWAY_PTS": away_pts,
                "HOME_WIN": home_win,
                "POINT_DIFF": home_pts - away_pts,
            }
        )
    df = pd.DataFrame(rows).sort_values("GAME_DATE").reset_index(drop=True)
    return df


def main(
    source: str = "fte",
    min_year: int = 2005,
    *,
    nba_season_from: str = "2017-18",
    nba_season_to: str | None = None,
    nba_include_playoffs: bool = True,
    nba_include_playin: bool = False,
    nba_as_of: date | None = None,
) -> Path:
    out_dir = _project_root() / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / "games.csv"

    if source == "fte":
        games = load_fte_games(min_year=min_year)
        if games.empty:
            raise SystemExit(
                "No games after filters. Lower --min-year. "
                "The public FTE file only contains seasons through about year_id=2015."
            )
        games.to_csv(games_path, index=False)
        y0, y1 = int(games["SEASON_ID"].min()), int(games["SEASON_ID"].max())
        print("Source: FiveThirtyEight nbaallelo (regular season, neutral games skipped)")
        print(f"Filter: year_id>={min_year}  Loaded SEASON_ID range: {y0}..{y1}")
        print(f"Wrote {games_path} ({len(games)} games)")
        return games_path

    if source == "nba":
        as_of = nba_as_of or date.today()
        last = nba_season_to or default_nba_season_through(as_of)
        seasons = nba_season_labels_inclusive(nba_season_from, last)
        season_types = ["Regular Season"]
        if nba_include_playoffs:
            season_types.append("Playoffs")
        if nba_include_playin:
            season_types.append("PlayIn")
        raw = fetch_seasons(seasons, season_types)
        games = games_to_match_rows(raw)
        raw_path = out_dir / "raw_team_games.csv"
        raw.to_csv(raw_path, index=False)
        games.to_csv(games_path, index=False)
        print("Source: stats.nba.com via nba_api")
        print(f"Seasons: {seasons[0]} .. {seasons[-1]} ({len(seasons)} seasons)")
        print(f"Season types: {', '.join(season_types)}")
        print(f"Wrote {raw_path} ({len(raw)} team-rows)")
        print(f"Wrote {games_path} ({len(games)} games)")
        return games_path

    raise ValueError("source must be 'fte' or 'nba'")


def _parse_as_of(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _cli() -> None:
    p = argparse.ArgumentParser(description="Fetch NBA games into ./data/games.csv")
    p.add_argument(
        "--source",
        choices=("fte", "nba"),
        default="fte",
        help="fte: FiveThirtyEight CSV (default). nba: stats.nba.com via nba_api.",
    )
    p.add_argument(
        "--min-year",
        type=int,
        default=2005,
        help="FTE only: minimum year_id season label (public file ends ~2015).",
    )
    p.add_argument(
        "--nba-season-from",
        default="2017-18",
        help="nba: first season label to fetch (default 2017-18).",
    )
    p.add_argument(
        "--nba-season-to",
        default=None,
        help="nba: last season label inclusive (default: season that contains --nba-as-of / today).",
    )
    p.add_argument(
        "--nba-as-of",
        default=None,
        help="nba: YYYY-MM-DD used when --nba-season-to is omitted (default: today).",
    )
    p.add_argument(
        "--no-playoffs",
        action="store_true",
        help="nba: fetch regular season only (default is regular + playoffs).",
    )
    p.add_argument(
        "--playin",
        action="store_true",
        help="nba: also fetch Play-In tournament games.",
    )
    args = p.parse_args()
    nba_as_of = _parse_as_of(args.nba_as_of) if args.nba_as_of else None
    main(
        source=args.source,
        min_year=args.min_year,
        nba_season_from=args.nba_season_from,
        nba_season_to=args.nba_season_to,
        nba_include_playoffs=not args.no_playoffs,
        nba_include_playin=args.playin,
        nba_as_of=nba_as_of,
    )


if __name__ == "__main__":
    _cli()
