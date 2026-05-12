"""Fetch NBA completed games and future prediction games into separate CSV files.

Outputs:
  data/games.csv              completed games only, safe for training
  data/raw_team_games.csv     raw LeagueGameFinder team rows
  data/prediction_games.csv   future scheduled games only, safe for predict.py
  data/schedule.csv           normalized full schedule, completed + future
  data/raw_schedule.csv       raw ScheduleLeagueV2 frame
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_tz_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse datetimes and strip timezone info so Pandas comparisons never mix tz-aware/tz-naive values."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def _season_start_year(season: str) -> int:
    return int(str(season).strip().split("-")[0])


def nba_season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


def default_nba_season_through(as_of: date | None = None) -> str:
    d = as_of or date.today()
    return nba_season_label(d.year if d.month >= 10 else d.year - 1)


def nba_season_labels_inclusive(first: str, last: str) -> list[str]:
    y0 = _season_start_year(first)
    y1 = _season_start_year(last)
    if y1 < y0:
        raise ValueError(f"last season {last} is before first season {first}")
    return [nba_season_label(y) for y in range(y0, y1 + 1)]


def _canonical_nba_season_label(game_date: pd.Timestamp) -> str:
    ts = pd.Timestamp(game_date)
    start = int(ts.year) if int(ts.month) >= 10 else int(ts.year) - 1
    return nba_season_label(start)


def _game_type_from_game_id(game_id: object) -> str:
    gid = str(game_id)
    if gid.startswith("001"):
        return "Preseason"
    if gid.startswith("002"):
        return "Regular Season"
    if gid.startswith("003"):
        return "All-Star"
    if gid.startswith("004"):
        return "Playoffs"
    if gid.startswith("005"):
        return "Play-In"
    return "Unknown"


def _selected_game_type_mask(game_ids: pd.Series, include_playoffs: bool, include_playin: bool) -> pd.Series:
    gid = game_ids.astype(str)
    mask = gid.str.startswith("002")
    if include_playoffs:
        mask = mask | gid.str.startswith("004")
        if include_playin:
            mask = mask | gid.str.startswith("005")
    return mask


def _fetch_completed_one_season(
    season: str,
    *,
    include_playoffs: bool,
    include_playin: bool,
    sleep_s: float,
) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder

    parts: list[pd.DataFrame] = []

    regular = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    ).get_data_frames()[0]
    parts.append(regular)
    time.sleep(sleep_s)

    if include_playoffs:
        # Blank SeasonType tends to expose postseason rows faster than explicit Playoffs.
        full = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="",
        ).get_data_frames()[0]
        gid = full["GAME_ID"].astype(str)
        mask = gid.str.startswith("004")
        if include_playin:
            mask = mask | gid.str.startswith("005")
        parts.append(full.loc[mask].copy())
        time.sleep(sleep_s)

    out = pd.concat(parts, ignore_index=True)
    out["GAME_DATE"] = _to_tz_naive_datetime(out["GAME_DATE"])
    out["GAME_TYPE"] = out["GAME_ID"].map(_game_type_from_game_id)
    return out.dropna(subset=["GAME_DATE"]).drop_duplicates(subset=["GAME_ID", "TEAM_ID"], ignore_index=True)


def fetch_completed_seasons(
    seasons: list[str],
    *,
    include_playoffs: bool,
    include_playin: bool,
    sleep_s: float = 0.7,
) -> pd.DataFrame:
    frames = [
        _fetch_completed_one_season(
            season,
            include_playoffs=include_playoffs,
            include_playin=include_playin,
            sleep_s=sleep_s,
        )
        for season in seasons
    ]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], ignore_index=True)


def games_to_match_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if raw.empty:
        return pd.DataFrame()

    for gid, g in raw.groupby("GAME_ID"):
        if len(g) != 2:
            continue
        home_mask = g["MATCHUP"].astype(str).str.contains(" vs. ", regex=False)
        away_mask = g["MATCHUP"].astype(str).str.contains(" @ ", regex=False)
        if home_mask.sum() != 1 or away_mask.sum() != 1:
            continue

        home = g[home_mask].iloc[0]
        away = g[away_mask].iloc[0]
        if pd.isna(home.get("PTS")) or pd.isna(away.get("PTS")):
            continue

        home_pts = int(home["PTS"])
        away_pts = int(away["PTS"])
        game_date = pd.Timestamp(home["GAME_DATE"])

        rows.append(
            {
                "GAME_ID": str(gid),
                "GAME_DATE": game_date,
                "SEASON_ID": _canonical_nba_season_label(game_date),
                "GAME_TYPE": _game_type_from_game_id(gid),
                "HOME_TEAM_ID": int(home["TEAM_ID"]),
                "AWAY_TEAM_ID": int(away["TEAM_ID"]),
                "HOME_ABBR": str(home["TEAM_ABBREVIATION"]),
                "AWAY_ABBR": str(away["TEAM_ABBREVIATION"]),
                "HOME_PTS": home_pts,
                "AWAY_PTS": away_pts,
                "HOME_WIN": int(home_pts > away_pts),
                "POINT_DIFF": home_pts - away_pts,
                "HAS_SCORE": 1,
                "IS_COMPLETED": 1,
                "IS_TRAINING_ROW": 1,
                "IS_PREDICTION_ROW": 0,
                "ROW_SOURCE": "completed_results",
            }
        )

    return pd.DataFrame(rows).sort_values("GAME_DATE").reset_index(drop=True)


def _fetch_schedule_one_season(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import scheduleleaguev2

    return scheduleleaguev2.ScheduleLeagueV2(
        league_id="00",
        season=season,
    ).get_data_frames()[0]


def fetch_schedule_seasons(seasons: list[str], sleep_s: float = 0.7) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        df = _fetch_schedule_one_season(season)
        df["REQUESTED_SEASON"] = season
        frames.append(df)
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_schedule(raw: pd.DataFrame, *, include_playoffs: bool, include_playin: bool) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    # Current ScheduleLeagueV2 columns are camelCase/nested-style.
    rename = {
        "gameId": "GAME_ID",
        "gameDateTimeEst": "GAME_DATE",
        "gameDateEst": "GAME_DATE_EST",
        "gameStatusText": "GAME_STATUS_TEXT",
        "gameStatus": "GAME_STATUS",
        "gameLabel": "GAME_LABEL",
        "seriesText": "SERIES_TEXT",
        "homeTeam_teamId": "HOME_TEAM_ID",
        "awayTeam_teamId": "AWAY_TEAM_ID",
        "homeTeam_teamTricode": "HOME_ABBR",
        "awayTeam_teamTricode": "AWAY_ABBR",
        "homeTeam_score": "HOME_PTS",
        "awayTeam_score": "AWAY_PTS",
        "arenaName": "ARENA_NAME",
        "arenaCity": "ARENA_CITY",
        "arenaState": "ARENA_STATE",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    required = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_ABBR", "AWAY_ABBR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            "ScheduleLeagueV2 returned an unexpected schema. Missing columns: "
            f"{missing}\nAvailable columns: {list(raw.columns)}"
        )

    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["GAME_DATE"] = _to_tz_naive_datetime(df["GAME_DATE"])
    df = df.dropna(subset=["GAME_DATE"])
    df = df[_selected_game_type_mask(df["GAME_ID"], include_playoffs, include_playin)].copy()

    df["SEASON_ID"] = df["GAME_DATE"].map(_canonical_nba_season_label)
    df["GAME_TYPE"] = df["GAME_ID"].map(_game_type_from_game_id)

    for col in ["HOME_PTS", "AWAY_PTS"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Future scheduled games often come back with 0-0 scores instead of NaN.
    # Therefore, do NOT use only HOME_PTS/AWAY_PTS presence to decide whether a
    # game has been played. Use NBA's schedule status first:
    #   1 = scheduled, 2 = live/in progress, 3 = final/completed.
    if "GAME_STATUS" in df.columns:
        status_num = pd.to_numeric(df["GAME_STATUS"], errors="coerce")
    else:
        status_num = pd.Series(pd.NA, index=df.index, dtype="Float64")

    status_text = (
        df.get("GAME_STATUS_TEXT", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    is_final_status = status_num.eq(3) | status_text.str.contains("FINAL", na=False)
    score_cols_present = df["HOME_PTS"].notna() & df["AWAY_PTS"].notna()

    df["HAS_SCORE"] = (is_final_status & score_cols_present).astype(int)
    df["HOME_WIN"] = pd.NA
    scored = df["HAS_SCORE"].eq(1)
    df.loc[scored, "HOME_WIN"] = (df.loc[scored, "HOME_PTS"] > df.loc[scored, "AWAY_PTS"]).astype(int)
    df["POINT_DIFF"] = pd.NA
    df.loc[scored, "POINT_DIFF"] = df.loc[scored, "HOME_PTS"] - df.loc[scored, "AWAY_PTS"]

    # Do not use schedule rows for training. Training uses LeagueGameFinder completed rows.
    df["IS_COMPLETED"] = df["HAS_SCORE"].astype(int)
    df["IS_TRAINING_ROW"] = 0
    df["IS_PREDICTION_ROW"] = 0
    df["ROW_SOURCE"] = "schedule"

    cols = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON_ID",
        "GAME_TYPE",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "HOME_ABBR",
        "AWAY_ABBR",
        "HOME_PTS",
        "AWAY_PTS",
        "HOME_WIN",
        "POINT_DIFF",
        "GAME_STATUS",
        "GAME_STATUS_TEXT",
        "GAME_LABEL",
        "SERIES_TEXT",
        "HAS_SCORE",
        "IS_COMPLETED",
        "IS_TRAINING_ROW",
        "IS_PREDICTION_ROW",
        "ROW_SOURCE",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].sort_values("GAME_DATE").reset_index(drop=True)


def build_prediction_games(schedule: pd.DataFrame, *, as_of: date) -> pd.DataFrame:
    """Return future scheduled games that do not have scores yet.

    ScheduleLeagueV2 can return timezone-aware datetimes, while --nba-as-of is a plain
    date. Compare using date objects so the filter is stable across Pandas versions and
    avoids tz-aware/tz-naive comparison errors.
    """
    if schedule.empty:
        return schedule.copy()

    df = schedule.copy()
    df["GAME_DATE"] = _to_tz_naive_datetime(df["GAME_DATE"])
    game_dates = df["GAME_DATE"].dt.date

    # Include unplayed games from the cutoff date onward. Using >= matters when
    # you fetch earlier in the day and want tonight's games included.
    has_score = pd.to_numeric(df.get("HAS_SCORE", 0), errors="coerce").fillna(0).astype(int)
    is_completed = pd.to_numeric(df.get("IS_COMPLETED", 0), errors="coerce").fillna(0).astype(int)
    pred = df[(game_dates >= as_of) & (has_score == 0) & (is_completed == 0)].copy()
    pred["IS_PREDICTION_ROW"] = 1
    pred["IS_TRAINING_ROW"] = 0
    return pred.sort_values("GAME_DATE").reset_index(drop=True)


def main(
    *,
    nba_season_from: str = "2017-18",
    nba_season_to: str | None = None,
    nba_include_playoffs: bool = True,
    nba_include_playin: bool = True,
    nba_as_of: date | None = None,
) -> Path:
    project_root = _project_root()
    out_dir = project_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    as_of = nba_as_of or date.today()
    last = nba_season_to or default_nba_season_through(as_of)
    seasons = nba_season_labels_inclusive(nba_season_from, last)

    raw_completed = fetch_completed_seasons(
        seasons,
        include_playoffs=nba_include_playoffs,
        include_playin=nba_include_playin,
    )
    games = games_to_match_rows(raw_completed)

    raw_schedule = fetch_schedule_seasons(seasons)
    schedule = normalize_schedule(
        raw_schedule,
        include_playoffs=nba_include_playoffs,
        include_playin=nba_include_playin,
    )
    prediction_games = build_prediction_games(schedule, as_of=as_of)

    raw_completed_path = out_dir / "raw_team_games.csv"
    games_path = out_dir / "games.csv"
    raw_schedule_path = out_dir / "raw_schedule.csv"
    schedule_path = out_dir / "schedule.csv"
    prediction_path = out_dir / "prediction_games.csv"

    raw_completed.to_csv(raw_completed_path, index=False)
    games.to_csv(games_path, index=False)
    raw_schedule.to_csv(raw_schedule_path, index=False)
    schedule.to_csv(schedule_path, index=False)
    prediction_games.to_csv(prediction_path, index=False)

    print("Source: stats.nba.com via nba_api")
    print(f"Seasons: {seasons[0]} .. {seasons[-1]} ({len(seasons)} seasons)")
    print(f"As of: {as_of.isoformat()}")
    print(f"Wrote {games_path} ({len(games)} completed games for training)")
    print(f"Wrote {prediction_path} ({len(prediction_games)} future games for prediction)")
    print(f"Wrote {schedule_path} ({len(schedule)} total schedule rows)")
    if not schedule.empty:
        if "GAME_STATUS" in schedule.columns:
            print("\nSchedule GAME_STATUS counts:")
            print(schedule["GAME_STATUS"].value_counts(dropna=False).to_string())
        if "GAME_STATUS_TEXT" in schedule.columns:
            print("\nSchedule GAME_STATUS_TEXT counts:")
            print(schedule["GAME_STATUS_TEXT"].value_counts(dropna=False).head(15).to_string())

    if not prediction_games.empty:
        preview_cols = ["GAME_DATE", "GAME_TYPE", "AWAY_ABBR", "HOME_ABBR", "GAME_STATUS_TEXT", "GAME_LABEL", "SERIES_TEXT"]
        preview_cols = [c for c in preview_cols if c in prediction_games.columns]
        print("\nUpcoming prediction games:")
        print(prediction_games[preview_cols].head(20).to_string(index=False))

    return games_path


def _parse_as_of(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Fetch completed NBA games and future prediction games separately.")
    parser.add_argument("--nba-season-from", default="2017-18")
    parser.add_argument("--nba-season-to", default=None)
    parser.add_argument("--nba-as-of", default=None, help="YYYY-MM-DD cutoff for prediction_games.csv")
    parser.add_argument("--no-playoffs", action="store_true")
    parser.add_argument("--no-playin", action="store_true")
    args = parser.parse_args()

    main(
        nba_season_from=args.nba_season_from,
        nba_season_to=args.nba_season_to,
        nba_include_playoffs=not args.no_playoffs,
        nba_include_playin=not args.no_playin,
        nba_as_of=_parse_as_of(args.nba_as_of) if args.nba_as_of else None,
    )


if __name__ == "__main__":
    _cli()
