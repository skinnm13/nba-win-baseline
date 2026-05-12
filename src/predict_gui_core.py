"""Shared logic for predict GUIs (tkinter or browser).

This version prefers data/prediction_games.csv created by src.fetch_games.py.
If that file is missing or empty, it falls back to ScoreboardV2.
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from predict import predict_matchup, resolve_user_team
from src.scoreboard_schedule import ScheduledGame, upcoming_games


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bootstrap_cwd() -> Path:
    root = project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_context(root: Path) -> tuple[pd.DataFrame, dict]:
    games_path = root / "data" / "games.csv"
    model_path = root / "artifacts" / "baseline_logreg.pkl"
    if not games_path.exists():
        raise FileNotFoundError(f"Missing {games_path}. Run fetch_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run train.py first.")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    games = pd.read_csv(games_path, parse_dates=["GAME_DATE"])
    games["SEASON_ID"] = games["SEASON_ID"].astype(str)
    return games, artifact


def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse mixed NBA API date strings safely and strip timezone awareness."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def _truthy_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _int_or_default(value: Any, default: int = 1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _clean_abbr(value: Any) -> str:
    return str(value or "").strip().upper()


def _scheduled_from_prediction_csv(
    root: Path,
    *,
    days: int,
    include_live: bool,
    start: date | None = None,
) -> list[ScheduledGame]:
    """Load upcoming games from data/prediction_games.csv.

    Expected columns from the current fetcher:
      GAME_ID, GAME_DATE, HOME_ABBR, AWAY_ABBR, GAME_STATUS, GAME_STATUS_TEXT

    The function is intentionally tolerant of aliases so older generated files
    are still usable.
    """
    pred_path = root / "data" / "prediction_games.csv"
    if not pred_path.exists():
        return []

    try:
        df = pd.read_csv(pred_path)
    except pd.errors.EmptyDataError:
        return []

    if df.empty:
        return []

    date_col = _first_existing_col(df, ["GAME_DATE", "gameDateTimeEst", "gameDateEst", "GAME_DATE_EST"])
    gid_col = _first_existing_col(df, ["GAME_ID", "gameId", "GAMECODE", "gameCode"])
    home_col = _first_existing_col(df, ["HOME_ABBR", "HOME_TEAM_ABBREVIATION", "homeTeam_teamTricode", "HOME"])
    away_col = _first_existing_col(df, ["AWAY_ABBR", "VISITOR_TEAM_ABBREVIATION", "awayTeam_teamTricode", "AWAY"])
    status_col = _first_existing_col(df, ["GAME_STATUS", "GAME_STATUS_ID", "gameStatus"])
    status_text_col = _first_existing_col(df, ["GAME_STATUS_TEXT", "gameStatusText", "statusText"])

    missing = [
        label
        for label, col in {
            "date": date_col,
            "game id": gid_col,
            "home team": home_col,
            "away team": away_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(
            f"{pred_path} is missing required prediction schedule columns: {', '.join(missing)}"
        )

    df = df.copy()
    df["_GAME_DATE"] = _to_naive_datetime(df[date_col])
    df = df.dropna(subset=["_GAME_DATE"])

    d0 = start or date.today()
    d1 = d0 + timedelta(days=max(1, int(days)) - 1)
    game_dates = df["_GAME_DATE"].dt.date

    # prediction_games.csv should already be future-only, but keep this filter
    # so stale files do not show old slates in the GUI.
    mask = (game_dates >= d0) & (game_dates <= d1)

    # If marker columns exist, trust them.
    if "IS_PREDICTION_ROW" in df.columns:
        mask &= _truthy_series(df, "IS_PREDICTION_ROW", default=True)

    # Do not show completed rows if a broader schedule file was accidentally copied here.
    if "IS_COMPLETED" in df.columns:
        mask &= ~_truthy_series(df, "IS_COMPLETED", default=False)

    if not include_live:
        # NBA status: 1 scheduled, 2 live, 3 final. If status is unavailable,
        # keep the row because prediction_games.csv is already curated.
        if status_col is not None:
            status_num = pd.to_numeric(df[status_col], errors="coerce")
            mask &= status_num.fillna(1).astype(int) != 2

    df = df.loc[mask].copy()
    if df.empty:
        return []

    rows: list[ScheduledGame] = []
    seen: set[str] = set()
    for _, row in df.sort_values("_GAME_DATE").iterrows():
        gid = str(row[gid_col]).strip()
        if not gid or gid.lower() == "nan" or gid in seen:
            continue

        away = _clean_abbr(row[away_col])
        home = _clean_abbr(row[home_col])
        if not away or not home or away == "NAN" or home == "NAN":
            continue

        status = _int_or_default(row[status_col], default=1) if status_col else 1
        status_text = str(row[status_text_col]).strip() if status_text_col else ""
        if status_text.lower() == "nan":
            status_text = ""

        rows.append(
            ScheduledGame(
                game_id=gid,
                game_date=pd.Timestamp(row["_GAME_DATE"]).date(),
                away_abbr=away,
                home_abbr=home,
                game_status=status,
                status_text=status_text or "Scheduled",
            )
        )
        seen.add(gid)

    rows.sort(key=lambda x: (x.game_date, x.game_id))
    return rows


def refresh_schedule_rows(
    days: int,
    include_live: bool,
    *,
    prefer_prediction_csv: bool = True,
    fallback_scoreboard: bool = True,
) -> list[ScheduledGame]:
    """Return upcoming games for the GUI.

    Priority:
      1. data/prediction_games.csv from fetch_data.py / src.fetch_games
      2. ScoreboardV2 fallback for near-term games

    Keeping the return type as list[ScheduledGame] means predict_gui.py and
    predict_gui_web.py can keep using the same prediction path.
    """
    n = max(1, min(int(days), 60))
    root = project_root()

    rows: list[ScheduledGame] = []
    if prefer_prediction_csv:
        rows = _scheduled_from_prediction_csv(root, days=n, include_live=include_live)

    if rows:
        return rows

    if fallback_scoreboard:
        return upcoming_games(days=n, include_live=include_live)

    return []


def predict_scheduled_text(games: pd.DataFrame, artifact: dict, g: ScheduledGame) -> str:
    game_ts = pd.Timestamp(g.game_date)
    try:
        res = predict_matchup(
            games=games,
            artifact=artifact,
            home_abbr=g.home_abbr,
            away_abbr=g.away_abbr,
            game_date=game_ts,
            season_id=g.season_id,
            stable_prediction_id=False,
        )
    except Exception as e:
        return f"[error] {g.away_abbr} @ {g.home_abbr} {g.game_date}: {e}\n"
    return (
        f"{g.away_abbr} @ {g.home_abbr}  {g.game_date}  (SEASON_ID={res['season_id']})\n"
        f"  P(home win): {res['p_home']:.4f}  |  {res['home_label']}: {res['p_home']:.4f}  |  "
        f"{res['away_label']}: {res['p_away']:.4f}\n"
        f"  Pick (>0.5 home): {res['pick_abbr']} ({res['pick_side']})\n"
    )


def predict_custom_text(games: pd.DataFrame, artifact: dict, away: str, home: str, date_str: str) -> str:
    ha = resolve_user_team(home)
    aa = resolve_user_team(away)
    gd = pd.to_datetime(date_str.strip())
    try:
        res = predict_matchup(
            games=games,
            artifact=artifact,
            home_abbr=ha,
            away_abbr=aa,
            game_date=gd,
            season_id=None,
            stable_prediction_id=False,
        )
    except Exception as e:
        return f"[error] {aa} @ {ha} {gd.date()}: {e}\n"
    return (
        f"{aa} @ {ha}  {gd.date()}  (SEASON_ID={res['season_id']})\n"
        f"  P(home win): {res['p_home']:.4f}  |  {res['home_label']}: {res['p_home']:.4f}  |  "
        f"{res['away_label']}: {res['p_away']:.4f}\n"
        f"  Pick (>0.5 home): {res['pick_abbr']} ({res['pick_side']})\n"
    )
