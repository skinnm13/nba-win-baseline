#!/usr/bin/env python3
"""Score a hypothetical home game using the trained baseline (run from project root).

This file supports both:
  1. CLI usage: python predict.py --home Lakers --away Thunder --game-date 2026-05-20
  2. GUI/core usage: import predict_matchup, resolve_user_team
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import uuid
from difflib import get_close_matches
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _nba_team_rows():
    from nba_api.stats.static import teams as nba_teams

    return nba_teams.get_teams()


def _team_directory_text() -> str:
    rows = sorted(
        (t["abbreviation"], t["full_name"]) for t in _nba_team_rows() if t.get("abbreviation")
    )
    lines = ["NBA teams (use abbreviation in scripts, or a matching full / nickname name):", ""]
    w = max(len(ab) for ab, _ in rows)
    for ab, name in rows:
        lines.append(f"  {ab:<{w}}  {name}")
    return "\n".join(lines)


def _lookup_by_abbr(abbr: str):
    from nba_api.stats.static import teams as nba_teams

    return nba_teams.find_team_by_abbreviation(abbr)


def _display_team(abbr: str) -> str:
    t = _lookup_by_abbr(abbr)
    if t:
        return f"{t['full_name']} ({abbr})"
    return abbr


def resolve_user_team(text: str) -> str:
    """Map user input (abbr, full name, nickname, etc.) to canonical abbreviation."""
    s = (text or "").strip()
    if not s:
        raise ValueError("empty team")

    all_teams = _nba_team_rows()
    sl = s.lower()

    for t in all_teams:
        if s.upper() == str(t["abbreviation"]).upper():
            return t["abbreviation"]

    for t in all_teams:
        if sl == str(t["full_name"]).lower():
            return t["abbreviation"]

    for t in all_teams:
        if sl == str(t["nickname"]).lower():
            return t["abbreviation"]

    partial: list[dict] = []
    seen_abbr: set[str] = set()
    for t in all_teams:
        fn = str(t["full_name"]).lower()
        nick = str(t["nickname"]).lower()
        ab = t["abbreviation"]
        if (sl in fn or sl in nick) and ab not in seen_abbr:
            seen_abbr.add(ab)
            partial.append(t)
    if len(partial) == 1:
        return partial[0]["abbreviation"]
    if len(partial) > 1:
        opts = ", ".join(
            f"{x['nickname']} ({x['abbreviation']})"
            for x in sorted(partial, key=lambda z: z["full_name"])
        )
        raise ValueError(f"Ambiguous {s!r}; matches: {opts}")

    nicknames = [str(t["nickname"]) for t in all_teams]
    fulls = [str(t["full_name"]) for t in all_teams]
    abbrs = [str(t["abbreviation"]) for t in all_teams]
    pool = nicknames + fulls + abbrs
    hints = get_close_matches(s, pool, n=3, cutoff=0.55)
    if hints:
        raise ValueError(f"Unknown team {s!r}. Did you mean: {', '.join(hints)}?")

    raise ValueError(f"Unknown team {s!r}")


def _exit_with_team_directory(msg: str) -> None:
    print(msg, file=sys.stderr)
    print(file=sys.stderr)
    print(_team_directory_text(), file=sys.stderr)
    raise SystemExit(2)


def _abbr_to_team_id(games: pd.DataFrame, abbr: str):
    """Return latest known NBA team id for an abbreviation from games.csv history."""
    abbr = abbr.strip().upper()
    if "GAME_DATE" not in games.columns:
        raise ValueError("games DataFrame is missing GAME_DATE")

    g = games.sort_values("GAME_DATE")

    required = {"HOME_ABBR", "AWAY_ABBR", "HOME_TEAM_ID", "AWAY_TEAM_ID"}
    missing = sorted(required - set(g.columns))
    if missing:
        raise ValueError(f"games DataFrame missing required columns: {missing}")

    home = g.loc[
        g["HOME_ABBR"].astype(str).str.upper() == abbr,
        ["GAME_DATE", "HOME_TEAM_ID"],
    ].rename(columns={"HOME_TEAM_ID": "TEAM_ID"})

    away = g.loc[
        g["AWAY_ABBR"].astype(str).str.upper() == abbr,
        ["GAME_DATE", "AWAY_TEAM_ID"],
    ].rename(columns={"AWAY_TEAM_ID": "TEAM_ID"})

    sub = pd.concat([home, away], ignore_index=True)
    if sub.empty:
        raise ValueError(
            f"No games for team {abbr!r} in games.csv (fetch NBA data that includes this franchise). "
            f"See team list below.\n\n{_team_directory_text()}"
        )

    return sub.sort_values("GAME_DATE").iloc[-1]["TEAM_ID"]


def _infer_season_id(games: pd.DataFrame, game_date: pd.Timestamp, season_id: str | None) -> str:
    if season_id is not None:
        return str(season_id)

    prior = games[games["GAME_DATE"] < game_date]
    if prior.empty:
        return str(games["SEASON_ID"].max())

    return str(prior.sort_values("GAME_DATE").iloc[-1]["SEASON_ID"])


def predict_matchup(
    *,
    games: pd.DataFrame,
    artifact: dict,
    home_abbr: str,
    away_abbr: str,
    game_date: pd.Timestamp,
    season_id: str | None = None,
    stable_prediction_id: bool = True,
) -> dict:
    """Return home/away win probabilities for a hypothetical game.

    This function is intentionally in-memory. It does not write to CSV.

    Parameters
    ----------
    games:
        Historical completed games, usually loaded from data/games.csv.
        It must have the same columns used during training.
    artifact:
        Loaded artifacts/baseline_logreg.pkl dictionary. Must contain:
        - model
        - feature_columns
    home_abbr / away_abbr:
        Canonical NBA abbreviations, such as BOS, NYK, LAL, OKC.
    game_date:
        Date of the future or hypothetical matchup.
    season_id:
        Optional SEASON_ID. If omitted, the latest season before game_date is used.
    stable_prediction_id:
        If True, creates deterministic synthetic GAME_ID and checks for collisions.
        If False, appends a random suffix so repeated GUI calls do not collide.
    """
    from src.features import add_shifted_roll_features, feature_matrix

    if games.empty:
        raise ValueError("games DataFrame is empty. Run fetch_data.py first.")

    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    games = games.dropna(subset=["GAME_DATE"])
    games["SEASON_ID"] = games["SEASON_ID"].astype(str)

    model = artifact.get("model")
    if model is None:
        raise ValueError("Artifact missing model; retrain with train.py.")

    expected_cols = artifact.get("feature_columns")
    if expected_cols is None:
        raise ValueError("Artifact missing feature_columns; retrain with current train_baseline.py.")

    home_abbr = home_abbr.strip().upper()
    away_abbr = away_abbr.strip().upper()
    if home_abbr == away_abbr:
        raise ValueError("home and away teams cannot be the same")

    game_date = pd.to_datetime(game_date)

    home_id = _abbr_to_team_id(games, home_abbr)
    away_id = _abbr_to_team_id(games, away_abbr)
    sid = _infer_season_id(games, game_date, season_id)

    base_gid = f"PRED_{game_date.strftime('%Y%m%d')}_{home_abbr}_VS_{away_abbr}"
    pred_gid = base_gid if stable_prediction_id else f"{base_gid}_{uuid.uuid4().hex[:8]}"

    if stable_prediction_id and (games["GAME_ID"].astype(str) == pred_gid).any():
        raise ValueError(f"GAME_ID {pred_gid} already exists in games table")

    placeholder = pd.DataFrame(
        [
            {
                "GAME_ID": pred_gid,
                "GAME_DATE": game_date,
                "SEASON_ID": str(sid),
                "HOME_TEAM_ID": home_id,
                "AWAY_TEAM_ID": away_id,
                "HOME_ABBR": home_abbr,
                "AWAY_ABBR": away_abbr,
                # These are placeholders only. The feature code uses shifted
                # rolling features, so the synthetic row's own score/result
                # should not leak into its prediction features.
                "HOME_PTS": 0,
                "AWAY_PTS": 0,
                "HOME_WIN": 0,
                "POINT_DIFF": 0,
            }
        ]
    )

    combined = pd.concat([games, placeholder], ignore_index=True)
    combined = combined.sort_values("GAME_DATE").reset_index(drop=True)

    fe = add_shifted_roll_features(combined)
    row = fe[fe["GAME_ID"].astype(str) == pred_gid]
    if row.empty:
        raise RuntimeError("Internal error: synthetic row missing after feature build.")

    X, cols = feature_matrix(row)
    if list(cols) != list(expected_cols):
        raise ValueError(
            f"Feature column mismatch. Model expects {expected_cols}, got {cols}. "
            "Retrain or match code version."
        )

    p_home = float(model.predict_proba(X)[0, 1])
    p_away = 1.0 - p_home
    pred_label = int(p_home >= 0.5)

    home_label = _display_team(home_abbr)
    away_label = _display_team(away_abbr)
    pick_abbr = home_abbr if pred_label == 1 else away_abbr

    return {
        "p_home": p_home,
        "p_away": p_away,
        "pick_abbr": pick_abbr,
        "pick_side": "home" if pred_label == 1 else "away",
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "home_label": home_label,
        "away_label": away_label,
        "game_date": game_date,
        "season_id": str(sid),
        "game_id": pred_gid,
    }


def main() -> None:
    root = _project_root()
    os.chdir(root)
    sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(
        description="Predict home win probability for a matchup using artifacts/baseline_logreg.pkl "
        "and history in data/games.csv."
    )
    p.add_argument(
        "--home",
        required=True,
        help="Home team: abbreviation (LAL), nickname (Lakers), or full name.",
    )
    p.add_argument(
        "--away",
        required=True,
        help="Away team: abbreviation (OKC), nickname (Thunder), or full name.",
    )
    p.add_argument("--game-date", required=True, help="Game date (ISO), e.g. 2026-05-15")
    p.add_argument(
        "--season",
        default=None,
        help="SEASON_ID for the synthetic row (default: latest SEASON_ID in games.csv before this date)",
    )
    p.add_argument("--games", type=Path, default=root / "data" / "games.csv")
    p.add_argument("--model", type=Path, default=root / "artifacts" / "baseline_logreg.pkl")
    args = p.parse_args()

    games_path = args.games.resolve()
    model_path = args.model.resolve()

    if not games_path.exists():
        raise SystemExit(f"Missing {games_path}. Run fetch_data.py first.")
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}. Run train.py first.")

    try:
        home_abbr = resolve_user_team(args.home)
        away_abbr = resolve_user_team(args.away)
    except ValueError as e:
        _exit_with_team_directory(str(e))

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    games = pd.read_csv(games_path, parse_dates=["GAME_DATE"])
    games["SEASON_ID"] = games["SEASON_ID"].astype(str)
    game_date = pd.to_datetime(args.game_date)

    try:
        out = predict_matchup(
            games=games,
            artifact=artifact,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            game_date=game_date,
            season_id=args.season,
            stable_prediction_id=True,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    except RuntimeError as e:
        raise SystemExit(str(e)) from e

    if args.season is None:
        print(f"Using SEASON_ID={out['season_id']}. Pass --season to override.")

    p_home = out["p_home"]
    p_away = out["p_away"]
    home_label = out["home_label"]
    away_label = out["away_label"]
    season_id = out["season_id"]

    print(f"{away_abbr} @ {home_abbr}  {game_date.date()}  (SEASON_ID={season_id})")
    print(f"P(home win): {p_home:.4f}")
    print()
    print(f"{home_label}  P(Win) = {p_home:.4f}")
    print(f"{away_label}  P(Win) = {p_away:.4f}")
    print()
    print(f"Pick (>0.5 home): {out['pick_abbr']} ({out['pick_side']})")


if __name__ == "__main__":
    main()
