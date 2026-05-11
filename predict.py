#!/usr/bin/env python3
"""Score a hypothetical home game using the trained baseline (run from project root)."""

import argparse
import os
import pickle
import sys
from difflib import get_close_matches
from pathlib import Path


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
        opts = ", ".join(f"{x['nickname']} ({x['abbreviation']})" for x in sorted(partial, key=lambda z: z["full_name"]))
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


def _abbr_to_team_id(games, abbr: str):
    import pandas as pd

    abbr = abbr.strip().upper()
    g = games.sort_values("GAME_DATE")
    home = g.loc[g["HOME_ABBR"].astype(str).str.upper() == abbr, ["GAME_DATE", "HOME_TEAM_ID"]].rename(
        columns={"HOME_TEAM_ID": "TEAM_ID"}
    )
    away = g.loc[g["AWAY_ABBR"].astype(str).str.upper() == abbr, ["GAME_DATE", "AWAY_TEAM_ID"]].rename(
        columns={"AWAY_TEAM_ID": "TEAM_ID"}
    )
    sub = pd.concat([home, away], ignore_index=True)
    if sub.empty:
        raise SystemExit(
            f"No games for team {abbr!r} in games.csv (fetch NBA data that includes this franchise). "
            f"See team list below.\n\n{_team_directory_text()}"
        )
    return sub.sort_values("GAME_DATE").iloc[-1]["TEAM_ID"]


def main() -> None:
    root = _project_root()
    os.chdir(root)
    sys.path.insert(0, str(root))

    import pandas as pd

    from src.features import add_shifted_roll_features, feature_matrix

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
    model = artifact["model"]
    expected_cols = artifact.get("feature_columns")
    if expected_cols is None:
        raise SystemExit("Artifact missing feature_columns; retrain with current train_baseline.py.")

    games = pd.read_csv(games_path, parse_dates=["GAME_DATE"])
    games["SEASON_ID"] = games["SEASON_ID"].astype(str)
    game_date = pd.to_datetime(args.game_date)

    home_id = _abbr_to_team_id(games, home_abbr)
    away_id = _abbr_to_team_id(games, away_abbr)

    prior = games[games["GAME_DATE"] < game_date]
    season_id = args.season
    if season_id is None:
        if prior.empty:
            season_id = str(games["SEASON_ID"].max())
        else:
            season_id = str(prior.sort_values("GAME_DATE").iloc[-1]["SEASON_ID"])
        print(f"Using SEASON_ID={season_id}. Pass --season to override.")

    pred_gid = f"PRED_{game_date.strftime('%Y%m%d')}_{home_abbr}_VS_{away_abbr}"
    if (games["GAME_ID"].astype(str) == pred_gid).any():
        raise SystemExit(f"GAME_ID {pred_gid} already exists in {games_path}")

    placeholder = pd.DataFrame(
        [
            {
                "GAME_ID": pred_gid,
                "GAME_DATE": game_date,
                "SEASON_ID": str(season_id),
                "HOME_TEAM_ID": home_id,
                "AWAY_TEAM_ID": away_id,
                "HOME_ABBR": home_abbr,
                "AWAY_ABBR": away_abbr,
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
        raise SystemExit("Internal error: synthetic row missing after feature build.")

    X, cols = feature_matrix(row)
    if list(cols) != list(expected_cols):
        raise SystemExit(
            f"Feature column mismatch. Model expects {expected_cols}, got {cols}. Retrain or match code version."
        )

    p_home = float(model.predict_proba(X)[0, 1])
    p_away = 1.0 - p_home
    pred_label = int(p_home >= 0.5)

    home_label = _display_team(home_abbr)
    away_label = _display_team(away_abbr)

    print(f"{away_abbr} @ {home_abbr}  {game_date.date()}  (SEASON_ID={season_id})")
    print(f"P(home win): {p_home:.4f}")
    print()
    print(f"{home_label}  P(Win) = {p_home:.4f}")
    print(f"{away_label}  P(Win) = {p_away:.4f}")
    print()
    pick = home_abbr if pred_label == 1 else away_abbr
    print(f"Pick (>0.5 home): {pick} ({'home' if pred_label == 1 else 'away'})")


if __name__ == "__main__":
    main()
