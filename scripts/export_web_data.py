#!/usr/bin/env python3
"""Export trained model and CSV data as JSON for the static React app."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "data"

GAME_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON_ID",
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
    "HOME_ABBR",
    "AWAY_ABBR",
    "HOME_PTS",
    "AWAY_PTS",
    "HOME_WIN",
]

SCHEDULE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON_ID",
    "HOME_ABBR",
    "AWAY_ABBR",
    "GAME_STATUS",
    "GAME_STATUS_TEXT",
    "IS_PREDICTION_ROW",
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def export_teams() -> list[dict]:
    from nba_api.stats.static import teams as nba_teams

    rows = []
    for t in nba_teams.get_teams():
        rows.append(
            {
                "id": int(t["id"]),
                "abbreviation": str(t["abbreviation"]),
                "fullName": str(t["full_name"]),
                "nickname": str(t["nickname"]),
                "city": str(t["city"]),
            }
        )
    return sorted(rows, key=lambda x: x["fullName"])


def export_model(artifact_path: Path) -> dict:
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    return {
        "featureColumns": artifact["feature_columns"],
        "testSeasonId": artifact.get("test_season_id"),
        "trainingSource": artifact.get("training_source", "data/games.csv"),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "classifier": {
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
        },
    }


def export_games(games_path: Path) -> list[dict]:
    df = pd.read_csv(games_path, parse_dates=["GAME_DATE"])
    df = df.dropna(subset=["GAME_DATE", "HOME_PTS", "AWAY_PTS", "HOME_WIN"])
    missing = [c for c in GAME_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{games_path} missing columns: {missing}")

    df = df[GAME_COLUMNS].copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["GAME_DATE"] = df["GAME_DATE"].dt.strftime("%Y-%m-%d")
    df["SEASON_ID"] = df["SEASON_ID"].astype(str)
    df["HOME_ABBR"] = df["HOME_ABBR"].astype(str).str.upper()
    df["AWAY_ABBR"] = df["AWAY_ABBR"].astype(str).str.upper()
    df["HOME_WIN"] = df["HOME_WIN"].astype(int)
    return df.to_dict(orient="records")


def export_schedule(schedule_path: Path) -> list[dict]:
    if not schedule_path.exists():
        return []

    df = pd.read_csv(schedule_path, parse_dates=["GAME_DATE"])
    if df.empty:
        return []

    cols = [c for c in SCHEDULE_COLUMNS if c in df.columns]
    df = df[cols].copy()
    if "GAME_ID" in df.columns:
        df["GAME_ID"] = df["GAME_ID"].astype(str)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["HOME_ABBR"] = df["HOME_ABBR"].astype(str).str.upper()
    df["AWAY_ABBR"] = df["AWAY_ABBR"].astype(str).str.upper()
    if "IS_PREDICTION_ROW" in df.columns:
        df["IS_PREDICTION_ROW"] = df["IS_PREDICTION_ROW"].fillna(0).astype(int)
    return df.to_dict(orient="records")


def main() -> None:
    games_path = ROOT / "data" / "games.csv"
    model_path = ROOT / "artifacts" / "baseline_logreg.pkl"
    prediction_path = ROOT / "data" / "prediction_games.csv"

    if not games_path.exists():
        raise SystemExit(f"Missing {games_path}. Run fetch_data.py first.")
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}. Run train.py first.")

    meta = {
        "exportedFrom": "scripts/export_web_data.py",
        "gamesCount": 0,
        "predictionGamesCount": 0,
        "testSeasonId": None,
    }

    model = export_model(model_path)
    meta["testSeasonId"] = model.get("testSeasonId")
    _write_json(OUT / "model.json", model)

    games = export_games(games_path)
    meta["gamesCount"] = len(games)
    _write_json(OUT / "games.json", games)

    schedule_source = prediction_path if prediction_path.exists() else ROOT / "data" / "schedule.csv"
    schedule = export_schedule(schedule_source)
    meta["predictionGamesCount"] = len(schedule)
    _write_json(OUT / "prediction_games.json", schedule)

    teams = export_teams()
    _write_json(OUT / "teams.json", teams)

    _write_json(OUT / "meta.json", meta)

    print(f"Wrote {OUT}/model.json")
    print(f"Wrote {OUT}/games.json ({len(games)} completed games)")
    print(f"Wrote {OUT}/prediction_games.json ({len(schedule)} schedule rows)")
    print(f"Wrote {OUT}/teams.json ({len(teams)} teams)")
    print(f"Test season holdout: {meta['testSeasonId']}")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
