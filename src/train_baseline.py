"""Train a logistic regression baseline on completed NBA games only.

This version is safe to use even if your data folder also contains future/prediction
rows from a schedule fetcher. It trains only on completed rows with HOME_WIN labels.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import add_shifted_roll_features, feature_matrix


REQUIRED_TRAINING_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON_ID",
    "HOME_TEAM_ID",
    "AWAY_TEAM_ID",
    "HOME_PTS",
    "AWAY_PTS",
    "HOME_WIN",
    "POINT_DIFF",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_completed_games(project_root: Path) -> pd.DataFrame:
    """Load completed games from data/games.csv.

    games.csv should contain only played games. However, this function also guards
    against accidentally mixed future rows by filtering them out if marker columns
    are present.
    """
    games_path = project_root / "data" / "games.csv"
    marked_path = project_root / "data" / "all_games_marked.csv"

    if games_path.exists():
        path = games_path
    elif marked_path.exists():
        # Fallback only. Prefer games.csv because it should already be clean.
        path = marked_path
    else:
        raise SystemExit(
            f"Missing training data. Expected {games_path} or {marked_path}.\n"
            "Run your fetch script first."
        )

    games = pd.read_csv(path, parse_dates=["GAME_DATE"])

    # If the file came from the marked fetcher, keep only rows safe for training.
    if "IS_TRAINING_ROW" in games.columns:
        games = games[games["IS_TRAINING_ROW"].fillna(0).astype(int) == 1].copy()
    if "IS_PREDICTION_ROW" in games.columns:
        games = games[games["IS_PREDICTION_ROW"].fillna(0).astype(int) == 0].copy()
    if "HAS_SCORE" in games.columns:
        games = games[games["HAS_SCORE"].fillna(0).astype(int) == 1].copy()

    missing = [c for c in REQUIRED_TRAINING_COLUMNS if c not in games.columns]
    if missing:
        raise SystemExit(
            f"Training file {path} is missing required columns: {missing}\n"
            "Use a completed-games file for training, not a future schedule-only file."
        )

    # Drop future/scheduled rows that do not have labels or scores.
    games = games.dropna(subset=["GAME_DATE", "HOME_PTS", "AWAY_PTS", "HOME_WIN"]).copy()
    games["SEASON_ID"] = games["SEASON_ID"].astype(str)
    games["HOME_WIN"] = games["HOME_WIN"].astype(int)

    # Protect against duplicate completed games, especially if merged from multiple APIs.
    source_rank = pd.Series(0, index=games.index)
    if "ROW_SOURCE" in games.columns:
        source_rank = games["ROW_SOURCE"].astype(str).map(
            {"completed_results": 2, "leaguegamefinder": 2, "schedule": 1}
        ).fillna(0)
    games = (
        games.assign(_source_rank=source_rank)
        .sort_values(["GAME_DATE", "GAME_ID", "_source_rank"])
        .drop_duplicates(subset=["GAME_ID"], keep="last")
        .drop(columns=["_source_rank"], errors="ignore")
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )

    if games.empty:
        raise SystemExit(
            f"No completed training rows found in {path}.\n"
            "Check that data/games.csv contains played games with HOME_WIN labels."
        )

    return games


def _pick_test_season(df: pd.DataFrame, requested: str | None = None) -> str:
    """Pick a test season that still leaves at least one earlier training season."""
    seasons = sorted(df["SEASON_ID"].astype(str).unique())
    if not seasons:
        raise SystemExit("No seasons found after filtering completed games.")

    if requested:
        requested = str(requested)
        if requested not in seasons:
            raise SystemExit(
                f"Requested test season {requested!r} is not in the completed training data.\n"
                f"Available seasons: {seasons}"
            )
        if (df["SEASON_ID"] != requested).sum() == 0:
            raise SystemExit(
                f"Requested test season {requested!r} leaves 0 training rows.\n"
                "Fetch at least two seasons or choose a different test split."
            )
        return requested

    # Default: latest season that leaves non-empty train and test sets.
    for season in reversed(seasons):
        if (df["SEASON_ID"] != season).sum() > 0 and (df["SEASON_ID"] == season).sum() > 0:
            return season

    raise SystemExit(
        "Only one completed season is available, so season holdout would leave 0 training rows.\n"
        "Fetch more seasons, for example: --nba-season-from 2017-18 --nba-season-to 2025-26"
    )


def time_split(df: pd.DataFrame, test_season_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["SEASON_ID"].astype(str) != str(test_season_id)].copy()
    test = df[df["SEASON_ID"].astype(str) == str(test_season_id)].copy()
    return train, test


def majority_home_baseline(y_test: np.ndarray) -> float:
    return float(np.mean(y_test == 1))


def _validate_feature_sets(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    if len(y_train) == 0 or X_train.shape[0] == 0:
        raise SystemExit(
            "After feature generation, the training set has 0 rows.\n"
            "This usually means you fetched only one season or your rolling-feature code dropped too many early games.\n"
            "Fetch multiple completed seasons and make sure data/games.csv has completed games."
        )
    if len(y_test) == 0 or X_test.shape[0] == 0:
        raise SystemExit(
            "After feature generation, the test set has 0 rows. Choose another --test-season or fetch more data."
        )
    if len(np.unique(y_train)) < 2:
        raise SystemExit(
            "Training labels contain only one class after filtering. Need both home wins and home losses."
        )


def main(test_season: str | None = None) -> None:
    project_root = _project_root()
    games = _load_completed_games(project_root)

    # Build pre-game rolling features only from completed games.
    df = add_shifted_roll_features(games)
    df = df.dropna(subset=["HOME_WIN"]).copy()
    df["SEASON_ID"] = df["SEASON_ID"].astype(str)

    test_season = _pick_test_season(df, requested=test_season)
    train_df, test_df = time_split(df, test_season)

    X_train, cols = feature_matrix(train_df)
    y_train = train_df["HOME_WIN"].astype(int).to_numpy()
    X_test, _ = feature_matrix(test_df)
    y_test = test_df["HOME_WIN"].astype(int).to_numpy()

    _validate_feature_sets(X_train, y_train, X_test, y_test)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("Features:", cols)
    print(f"Completed games loaded: {len(games)}")
    print(f"Train seasons: {sorted(train_df['SEASON_ID'].unique())}  n={len(train_df)}")
    print(f"Test season:  {test_season}  n={len(test_df)}")
    print(f"Accuracy:     {accuracy_score(y_test, pred):.4f}")
    print(f"Log loss:     {log_loss(y_test, proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_test, proba):.4f}")
    print(f"Always-home accuracy (test): {majority_home_baseline(y_test):.4f}")

    out_dir = project_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": cols,
        "test_season_id": test_season,
        "training_source": "data/games.csv",
    }
    out_path = out_dir / "baseline_logreg.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved {out_path}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train NBA baseline model on completed games only.")
    parser.add_argument(
        "--test-season",
        default=None,
        help="Season to hold out for testing, e.g. 2024-25. Default: latest completed season available.",
    )
    args = parser.parse_args()
    main(test_season=args.test_season)


if __name__ == "__main__":
    _cli()
