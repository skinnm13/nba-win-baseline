"""Train a logistic regression baseline on pre-game rolling features."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import add_shifted_roll_features, feature_matrix


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def time_split(df: pd.DataFrame, test_season_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["SEASON_ID"] != test_season_id].copy()
    test = df[df["SEASON_ID"] == test_season_id].copy()
    return train, test


def majority_home_baseline(y_test: np.ndarray) -> float:
    return float(np.mean(y_test == 1))


def main() -> None:
    games_path = _project_root() / "data" / "games.csv"
    if not games_path.exists():
        raise SystemExit(f"Missing {games_path}. Run: python -m src.fetch_games")

    games = pd.read_csv(games_path, parse_dates=["GAME_DATE"])
    games["SEASON_ID"] = games["SEASON_ID"].astype(str)
    df = add_shifted_roll_features(games)

    # Hold out last downloaded season for test
    test_season = str(df["SEASON_ID"].max())
    train_df, test_df = time_split(df, test_season)

    X_train, cols = feature_matrix(train_df)
    y_train = train_df["HOME_WIN"].to_numpy()
    X_test, _ = feature_matrix(test_df)
    y_test = test_df["HOME_WIN"].to_numpy()

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
    print(f"Train seasons: {sorted(train_df['SEASON_ID'].unique())}  n={len(train_df)}")
    print(f"Test season:  {test_season}  n={len(test_df)}")
    print(f"Accuracy:     {accuracy_score(y_test, pred):.4f}")
    print(f"Log loss:     {log_loss(y_test, proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_test, proba):.4f}")
    print(f"Always-home accuracy (test): {majority_home_baseline(y_test):.4f}")

    out_dir = _project_root() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": cols,
        "test_season_id": test_season,
    }
    with open(out_dir / "baseline_logreg.pkl", "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved {out_dir / 'baseline_logreg.pkl'}")


if __name__ == "__main__":
    main()
