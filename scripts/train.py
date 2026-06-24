#!/usr/bin/env python3
"""Train a logistic regression model to predict NBA home team wins."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_games_data(games_path: Path) -> pd.DataFrame:
    """Load and validate games data."""
    if not games_path.exists():
        raise SystemExit(f"Missing {games_path}. Run fetch_data.py first.")
    
    df = pd.read_csv(games_path)
    print(f"Loaded {len(df)} games")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Engineer features for the model."""
    features = df.copy()
    
    # Create features - basic approach using team IDs and date features
    features["HOME_TEAM_ID"] = features["HOME_TEAM_ID"].astype(str)
    features["AWAY_TEAM_ID"] = features["AWAY_TEAM_ID"].astype(str)
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"])
    features["GAME_MONTH"] = features["GAME_DATE"].dt.month
    features["GAME_DAYOFWEEK"] = features["GAME_DATE"].dt.dayofweek
    
    # Point differential encoding as a simple feature
    features["POINT_DIFF"] = features["HOME_PTS"] - features["AWAY_PTS"]
    
    feature_cols = ["GAME_MONTH", "GAME_DAYOFWEEK"]
    
    # One-hot encode team IDs (top teams by frequency)
    home_teams = features["HOME_TEAM_ID"].value_counts().head(10)
    away_teams = features["AWAY_TEAM_ID"].value_counts().head(10)
    
    for team in home_teams.index:
        col = f"HOME_TEAM_{team}"
        features[col] = (features["HOME_TEAM_ID"] == team).astype(int)
        feature_cols.append(col)
    
    for team in away_teams.index:
        col = f"AWAY_TEAM_{team}"
        features[col] = (features["AWAY_TEAM_ID"] == team).astype(int)
        feature_cols.append(col)
    
    return features[feature_cols], feature_cols


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
) -> dict:
    """Train a logistic regression model."""
    print("Training logistic regression model...")
    
    # Create pipeline with scaler and classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    
    pipeline.fit(X, y)
    
    print(f"Model trained with {len(feature_columns)} features")
    print(f"Model accuracy on training set: {pipeline.score(X, y):.3f}")
    
    return {
        "model": pipeline,
        "feature_columns": feature_columns,
        "training_source": "data/games.csv",
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    games_path = DATA_DIR / "games.csv"
    model_path = ARTIFACTS_DIR / "baseline_logreg.pkl"
    
    # Load data
    df = load_games_data(games_path)
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Engineer features
    X, feature_cols = engineer_features(df)
    y = df["HOME_WIN"].values
    
    print(f"Training on {len(X)} samples with {len(feature_cols)} features")
    
    # Train model
    artifact = train_model(X.values, y, feature_cols)
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
