"""Pre-game features from prior games only (shifted rolling stats)."""

from __future__ import annotations

import pandas as pd


def _team_long_table(games: pd.DataFrame) -> pd.DataFrame:
    """Each team-game as one row for computing rolling team form."""
    home = games.assign(
        TEAM_ID=games["HOME_TEAM_ID"],
        OPP_ID=games["AWAY_TEAM_ID"],
        IS_HOME=1,
        WIN=games["HOME_WIN"],
        PF=games["HOME_PTS"],
        PA=games["AWAY_PTS"],
    )[["GAME_ID", "GAME_DATE", "TEAM_ID", "OPP_ID", "IS_HOME", "WIN", "PF", "PA"]]

    away = games.assign(
        TEAM_ID=games["AWAY_TEAM_ID"],
        OPP_ID=games["HOME_TEAM_ID"],
        IS_HOME=0,
        WIN=1 - games["HOME_WIN"],
        PF=games["AWAY_PTS"],
        PA=games["HOME_PTS"],
    )[["GAME_ID", "GAME_DATE", "TEAM_ID", "OPP_ID", "IS_HOME", "WIN", "PF", "PA"]]

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return long_df


def add_shifted_roll_features(games: pd.DataFrame, window: int = 10, min_periods: int = 3) -> pd.DataFrame:
    long_df = _team_long_table(games)

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        w = g["WIN"].shift(1)
        margin = (g["PF"] - g["PA"]).shift(1)
        g["WIN_RATE_L10"] = w.rolling(window, min_periods=min_periods).mean()
        g["MARGIN_L10"] = margin.rolling(window, min_periods=min_periods).mean()
        g["REST_DAYS"] = g["GAME_DATE"].diff().dt.days
        return g

    long_df = long_df.groupby("TEAM_ID", group_keys=False).apply(_roll)

    home_side = long_df[long_df["IS_HOME"] == 1][
        ["GAME_ID", "WIN_RATE_L10", "MARGIN_L10", "REST_DAYS"]
    ].rename(
        columns={
            "WIN_RATE_L10": "HOME_WIN_RATE_L10",
            "MARGIN_L10": "HOME_MARGIN_L10",
            "REST_DAYS": "HOME_REST_DAYS",
        }
    )
    away_side = long_df[long_df["IS_HOME"] == 0][
        ["GAME_ID", "WIN_RATE_L10", "MARGIN_L10", "REST_DAYS"]
    ].rename(
        columns={
            "WIN_RATE_L10": "AWAY_WIN_RATE_L10",
            "MARGIN_L10": "AWAY_MARGIN_L10",
            "REST_DAYS": "AWAY_REST_DAYS",
        }
    )

    out = games.merge(home_side, on="GAME_ID", how="left").merge(away_side, on="GAME_ID", how="left")
    return out


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        "HOME_WIN_RATE_L10",
        "HOME_MARGIN_L10",
        "HOME_REST_DAYS",
        "AWAY_WIN_RATE_L10",
        "AWAY_MARGIN_L10",
        "AWAY_REST_DAYS",
    ]
    X = df[feature_cols].fillna(0.0)
    return X, feature_cols
