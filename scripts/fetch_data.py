#!/usr/bin/env python3
"""Fetch NBA games data from the NBA API."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def fetch_games_data() -> pd.DataFrame:
    """Fetch NBA games data from the NBA API."""
    print("Fetching NBA games data from NBA API...")
    
    try:
        # Get all games from 2003 onwards (when reliable data is available)
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_type_nullable="Regular Season"
        )
        df = gamefinder.get_data_frames()[0]
        
        # Rename and select relevant columns
        df = df[["GAME_ID", "GAME_DATE", "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]].copy()
        
        # Convert GAME_DATE to datetime
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        
        # Pivot to get home and away teams in one row
        games = []
        grouped = df.groupby("GAME_ID")
        
        for game_id, group in grouped:
            if len(group) != 2:
                continue
                
            row1, row2 = group.iloc[0], group.iloc[1]
            
            # Determine home/away (typically first entry in API is away, second is home)
            # But we'll use PTS diff as a heuristic if needed
            game = {
                "GAME_ID": str(game_id),
                "GAME_DATE": row1["GAME_DATE"].strftime("%Y-%m-%d"),
                "SEASON_ID": str(int(row1["SEASON_ID"])),
                "HOME_TEAM_ID": str(int(row2["TEAM_ID"])),
                "AWAY_TEAM_ID": str(int(row1["TEAM_ID"])),
                "HOME_ABBR": row2["TEAM_ABBREVIATION"],
                "AWAY_ABBR": row1["TEAM_ABBREVIATION"],
                "HOME_PTS": int(row2["PTS"]),
                "AWAY_PTS": int(row1["PTS"]),
                "HOME_WIN": 1 if row2["PTS"] > row1["PTS"] else 0,
            }
            games.append(game)
        
        result_df = pd.DataFrame(games)
        print(f"Fetched {len(result_df)} games")
        return result_df
        
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        raise SystemExit(f"Failed to fetch NBA games data: {e}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    games_df = fetch_games_data()
    games_path = DATA_DIR / "games.csv"
    games_df.to_csv(games_path, index=False)
    print(f"Saved {len(games_df)} games to {games_path}")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
