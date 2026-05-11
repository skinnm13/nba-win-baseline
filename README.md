# NBA win baseline

A small baseline pipeline that downloads NBA game results, builds **pre-game** rolling features (last games, rest days), trains a **logistic regression** model with a scaler, and can score hypothetical matchups.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies include `nba_api`, `pandas`, `numpy`, and `scikit-learn`.

## 1. Get data (`data/games.csv`)

Use either the **FiveThirtyEight** archive (quick, offline after download) or **NBA Stats** (current seasons and playoffs).

### Option A — FiveThirtyEight (default)

```bash
python fetch_data.py
# equivalent:
python -m src.fetch_games
```

The public `nbaallelo` CSV is **not updated through the present**; it is useful for older-era smoke tests only.

Optional: `--min-year 2005` (FTE only) filters by `year_id`.

### Option B — NBA Stats (recommended for this season and playoffs)

Requires network access to `stats.nba.com` (VPN or rate limits can block requests).

```bash
python fetch_data.py --source nba
```

By default this fetches:

- Seasons **2017-18** through the **current NBA season** (inferred from today’s date),
- **Regular season** and **Playoffs**.

Useful flags:

| Flag | Purpose |
|------|--------|
| `--nba-season-from 2017-18` | First season to include |
| `--nba-season-to 2025-26` | Last season (overrides “through current”) |
| `--nba-as-of YYYY-MM-DD` | When `--nba-season-to` is omitted, infer “current” season from this date |
| `--no-playoffs` | Regular season only |
| `--playin` | Include Play-In tournament games |

Examples:

```bash
# Refresh only the latest season file range
python fetch_data.py --source nba --nba-season-from 2025-26 --nba-season-to 2025-26

# Regular season + playoffs + play-in
python fetch_data.py --source nba --playin
```

Fetched games use a **canonical** `SEASON_ID` label (e.g. `2025-26`) so regular season and playoffs stay in one season bucket for training.

## 2. Train (`artifacts/baseline_logreg.pkl`)

After `data/games.csv` exists:

```bash
python train.py
# equivalent:
python -m src.train_baseline
```

The script:

- Builds shifted rolling team features,
- Holds out the **latest** `SEASON_ID` in the file as the test set,
- Prints accuracy, log loss, and Brier score,
- Saves `artifacts/baseline_logreg.pkl` (model, feature column names, and test season id).

## 3. Predict a matchup (`predict.py`)

Requires `data/games.csv` and `artifacts/baseline_logreg.pkl` (run fetch + train first).

```bash
python predict.py --home Lakers --away Thunder --game-date 2026-05-20
```

### Team names

`--home` and `--away` accept:

- Abbreviations (`LAL`, `OKC`),
- Nicknames (`Lakers`, `Thunder`),
- Full names (`Los Angeles Lakers`),
- Or a **unique** partial match on the official full or nickname string.

If the name is unknown, ambiguous, or not close to any team, the program prints an error and a **full list of NBA abbreviations and official team names**, then exits with code `2`.

Team metadata comes from `nba_api` static data; **game history** still comes from `games.csv`, so teams need to appear in your fetched data.

### Other flags

| Flag | Default | Purpose |
|------|---------|--------|
| `--game-date` | (required) | ISO date for the hypothetical game |
| `--season` | inferred | `SEASON_ID` for the synthetic row; omit to use the latest season in the CSV before `--game-date` |
| `--games` | `data/games.csv` | Alternate games table |
| `--model` | `artifacts/baseline_logreg.pkl` | Alternate trained artifact |

### Output

You get `P(home win)`, then **each team’s** win probability (they sum to 1), using full names and abbreviations, plus a simple pick at a 0.5 threshold on the home win probability.

## Project layout

| Path | Role |
|------|------|
| `fetch_data.py` | Thin entrypoint → `src.fetch_games` |
| `train.py` | Thin entrypoint → `src.train_baseline` |
| `predict.py` | CLI scoring for one hypothetical game |
| `src/fetch_games.py` | Download / normalize games |
| `src/features.py` | Rolling pre-game features |
| `src/train_baseline.py` | Train + evaluate + write pickle |
| `data/games.csv` | One row per game (created by fetch) |
| `artifacts/baseline_logreg.pkl` | Trained pipeline + metadata (created by train) |

## Notes

- **NBA API**: `--source nba` uses the community `nba_api` client; failures are often network or blocking by the stats site, not your code.
- **Model**: This is a simple baseline for experimentation, not a tuned production forecaster.
