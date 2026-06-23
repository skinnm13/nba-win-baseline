# NBA Win Baseline Predictor

A small NBA prediction project that downloads historical NBA results, builds pre-game rolling features, trains a logistic regression baseline model, and serves predictions through a **React web app** (GitHub Pages) or the CLI.

The current version separates **completed games** from **future scheduled games** so the model trains only on games that already have results, while the web UI shows future matchups for prediction.

---

## Web app (React + GitHub Pages)

The interactive UI lives in `web/` and runs entirely in the browser using exported JSON from your trained model.

### Local development

After fetch + train:

```bash
python scripts/export_web_data.py   # writes web/public/data/*.json
cd web
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173/nba-win-baseline/`).

### Deploy to GitHub Pages

1. In GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**
2. Commit exported data under `web/public/data/` (required unless `data/` and `artifacts/` are also in the repo)
3. Push to `main` — the workflow in `.github/workflows/deploy-pages.yml` builds and deploys

Live site URL (default): `https://<username>.github.io/nba-win-baseline/`

To refresh predictions on the site, re-run fetch → train → export, commit the updated JSON, and push.

---

## App Preview

![NBA Predictor GUI Screenshot](docs/ss1.png)


![NBA Predictor GUI Screenshot](docs/ss2.png)

---

## What This Project Does

The project has three main stages:

```text
1. Fetch data
2. Train model
3. Predict matchups
```

The key data split is:

```text
data/games.csv              completed games only, used for training
data/prediction_games.csv   future/unplayed games only, used by the GUI
data/schedule.csv           full normalized schedule
```

This matters because completed games have final labels like:

```text
HOME_WIN
HOME_PTS
AWAY_PTS
POINT_DIFF
```

Future games do not have these values yet, so they should not be used for training.

---

## Features

- Fetches NBA game data using `nba_api`
- Supports regular season, playoffs, and Play-In games
- Keeps training data and prediction schedule data separate
- Builds shifted rolling pre-game features
- Trains a logistic regression baseline model
- Supports command-line matchup prediction
- **React web UI** with upcoming schedule + custom matchups (GitHub Pages)
- Loads upcoming games from exported `prediction_games.json`

---

## Project Structure

```text
nba-win-baseline/
├── .github/workflows/deploy-pages.yml
├── artifacts/
│   └── baseline_logreg.pkl
├── data/
│   ├── games.csv
│   ├── prediction_games.csv
│   └── schedule.csv
├── docs/
├── scripts/
│   └── export_web_data.py
├── src/
│   ├── fetch_games.py
│   ├── features.py
│   └── train_baseline.py
├── web/                         React app (GitHub Pages)
│   ├── public/data/             exported JSON (commit for deploy)
│   └── src/
├── fetch_data.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```


| Path                            | Purpose                                                    |
| ------------------------------- | ---------------------------------------------------------- |
| `fetch_data.py`                 | Entry point for fetching NBA data                          |
| `train.py`                      | Entry point for training the model                         |
| `predict.py`                    | CLI prediction for one matchup                             |
| `scripts/export_web_data.py`    | Export model + CSVs → `web/public/data/*.json`             |
| `web/`                          | React UI (schedule + predictions, deploys to GitHub Pages) |
| `data/games.csv`                | Completed games only                                       |
| `data/prediction_games.csv`     | Future games only                                          |
| `data/schedule.csv`             | Full normalized schedule                                   |
| `artifacts/baseline_logreg.pkl` | Trained model artifact                                     |


---

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Required packages include:

```text
nba_api
pandas
numpy
scikit-learn
```

---

## Full Workflow

Run these commands from the project root.

### 1. Fetch NBA Data

```bash
python fetch_data.py \
  --nba-season-from 2017-18 \
  --nba-season-to 2025-26 \
  --nba-as-of 2026-05-12
```

You can also run it on one line:

```bash
python fetch_data.py --nba-season-from 2017-18 --nba-season-to 2025-26 --nba-as-of 2026-05-12
```

This creates:

```text
data/games.csv
data/prediction_games.csv
data/schedule.csv
data/raw_team_games.csv
data/raw_schedule.csv
```

### 2. Train the Model

```bash
python train.py
```

This trains on:

```text
data/games.csv
```

and creates:

```text
artifacts/baseline_logreg.pkl
```

### 4. Export for the web app

```bash
python scripts/export_web_data.py
cd web && npm install && npm run dev
```

### 5. Predict one custom matchup (CLI)

```bash
python predict.py --home Lakers --away Thunder --game-date 2026-05-20
```

Example with abbreviations:

```bash
python predict.py --home BOS --away NYK --game-date 2026-05-15
```

---

## Data Files Explained

### `data/games.csv`

This file contains only completed games.

It is used for:

```text
training
feature history
command-line predictions
GUI predictions
```

Typical columns:

```text
GAME_ID
GAME_DATE
SEASON_ID
HOME_TEAM_ID
AWAY_TEAM_ID
HOME_ABBR
AWAY_ABBR
HOME_PTS
AWAY_PTS
HOME_WIN
POINT_DIFF
```

The model trains only on this file.

---

### `data/prediction_games.csv`

This file contains future or unplayed scheduled games.

It is used by:

```text
web/ (via exported prediction_games.json)
predict.py (CLI)
```

Typical columns:

```text
GAME_ID
GAME_DATE
SEASON_ID
GAME_TYPE
HOME_TEAM_ID
AWAY_TEAM_ID
HOME_ABBR
AWAY_ABBR
GAME_STATUS
GAME_STATUS_TEXT
IS_PREDICTION_ROW
```

These rows should not be used for training because the final scores are not available yet.

---

### `data/schedule.csv`

This is the full normalized schedule.

It can contain:

```text
completed games
future games
playoff games
Play-In games
regular season games
```

It is useful for debugging and checking what the NBA schedule endpoint returned.

---

## How the Model Works

The model is a simple baseline classifier.

It uses:

```text
historical completed games
rolling team features
rest-day style features
home/away context
```

The training pipeline:

```text
data/games.csv
    -> add_shifted_roll_features()
    -> feature_matrix()
    -> StandardScaler()
    -> LogisticRegression()
    -> artifacts/baseline_logreg.pkl
```

The prediction pipeline:

```text
data/games.csv
    -> add one temporary synthetic future matchup row
    -> build shifted rolling features
    -> score the synthetic row
    -> return home and away win probabilities
```

The synthetic prediction row is created only in memory. It is not written back to `games.csv`.

---

## Web UI Behavior

The React app loads static JSON from `web/public/data/`:

```text
games.json              completed games (feature history)
prediction_games.json   upcoming schedule
model.json              scaler + logistic regression weights
teams.json              team names for display / input resolution
```

Flow:

```text
fetch_data.py → train.py → scripts/export_web_data.py → web/public/data/
npm run dev / GitHub Pages deploy → browser loads JSON → predict in TypeScript
```

The web app uses the same feature logic and model weights as `predict.py`, aligned with the train/test split (`testSeasonId` shown in the UI header).

---

## Command Reference

### Fetch

```bash
python fetch_data.py --nba-season-from 2017-18 --nba-season-to 2025-26 --nba-as-of 2026-05-12
```

Useful flags:


| Flag                        | Purpose                                          |
| --------------------------- | ------------------------------------------------ |
| `--nba-season-from 2017-18` | First season to fetch                            |
| `--nba-season-to 2025-26`   | Last season to fetch                             |
| `--nba-as-of YYYY-MM-DD`    | Date used to decide which games are future games |
| `--no-playoffs`             | Exclude playoff games                            |
| `--no-playin`               | Exclude Play-In games                            |


### Train

```bash
python train.py
```

### Export for web

```bash
python scripts/export_web_data.py
```

### Predict CLI

```bash
python predict.py --home HOME_TEAM --away AWAY_TEAM --game-date YYYY-MM-DD
```

Examples:

```bash
python predict.py --home Lakers --away Thunder --game-date 2026-05-20
python predict.py --home BOS --away NYK --game-date 2026-05-15
```

### Web app

```bash
python scripts/export_web_data.py
cd web && npm install && npm run dev
```

---

## Team Input Rules

For `predict.py`, `--home` and `--away` accept:

```text
abbreviation
nickname
full team name
unique partial match
```

Examples:

```bash
python predict.py --home LAL --away OKC --game-date 2026-05-20
python predict.py --home Lakers --away Thunder --game-date 2026-05-20
python predict.py --home "Los Angeles Lakers" --away "Oklahoma City Thunder" --game-date 2026-05-20
```

If the team name is unknown or ambiguous, the script prints valid NBA team names.

---

## Adding a Screenshot to the README

Create a folder:

```bash
mkdir -p docs
```

Add your screenshot:

```text
docs/app-screenshot.png
```

Then keep this Markdown in the README:

```markdown
![NBA Predictor GUI Screenshot](docs/app-screenshot.png)
```

If the image does not show on GitHub, check:

```text
1. The file is committed
2. The path is correct
3. The filename capitalization matches exactly
```

---

## Troubleshooting

### 1. `zsh: command not found: --nba-season-from`

This happens when a multi-line command is copied incorrectly.

Wrong:

```bash
python fetch_data.py \\
  --nba-season-from 2017-18
```

Correct:

```bash
python fetch_data.py \
  --nba-season-from 2017-18
```

The backslash must be a single `\` at the very end of the line.

Safest option: run it on one line.

```bash
python fetch_data.py --nba-season-from 2017-18 --nba-season-to 2025-26 --nba-as-of 2026-05-12
```

---

### 2. `unrecognized arguments: \`

This is the same shell issue as above. Remove the extra backslash or run the command on one line.

---

### 3. `ValueError: Found array with 0 sample(s)`

This means the training script found zero valid training rows.

Common causes:

```text
data/games.csv is empty
only one season was fetched
future games were accidentally mixed into games.csv
the latest season was held out as test, leaving no training seasons
```

Check your training file:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/games.csv'); print(df.shape); print(df['SEASON_ID'].value_counts().sort_index())"
```

Fix by fetching multiple seasons:

```bash
python fetch_data.py --nba-season-from 2017-18 --nba-season-to 2025-26
python train.py
```

---

### 4. `Cannot compare tz-naive and tz-aware datetime-like objects`

This means Pandas is comparing timezone-aware NBA API dates with timezone-naive local dates.

The fixed fetch logic normalizes these dates. If you see this error, make sure your current `src/fetch_games.py` includes timezone normalization before filtering future games.

Then rerun:

```bash
python fetch_data.py --nba-season-from 2017-18 --nba-season-to 2025-26 --nba-as-of 2026-05-12
```

---

### 5. `prediction_games.csv` is empty

First check the file:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/prediction_games.csv'); print(df.shape); print(df.head())"
```

Possible causes:

```text
There are no future NBA games in the selected season range
--nba-as-of is after the scheduled games
future games are being marked as completed
the schedule endpoint returned no future rows
```

Try setting `--nba-as-of` earlier:

```bash
python fetch_data.py --nba-season-from 2025-26 --nba-season-to 2025-26 --nba-as-of 2026-05-01
```

Also inspect the full schedule:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/schedule.csv'); print(df[['GAME_DATE','AWAY_ABBR','HOME_ABBR','GAME_STATUS_TEXT']].tail(30))"
```

---

### 6. GUI opens but shows no upcoming games

Check if `prediction_games.csv` has rows:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/prediction_games.csv'); print(df[['GAME_DATE','AWAY_ABBR','HOME_ABBR','GAME_STATUS_TEXT']].head(20))"
```

If it is empty, rerun the fetch command.

If it has rows but the web app schedule is empty, re-run `python scripts/export_web_data.py` and refresh. Check the **Days ahead** filter in the UI.

---

### 7. `ImportError: cannot import name 'predict_matchup' from 'predict'`

This means `predict.py` is missing the reusable `predict_matchup()` function.

Check:

```bash
python -c "from predict import predict_matchup, resolve_user_team; print('predict import works')"
```

If this fails, restore the version of `predict.py` that includes:

```python
def predict_matchup(...):
    ...
```

The GUI depends on this function.

---

### 8. `Missing artifacts/baseline_logreg.pkl`

You need to train the model first.

```bash
python train.py
```

Then re-export and open the web app:

```bash
python scripts/export_web_data.py
cd web && npm run dev
```

---

### 9. `Missing data/games.csv`

You need to fetch data first.

```bash
python fetch_data.py --nba-season-from 2017-18 --nba-season-to 2025-26
```

Then train:

```bash
python train.py
```

---

### 10. Web app shows “Failed to load …”

Run the export step after fetch + train:

```bash
python scripts/export_web_data.py
```

Commit `web/public/data/*.json` if deploying via GitHub Pages without committing `data/`.

---

### 11. NBA API request fails or hangs

The NBA Stats API can be sensitive to network conditions.

Try:

```text
turn off VPN
try a different network
wait and rerun
reduce the season range
rerun only the latest season
```

Example:

```bash
python fetch_data.py --nba-season-from 2025-26 --nba-season-to 2025-26
```

---

### 12. Feature column mismatch

If you see an error like:

```text
Feature column mismatch. Model expects ..., got ...
```

then the feature code changed after the model was trained.

Fix:

```bash
python train.py
```

Then rerun prediction.

---

## Development Notes

### Why `games.csv` and `prediction_games.csv` are separate

The model should not train on future games because they do not have real labels.

Good:

```text
games.csv -> train.py
prediction_games.csv -> scripts/export_web_data.py -> web app
```

Bad:

```text
prediction_games.csv -> train.py
```

### Why future games use placeholder scores only during prediction

`predict_matchup()` creates a synthetic future row with placeholder values in memory. Because the rolling features are shifted, the synthetic row's own fake score should not leak into its prediction features.

The placeholder row is never saved to disk.

---

## Quick Sanity Checks

Check completed games:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/games.csv'); print(df.shape); print(df.tail())"
```

Check future prediction games:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/prediction_games.csv'); print(df.shape); print(df[['GAME_DATE','AWAY_ABBR','HOME_ABBR','GAME_STATUS_TEXT']].head(20))"
```

Check full schedule:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/schedule.csv'); print(df.shape); print(df.tail())"
```

Check trained model artifact:

```bash
python -c "import pickle; print(pickle.load(open('artifacts/baseline_logreg.pkl','rb')).keys())"
```

Check prediction import:

```bash
python -c "from predict import predict_matchup, resolve_user_team; print('predict import works')"
```

Run a quick prediction:

```bash
python predict.py --home BOS --away NYK --game-date 2026-05-15
```

---

## Current End-to-End Flow

```text
python fetch_data.py → data/games.csv, data/prediction_games.csv
python train.py → artifacts/baseline_logreg.pkl
python scripts/export_web_data.py → web/public/data/*.json
web app (npm run dev or GitHub Pages) → scores selected / custom games
python predict.py → scores one custom matchup from CLI
```

---

## Limitations

- This is a baseline model, not a production betting model.
- Predictions depend on the quality and coverage of the fetched historical data.
- NBA API availability can vary by network.
- The model does not account for injuries, player rotations, betting lines, travel difficulty, or real-time roster changes.
- Playoff context is included through historical game results, but the model is still intentionally simple.
