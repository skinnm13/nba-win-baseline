import { useEffect, useMemo, useState } from "react";
import { loadAppData } from "./lib/data";
import { formatPrediction, predictMatchup } from "./lib/predict";
import { filterSchedule } from "./lib/schedule";
import { loadTeams, resolveUserTeam } from "./lib/teams";
import type { AppData, PredictionResult, ScheduledGame } from "./lib/types";
import "./App.css";

function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(14);
  const [includeLive, setIncludeLive] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [log, setLog] = useState<string[]>([]);
  const [customAway, setCustomAway] = useState("");
  const [customHome, setCustomHome] = useState("");
  const [customDate, setCustomDate] = useState("");

  useEffect(() => {
    loadAppData()
      .then((d) => {
        loadTeams(d.teams);
        setData(d);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const schedule = useMemo(() => {
    if (!data) return [];
    return filterSchedule(data.schedule, days, includeLive);
  }, [data, days, includeLive]);

  function toggleSelect(gameId: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(gameId)) next.delete(gameId);
      else next.add(gameId);
      return next;
    });
  }

  function appendResults(results: PredictionResult[]) {
    const lines = results.map((r) => formatPrediction(r));
    setLog((prev) => [...prev, "---", ...lines]);
  }

  function predictRows(rows: ScheduledGame[]) {
    if (!data) return;
    const results: PredictionResult[] = [];
    for (const g of rows) {
      try {
        results.push(
          predictMatchup(
            data.games,
            data.model,
            g.HOME_ABBR,
            g.AWAY_ABBR,
            g.GAME_DATE,
            g.SEASON_ID,
          ),
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setLog((prev) => [...prev, `[error] ${g.AWAY_ABBR} @ ${g.HOME_ABBR} ${g.GAME_DATE}: ${msg}`]);
      }
    }
    if (results.length) appendResults(results);
  }

  function onPredictSelected() {
    const rows = schedule.filter((g) => selected.has(g.GAME_ID));
    if (!rows.length) {
      setLog((prev) => [...prev, "Select at least one game first."]);
      return;
    }
    predictRows(rows);
  }

  function onPredictCustom(e: React.FormEvent) {
    e.preventDefault();
    if (!data) return;
    try {
      const home = resolveUserTeam(customHome);
      const away = resolveUserTeam(customAway);
      const res = predictMatchup(data.games, data.model, home, away, customDate.trim());
      appendResults([res]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLog((prev) => [...prev, `[error] custom predict: ${msg}`]);
    }
  }

  if (error) {
    return (
      <main className="app">
        <h1>NBA Win Baseline</h1>
        <p className="error">{error}</p>
        <p>
          From the project root run: <code>python fetch_data.py</code>, <code>python train.py</code>, then{" "}
          <code>python scripts/export_web_data.py</code>
        </p>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="app">
        <p>Loading model and schedule…</p>
      </main>
    );
  }

  return (
    <main className="app">
      <header>
        <h1>NBA Win Baseline</h1>
        <p className="subtitle">
          Trained on {data.meta.gamesCount.toLocaleString()} completed games · test holdout{" "}
          {data.meta.testSeasonId ?? data.model.testSeasonId ?? "—"}
        </p>
      </header>

      <section className="panel">
        <div className="controls">
          <label>
            Days ahead
            <input
              type="number"
              min={1}
              max={60}
              value={days}
              onChange={(e) => setDays(Number(e.target.value) || 7)}
            />
          </label>
          <label className="checkbox">
            <input type="checkbox" checked={includeLive} onChange={(e) => setIncludeLive(e.target.checked)} />
            Include live games
          </label>
          <button type="button" onClick={onPredictSelected}>
            Predict selected
          </button>
        </div>

        {schedule.length === 0 ? (
          <p className="muted">No upcoming games in this window. Re-export data or use custom predict below.</p>
        ) : (
          <ul className="schedule">
            {schedule.map((g) => (
              <li key={g.GAME_ID}>
                <label>
                  <input
                    type="checkbox"
                    checked={selected.has(g.GAME_ID)}
                    onChange={() => toggleSelect(g.GAME_ID)}
                  />
                  <span className="date">{g.GAME_DATE}</span>
                  <span className="matchup">
                    {g.AWAY_ABBR} @ {g.HOME_ABBR}
                  </span>
                  <span className="status">{g.GAME_STATUS_TEXT || "Scheduled"}</span>
                </label>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>Custom matchup</h2>
        <form className="custom-form" onSubmit={onPredictCustom}>
          <label>
            Away
            <input value={customAway} onChange={(e) => setCustomAway(e.target.value)} placeholder="OKC" />
          </label>
          <label>
            Home
            <input value={customHome} onChange={(e) => setCustomHome(e.target.value)} placeholder="LAL" />
          </label>
          <label>
            Date
            <input value={customDate} onChange={(e) => setCustomDate(e.target.value)} placeholder="2026-05-15" />
          </label>
          <button type="submit">Predict custom</button>
        </form>
      </section>

      <section className="panel output">
        <h2>Output</h2>
        <pre>{log.length ? log.join("\n\n") : "(no predictions yet)"}</pre>
      </section>
    </main>
  );
}

export default App;
