#!/usr/bin/env python3
"""Browser UI (stdlib only): upcoming games + baseline predictions. Use when Tk is unavailable."""

from __future__ import annotations

import html
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from src.predict_gui_core import (
    bootstrap_cwd,
    load_context,
    predict_custom_text,
    predict_scheduled_text,
    refresh_schedule_rows,
)
from src.scoreboard_schedule import ScheduledGame


def _page(
    *,
    status: str,
    log: str,
    rows: list[ScheduledGame],
    days: int,
    include_live: bool,
    port: int,
) -> bytes:
    esc = html.escape
    check_rows = []
    for g in rows:
        gid = esc(g.game_id)
        label = esc(f"{g.game_date}  {g.away_abbr} @ {g.home_abbr}  ({g.status_text or 'Scheduled'})")
        check_rows.append(
            f'<label style="display:block;margin:0.25em 0;"><input type="checkbox" name="gid" value="{gid}"> {label}</label>'
        )
    checks = "\n".join(check_rows) if check_rows else "<p><em>No prediction games in this window.</em></p>"
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NBA baseline — upcoming games</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 52rem; margin: 1rem auto; padding: 0 1rem; }}
  pre {{ background: #f4f4f4; padding: 0.75rem; overflow: auto; white-space: pre-wrap; }}
  fieldset {{ margin: 1rem 0; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 0.5rem 0; }}
</style>
</head>
<body>
<h1>NBA baseline predictor</h1>
<p>Local server on port <code>{port}</code> — close this terminal or press Ctrl+C to stop.</p>

<form method="post" action="/refresh">
  <fieldset>
    <legend>Schedule</legend>
    <p>Loads from <code>data/prediction_games.csv</code> first, then falls back to NBA ScoreboardV2 if needed.</p>
    <div class="row">
      <label>Days ahead <input name="days" type="number" min="1" max="60" value="{days}"></label>
      <label><input name="live" type="checkbox" value="1" {"checked" if include_live else ""}> Include live games</label>
      <button type="submit">Refresh schedule</button>
    </div>
  </fieldset>
</form>

<form method="post" action="/predict">
  <fieldset>
    <legend>Upcoming prediction games</legend>
    {checks}
    <div class="row" style="margin-top:0.75rem;">
      <button type="submit">Predict selected</button>
    </div>
  </fieldset>
</form>

<form method="post" action="/custom">
  <fieldset>
    <legend>Custom matchup</legend>
    <div class="row">
      <label>Away <input name="away" type="text" placeholder="OKC" style="width:7rem;"></label>
      <label>Home <input name="home" type="text" placeholder="LAL" style="width:7rem;"></label>
      <label>Date <input name="game_date" type="text" placeholder="2026-05-15" style="width:9rem;"></label>
      <button type="submit">Predict custom</button>
    </div>
  </fieldset>
</form>

<p><strong>Status:</strong> {esc(status)}</p>
<h2>Output</h2>
<pre>{esc(log) if log else "(no predictions yet)"}</pre>
</body>
</html>
"""
    return body.encode("utf-8")


class AppState:
    def __init__(self) -> None:
        self.status = "Starting…"
        self.log = ""
        self.schedule: list[ScheduledGame] = []
        self.by_id: dict[str, ScheduledGame] = {}
        self.days = 7
        self.include_live = False


def main() -> None:
    root = bootstrap_cwd()
    try:
        games, artifact = load_context(root)
    except FileNotFoundError as e:
        print(e, file=__import__("sys").stderr)
        raise SystemExit(1) from e

    state = AppState()

    def do_refresh() -> None:
        state.status = "Loading prediction schedule…"
        try:
            rows = refresh_schedule_rows(state.days, state.include_live)
        except Exception as e:
            state.status = f"Schedule load failed: {e}"
            state.schedule = []
            state.by_id = {}
            return
        state.schedule = rows
        state.by_id = {g.game_id: g for g in rows}
        if not rows:
            state.status = "No prediction games in this window. Run fetch_data.py again or use Custom matchup."
        else:
            state.status = f"Loaded {len(rows)} game(s). Select checkboxes and click Predict selected."

    do_refresh()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _read_body(self) -> dict[str, list[str]]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            return parse_qs(raw, keep_blank_values=True)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in ("", "/"):
                self.send_error(404)
                return
            port = self.server.server_address[1]
            data = _page(
                status=state.status,
                log=state.log,
                rows=state.schedule,
                days=state.days,
                include_live=state.include_live,
                port=port,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            body = self._read_body()
            if parsed.path == "/refresh":
                try:
                    state.days = max(1, min(int(body.get("days", ["7"])[0] or 7), 60))
                except ValueError:
                    state.days = 7
                state.include_live = "live" in body
                do_refresh()
            elif parsed.path == "/predict":
                ids = body.get("gid", [])
                if not ids:
                    state.status = "Select at least one game, or use Custom matchup."
                else:
                    state.log += "\n---\n" if state.log else ""
                    for gid in ids:
                        g = state.by_id.get(str(gid))
                        if g is None:
                            state.log += f"[error] unknown game id {gid!r}\n"
                            continue
                        state.log += predict_scheduled_text(games, artifact, g)
                    state.status = f"Predicted {len(ids)} selection(s)."
            elif parsed.path == "/custom":
                away = (body.get("away", [""])[0] or "").strip()
                home = (body.get("home", [""])[0] or "").strip()
                gd = (body.get("game_date", [""])[0] or "").strip()
                if not away or not home or not gd:
                    state.status = "Fill Away, Home, and Date for custom predict."
                else:
                    state.log += "\n---\n" if state.log else ""
                    state.log += predict_custom_text(games, artifact, away, home, gd)
                    state.status = "Custom prediction appended."
            else:
                self.send_error(404)
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"Open {url} in your browser (or it may open automatically). Ctrl+C to quit.")

    def _open() -> None:
        webbrowser.open(url)

    threading.Timer(0.35, _open).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
