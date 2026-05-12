#!/usr/bin/env python3
"""Desktop UI: load upcoming NBA games and run the baseline predictor."""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd

from predict import resolve_user_team
from src.predict_gui_core import bootstrap_cwd, load_context, predict_custom_text, predict_scheduled_text, refresh_schedule_rows
from src.scoreboard_schedule import ScheduledGame


def main() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = bootstrap_cwd()
    games_path = root / "data" / "games.csv"
    prediction_path = root / "data" / "prediction_games.csv"
    model_path = root / "artifacts" / "baseline_logreg.pkl"

    if not games_path.exists():
        messagebox.showerror(
            "Missing data",
            f"Could not find {games_path}.\nRun fetch_data.py first.",
        )
        return
    if not prediction_path.exists():
        # Not fatal because predict_gui_core can fall back to ScoreboardV2.
        print(f"Warning: {prediction_path} not found. GUI will fall back to NBA ScoreboardV2.", file=sys.stderr)
    if not model_path.exists():
        messagebox.showerror(
            "Missing model",
            f"Could not find {model_path}.\nRun train.py after fetching games.",
        )
        return

    try:
        games, artifact = load_context(root)
    except FileNotFoundError as e:
        messagebox.showerror("Setup", str(e))
        return

    win = tk.Tk()
    win.title("NBA baseline — upcoming games")
    win.minsize(720, 520)

    top = ttk.Frame(win, padding=8)
    top.pack(fill=tk.X)

    ttk.Label(top, text="Days ahead:").pack(side=tk.LEFT)
    days_var = tk.StringVar(value="7")
    days_spin = ttk.Spinbox(top, from_=1, to=60, width=4, textvariable=days_var)
    days_spin.pack(side=tk.LEFT, padx=(4, 12))

    live_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(top, text="Include live games", variable=live_var).pack(side=tk.LEFT, padx=(0, 12))

    status_var = tk.StringVar(value="Load schedule to see upcoming games.")
    ttk.Label(win, textvariable=status_var).pack(fill=tk.X, padx=8, pady=(0, 4))

    table_frame = ttk.Frame(win, padding=(8, 0))
    table_frame.pack(fill=tk.BOTH, expand=True)

    cols = ("date", "matchup", "status")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended", height=14)
    tree.heading("date", text="Date")
    tree.heading("matchup", text="Matchup")
    tree.heading("status", text="Status")
    tree.column("date", width=100, anchor=tk.W)
    tree.column("matchup", width=220, anchor=tk.W)
    tree.column("status", width=160, anchor=tk.W)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scroll.set)

    schedule_rows: dict[str, ScheduledGame] = {}

    def refresh_schedule() -> None:
        tree.delete(*tree.get_children())
        schedule_rows.clear()
        try:
            n = int(days_var.get())
        except ValueError:
            n = 7
        status_var.set("Loading prediction schedule…")
        win.update_idletasks()
        try:
            rows = refresh_schedule_rows(n, live_var.get())
        except Exception as e:
            status_var.set("Schedule load failed.")
            messagebox.showerror("Schedule error", str(e))
            return
        for g in rows:
            iid = g.game_id
            schedule_rows[iid] = g
            tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(g.game_date.isoformat(), f"{g.away_abbr} @ {g.home_abbr}", g.status_text or "Scheduled"),
            )
        if not rows:
            status_var.set("No prediction games in this window. Run fetch_data.py again or use custom predict below.")
        else:
            status_var.set(f"Loaded {len(rows)} game(s). Select row(s), then Predict selection.")

    btn_bar = ttk.Frame(win, padding=8)
    btn_bar.pack(fill=tk.X)

    ttk.Button(btn_bar, text="Refresh schedule", command=refresh_schedule).pack(side=tk.LEFT)

    out = tk.Text(win, height=12, wrap=tk.WORD, state=tk.DISABLED)
    out.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def append_out(text: str) -> None:
        out.configure(state=tk.NORMAL)
        out.insert(tk.END, text)
        if not text.endswith("\n"):
            out.insert(tk.END, "\n")
        out.configure(state=tk.DISABLED)
        out.see(tk.END)

    def on_predict_selection() -> None:
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Predict", "Select one or more games in the table first.")
            return
        append_out("---")
        for iid in sel:
            g = schedule_rows.get(str(iid))
            if g is None:
                continue
            append_out(predict_scheduled_text(games, artifact, g).rstrip("\n"))

    ttk.Button(btn_bar, text="Predict selection", command=on_predict_selection).pack(side=tk.LEFT, padx=(8, 0))

    custom = ttk.LabelFrame(win, text="Custom matchup (any date)", padding=8)
    custom.pack(fill=tk.X, padx=8, pady=(0, 8))

    r1 = ttk.Frame(custom)
    r1.pack(fill=tk.X)
    ttk.Label(r1, text="Away").pack(side=tk.LEFT)
    away_e = ttk.Entry(r1, width=14)
    away_e.pack(side=tk.LEFT, padx=4)
    ttk.Label(r1, text="Home").pack(side=tk.LEFT, padx=(12, 0))
    home_e = ttk.Entry(r1, width=14)
    home_e.pack(side=tk.LEFT, padx=4)
    ttk.Label(r1, text="Date (YYYY-MM-DD)").pack(side=tk.LEFT, padx=(12, 0))
    date_e = ttk.Entry(r1, width=12)
    date_e.pack(side=tk.LEFT, padx=4)

    def on_predict_custom() -> None:
        try:
            resolve_user_team(home_e.get())
            resolve_user_team(away_e.get())
        except ValueError as e:
            messagebox.showerror("Custom predict", str(e))
            return
        try:
            pd.to_datetime(date_e.get().strip())
        except Exception as e:
            messagebox.showerror("Custom predict", f"Bad date: {e}")
            return
        append_out("---")
        append_out(predict_custom_text(games, artifact, away_e.get(), home_e.get(), date_e.get()).rstrip("\n"))

    ttk.Button(custom, text="Predict custom", command=on_predict_custom).pack(anchor=tk.W, pady=(6, 0))

    refresh_schedule()
    win.mainloop()


if __name__ == "__main__":
    if importlib.util.find_spec("_tkinter") is None:
        print(
            "This Python has no Tk bindings (_tkinter). "
            "For a native window on macOS (Homebrew): `brew install python-tk@3.14` "
            "(use the same major.minor as `python3 --version`).\n"
            "Starting browser UI instead…\n",
            file=sys.stderr,
        )
        import predict_gui_web

        predict_gui_web.main()
    else:
        main()
