import type { AppData } from "./types";
import { normalizeGame } from "./types";

const base = import.meta.env.BASE_URL;

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${base}data/${path}`);
  if (!res.ok) {
    throw new Error(`Failed to load ${path} (${res.status}). Run: python scripts/export_web_data.py`);
  }
  return res.json() as Promise<T>;
}

export async function loadAppData(): Promise<AppData> {
  const [meta, model, games, schedule, teams] = await Promise.all([
    fetchJson<AppData["meta"]>("meta.json"),
    fetchJson<AppData["model"]>("model.json"),
    fetchJson<AppData["games"]>("games.json"),
    fetchJson<AppData["schedule"]>("prediction_games.json"),
    fetchJson<AppData["teams"]>("teams.json"),
  ]);
  return { meta, model, games: games.map(normalizeGame), schedule, teams };
}
