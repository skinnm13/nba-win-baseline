import { featureVectorForGame } from "./features";
import { displayTeam } from "./teams";
import type { CompletedGame, ModelArtifact, PredictionResult } from "./types";
import { normalizeGame } from "./types";

function inferSeasonId(games: CompletedGame[], gameDate: string, seasonId?: string): string {
  if (seasonId) return seasonId;
  const ts = new Date(`${gameDate}T12:00:00`).getTime();
  const prior = games.filter((g) => new Date(`${g.GAME_DATE}T12:00:00`).getTime() < ts);
  if (prior.length === 0) {
    return games.reduce((best, g) => (g.SEASON_ID > best ? g.SEASON_ID : best), games[0].SEASON_ID);
  }
  prior.sort((a, b) => a.GAME_DATE.localeCompare(b.GAME_DATE));
  return prior[prior.length - 1].SEASON_ID;
}

function abbrToTeamId(games: CompletedGame[], abbr: string): number {
  const upper = abbr.toUpperCase();
  const matches = games.filter((g) => g.HOME_ABBR === upper || g.AWAY_ABBR === upper);
  if (matches.length === 0) {
    throw new Error(`No games for team ${upper} in training history`);
  }
  matches.sort((a, b) => a.GAME_DATE.localeCompare(b.GAME_DATE));
  const last = matches[matches.length - 1];
  return last.HOME_ABBR === upper ? last.HOME_TEAM_ID : last.AWAY_TEAM_ID;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function scoreFeatures(model: ModelArtifact, features: number[]): number {
  if (features.length !== model.featureColumns.length) {
    throw new Error("Feature column mismatch");
  }
  const { mean, scale } = model.scaler;
  const { coef, intercept } = model.classifier;
  let logit = intercept;
  for (let i = 0; i < features.length; i++) {
    const scaled = (features[i] - mean[i]) / scale[i];
    logit += coef[i] * scaled;
  }
  return sigmoid(logit);
}

export function predictMatchup(
  games: CompletedGame[],
  model: ModelArtifact,
  homeAbbr: string,
  awayAbbr: string,
  gameDate: string,
  seasonId?: string,
): PredictionResult {
  const home = homeAbbr.trim().toUpperCase();
  const away = awayAbbr.trim().toUpperCase();
  if (home === away) throw new Error("home and away teams cannot be the same");

  const homeId = abbrToTeamId(games, home);
  const awayId = abbrToTeamId(games, away);
  const sid = inferSeasonId(games, gameDate, seasonId);
  const predGid = `PRED_${gameDate.replace(/-/g, "")}_${home}_VS_${away}_${Math.random().toString(16).slice(2, 10)}`;

  const placeholder: CompletedGame = {
    GAME_ID: predGid,
    GAME_DATE: gameDate,
    SEASON_ID: sid,
    HOME_TEAM_ID: homeId,
    AWAY_TEAM_ID: awayId,
    HOME_ABBR: home,
    AWAY_ABBR: away,
    HOME_PTS: 0,
    AWAY_PTS: 0,
    HOME_WIN: 0,
  };

  const combined = [...games.map(normalizeGame), placeholder].sort((a, b) => {
    const d = a.GAME_DATE.localeCompare(b.GAME_DATE);
    return d !== 0 ? d : String(a.GAME_ID).localeCompare(String(b.GAME_ID));
  });

  const features = featureVectorForGame(combined, predGid);
  const pHome = scoreFeatures(model, features);
  const pAway = 1 - pHome;
  const pickSide: "home" | "away" = pHome >= 0.5 ? "home" : "away";

  return {
    pHome,
    pAway,
    pickAbbr: pickSide === "home" ? home : away,
    pickSide,
    homeAbbr: home,
    awayAbbr: away,
    homeLabel: displayTeam(home),
    awayLabel: displayTeam(away),
    gameDate,
    seasonId: sid,
  };
}

export function formatPrediction(res: PredictionResult): string {
  return (
    `${res.awayAbbr} @ ${res.homeAbbr}  ${res.gameDate}  (SEASON_ID=${res.seasonId})\n` +
    `  P(home win): ${res.pHome.toFixed(4)}  |  ${res.homeLabel}: ${res.pHome.toFixed(4)}  |  ` +
    `${res.awayLabel}: ${res.pAway.toFixed(4)}\n` +
    `  Pick (>0.5 home): ${res.pickAbbr} (${res.pickSide})`
  );
}
