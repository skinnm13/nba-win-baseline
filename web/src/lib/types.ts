export interface Team {
  id: number;
  abbreviation: string;
  fullName: string;
  nickname: string;
  city: string;
}

export interface CompletedGame {
  GAME_ID: string;
  GAME_DATE: string;
  SEASON_ID: string;
  HOME_TEAM_ID: number;
  AWAY_TEAM_ID: number;
  HOME_ABBR: string;
  AWAY_ABBR: string;
  HOME_PTS: number;
  AWAY_PTS: number;
  HOME_WIN: number;
}

export function normalizeGame(g: CompletedGame): CompletedGame {
  return {
    ...g,
    GAME_ID: String(g.GAME_ID),
    SEASON_ID: String(g.SEASON_ID),
    HOME_ABBR: String(g.HOME_ABBR).toUpperCase(),
    AWAY_ABBR: String(g.AWAY_ABBR).toUpperCase(),
  };
}

export interface ScheduledGame {
  GAME_ID: string;
  GAME_DATE: string;
  SEASON_ID?: string;
  HOME_ABBR: string;
  AWAY_ABBR: string;
  GAME_STATUS?: number;
  GAME_STATUS_TEXT?: string;
  IS_PREDICTION_ROW?: number;
}

export interface ModelArtifact {
  featureColumns: string[];
  testSeasonId?: string;
  trainingSource?: string;
  scaler: { mean: number[]; scale: number[] };
  classifier: { coef: number[]; intercept: number };
}

export interface Meta {
  exportedFrom: string;
  gamesCount: number;
  predictionGamesCount: number;
  testSeasonId?: string;
}

export interface PredictionResult {
  pHome: number;
  pAway: number;
  pickAbbr: string;
  pickSide: "home" | "away";
  homeAbbr: string;
  awayAbbr: string;
  homeLabel: string;
  awayLabel: string;
  gameDate: string;
  seasonId: string;
}

export interface AppData {
  meta: Meta;
  model: ModelArtifact;
  games: CompletedGame[];
  schedule: ScheduledGame[];
  teams: Team[];
}
